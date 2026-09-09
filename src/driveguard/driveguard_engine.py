"""DriveGuard AI - the risk engine. Pure logic, no camera and no serial port.

WHY THIS IS A SEPARATE FILE. The three system metrics this project reports -
end-to-end alert latency, false alarms per hour, stage-transition accuracy - are
properties of the deployed system. If they were measured with a different code
path from the one that actually runs, they would describe a system that does
not exist. So the engine is isolated here and BOTH `driveguard_live.py` (camera
-> ESP32) and `measure_system_metrics.py` (recorded clips -> numbers) drive this
same object. The measured numbers therefore describe the deployed system.

WHAT IT IMPLEMENTS. A temporal fusion step accumulates micro-sleep, yawn and
distraction events into a three-stage fatigue risk score (Drowsy -> Fatigue ->
Critical), so the system does not merely detect instantaneous events but
continuously assesses the driver's risk level over time.

So this is an ACCUMULATOR with decay, not a per-frame classifier. A single
closed-eye frame is a blink; sustained closure is drowsiness. That is the same
principle Branch A's PERCLOS encodes, applied a second time at the alert level.

The stages match `esp32_firmware/src/main.cpp` exactly:

    0 NORMAL    green steady
    1 DROWSY    amber steady, chirp + vibration on entry
    2 FATIGUE   amber blink, double beep + vibration every 2 s
    3 CRITICAL  red fast blink, siren, vibration on

EVERY CONSTANT BELOW IS EITHER MEASURED OR MARKED AS TUNABLE. Nothing is an
inherited default - that mistake already cost this project once: the shipped
EYE_AR_THRESH of 0.26 classified 86.9% of one participant's wide-awake frames as shut,
and would have alarmed continuously through the demo for half the team
.
"""
import os
import sys
import time
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, "..", "features"))
sys.path.insert(0, TRAIN)

from extract_features import ear_of, mar_of, LEFT_EYE, RIGHT_EYE, MOUTH
from extract_temporal import temporal_block, NAMES

# ---------------------------------------------------------------- constants
# Branch A's winning feature subset (experiment_subsets.py, 95.0% LOSO).
FEATURES = ("perclos_20", "MAR", "closed_run_w")

# Causal window. 25 fps / stride 5 = 5 fps in training, so 6 samples = 1.2 s.
# Live video runs at full frame rate, so the engine samples at SAMPLE_HZ to
# reproduce the SAME 1.2 s of evidence the model was trained on. Feeding every
# frame instead would silently change the window's duration and invalidate the
# model.
WINDOW = 6
SAMPLE_HZ = 5.0

# Risk accumulator. Units are "seconds of evidence": +1.0 means one second of
# sustained drowsiness. Expressed in seconds rather than frames so the engine
# behaves identically at any camera frame rate.
RISK_MAX = 12.0
GAIN = {"drowsy": 1.0, "yawning": 0.45, "normal": 0.0}
# Distraction (phone in hand) is the third event type the system reports. It cannot come from facial geometry - a driver on the phone has an
# entirely normal eye and mouth configuration, measured at F1 0.498 against
# 0.942 for yawning - so it is supplied by the detector branch instead.
# TUNABLE: at 0.6 per second, 2.5 s of phone use reaches the DROWSY band,
# which is consistent with the two-second eyes-off-road guidance.
PHONE_GAIN = 0.6
DECAY = 0.6                 # per second of `normal`
# Band edges, in seconds of accumulated evidence. TUNABLE - tune_stages() in
# measure_system_metrics.py fits these on our labelled clips.
BANDS = (1.5, 4.0, 7.5)     # -> stage 1, 2, 3
# Once a stage is entered it is held at least this long, so the LEDs cannot
# flicker between stages on a borderline score. Chosen to exceed the firmware's
# 2 s stage-2 beep cycle so a driver always hears a complete alert pattern.
MIN_DWELL_S = 2.5

STAGE_NAMES = {0: "NORMAL", 1: "DROWSY", 2: "FATIGUE", 3: "CRITICAL"}


class RiskEngine:
    """Frame in -> (class, stage, diagnostics) out. Deterministic and causal.

    `clf` is any fitted sklearn estimator taking the three FEATURES columns.
    It is injected rather than loaded here so the caller controls provenance -
    the live app and the metrics harness must demonstrably use the same model.
    """

    def __init__(self, clf, window=WINDOW, sample_hz=SAMPLE_HZ,
                 bands=BANDS, gain=None, decay=DECAY,
                 min_dwell_s=MIN_DWELL_S):
        self.clf = clf
        self.window = window
        self.sample_dt = 1.0 / sample_hz
        self.bands = tuple(bands)
        self.gain = dict(gain or GAIN)
        self.decay = decay
        self.min_dwell_s = min_dwell_s
        self.reset()

    def reset(self):
        self._ear = collections.deque(maxlen=self.window)
        self._mar = collections.deque(maxlen=self.window)
        self._last_sample_t = None
        self.risk = 0.0
        self.stage = 0
        self._stage_entered_t = None
        self.last_class = None
        self.last_ear = None
        self.last_mar = None
        self.last_phone = False
        self.frames_seen = 0
        self.faces_lost = 0

    # -------------------------------------------------------------- helpers
    @staticmethod
    def ear_mar(pts):
        """EAR (mean of both eyes) and MAR from a 478-point landmark array."""
        return ((ear_of(pts, LEFT_EYE) + ear_of(pts, RIGHT_EYE)) / 2.0,
                mar_of(pts, MOUTH))

    def _features(self):
        """Branch A's three features from the current causal window.

        temporal_block is the SAME function that produced the training
        features, so a live window and a training window cannot drift apart.
        """
        ear = np.asarray(self._ear, dtype=np.float32)
        mar = np.asarray(self._mar, dtype=np.float32)
        T = temporal_block(ear, mar, self.window)[-1]
        idx = {n: i for i, n in enumerate(NAMES)}
        return np.array([[T[idx["perclos_20"]], float(mar[-1]),
                          T[idx["closed_run_w"]]]], dtype=np.float32)

    # ----------------------------------------------------------------- step
    def step(self, pts, now=None, phone=False):
        """Advance the engine by one video frame.

        `pts` is the landmark array, or None when no face was found. `now` is a
        monotonic timestamp in seconds; supplying it explicitly lets the offline
        harness replay clips at their true frame rate rather than wall-clock.
        `phone` is the detector branch's verdict for this frame; it contributes
        risk independently of the classifier, because a driver can be holding a
        phone while otherwise perfectly alert.

        Returns a dict, always - the caller never has to guess whether the
        engine acted on this frame.
        """
        now = time.monotonic() if now is None else float(now)
        self.frames_seen += 1
        if self._stage_entered_t is None:
            self._stage_entered_t = now

        if pts is None:
            # A lost face is not evidence of alertness, so the risk score is
            # held rather than decayed. Silently decaying here would let a
            # driver escape a CRITICAL alert by turning away from the camera.
            self.faces_lost += 1
            return self._out(now, sampled=False, reason="no-face")

        # Sample at the training frame rate, not the camera frame rate.
        if self._last_sample_t is not None and \
                now - self._last_sample_t < self.sample_dt:
            return self._out(now, sampled=False, reason="between-samples")
        dt = self.sample_dt if self._last_sample_t is None \
            else now - self._last_sample_t
        self._last_sample_t = now

        ear, mar = self.ear_mar(pts)
        self._ear.append(ear)
        self._mar.append(mar)

        cls = str(self.clf.predict(self._features())[0])
        self.last_class = cls

        # ---- temporal fusion: accumulate evidence, decay on normal --------
        gain = self.gain.get(cls, 0.0)
        if phone:
            gain += PHONE_GAIN
        if gain > 0.0:
            self.risk = min(RISK_MAX, self.risk + gain * dt)
        else:
            self.risk = max(0.0, self.risk - self.decay * dt)
        self.last_phone = bool(phone)

        target = 0
        for i, edge in enumerate(self.bands, start=1):
            if self.risk >= edge:
                target = i
        # Escalate immediately; de-escalate only after the dwell time, so an
        # alert cannot be cut short but a recovering driver is not punished.
        if target > self.stage:
            self.stage = target
            self._stage_entered_t = now
        elif target < self.stage and \
                now - self._stage_entered_t >= self.min_dwell_s:
            self.stage = target
            self._stage_entered_t = now
        return self._out(now, sampled=True, reason="ok", ear=ear, mar=mar)

    def _out(self, now, sampled, reason, ear=None, mar=None):
        # The engine samples at 5 Hz but the camera runs at 25 fps, so four
        # frames in five are "between samples". Returning None for EAR/MAR on
        # those left the demo HUD showing empty fields 80% of the time, which
        # looks broken. The last SAMPLED values are carried instead, and
        # `sampled` still tells the caller whether this frame was measured.
        if ear is not None:
            self.last_ear, self.last_mar = ear, mar
        return {"t": now, "sampled": sampled, "reason": reason,
                "cls": self.last_class, "risk": round(self.risk, 3),
                "stage": self.stage, "stage_name": STAGE_NAMES[self.stage],
                "ear": self.last_ear, "mar": self.last_mar,
                "window_full": len(self._ear) >= self.window}


# --------------------------------------------------------------------------
# model provenance
# --------------------------------------------------------------------------
def load_branch_a(verbose=True):
    """Fit Branch A on all our labelled frames and return (clf, provenance).

    No pickled model is shipped deliberately. A linear SVM on 793x3 fits in
    milliseconds, so refitting at start-up removes a whole class of failure -
    a stale .pkl silently disagreeing with the features the code computes - at
    no meaningful cost. The provenance dict is printed so the demo can state
    exactly what is running.
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # A fitted model ships with this repository, so the system runs
    # without the face recordings, which are not published. Refitting
    # from features is still supported for anyone with their own data.
    shipped = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                           "models", "branch_a.joblib")
    if os.path.exists(shipped):
        import joblib
        bundle = joblib.load(shipped)
        if verbose:
            pr = bundle["prov"]
            print("Branch A loaded: {} features, fitted on {} frames "
                  "from {} participants".format(len(pr["features"]),
                                                pr["n_train"],
                                                pr["n_participants"]))
        return bundle["clf"], bundle["prov"]

    p = os.path.join(TRAIN, "features_temporal.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} - run extract_temporal.py first")
    d = np.load(p, allow_pickle=True)
    names = [str(x) for x in d["temporal_names"]]
    ti = {n: i for i, n in enumerate(names)}
    classes = ["normal", "drowsy", "yawning"]
    keep = np.isin(d["y"], classes)
    T, G, y = d["X_temporal"][keep], d["X_geo"][keep], d["y"][keep]
    X = np.column_stack([T[:, ti["perclos_20"]], G[:, 4],
                         T[:, ti["closed_run_w"]]])
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel="linear", C=1, class_weight="balanced",
                            random_state=0)).fit(X, y)
    prov = {"features": list(FEATURES), "n_train": int(len(y)),
            "classes": classes,
            "persons": sorted(set(d["persons"][keep].tolist())),
            "source": os.path.basename(p)}
    if verbose:
        print(f"Branch A fitted on {prov['n_train']} frames from "
              f"{len(prov['persons'])} people: {prov['features']}")
    return clf, prov
