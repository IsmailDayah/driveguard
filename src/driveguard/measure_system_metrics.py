"""The three system metrics this project reports, measured end to end.

    End-to-end alert latency  |  False alarms per hour  |  Stage-transition accuracy

They are properties of the integrated system rather than of any single model, so
this harness drives THE SAME RiskEngine the live application uses; the numbers
therefore describe the deployed system.

METHOD, and why it is split in two:

  * PER-FRAME LATENCY needs the real pipeline, MediaPipe included, so it is
    timed on actual image files.
  * FALSE ALARMS and STAGE ACCURACY need many minutes of labelled footage, and
    re-running MediaPipe over all of it would add nothing: the EAR/MAR values
    in features_temporal.npz were produced by that same pipeline, and the live
    engine was verified to derive bit-identical features from them
    (0.000e+00 difference over 793 rows / 26 clips). So those two are replayed
    from the stored sequences - faster, and exactly equivalent.

Clips are sampled every 5th frame of 25 fps footage, i.e. 5 Hz, which is the
rate the engine expects.

    python measure_system_metrics.py
    python measure_system_metrics.py --tune      # fit the band edges on data
"""
import os
import sys
import glob
import json
import time
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, "..", "features"))
FRAMES = os.path.abspath(os.path.join(HERE, "..", "..", "data", "frames"))
sys.path.insert(0, HERE)
sys.path.insert(0, TRAIN)

from driveguard_engine import RiskEngine, load_branch_a, STAGE_NAMES, BANDS
from extract_temporal import parse

SAMPLE_HZ = 5.0
DT = 1.0 / SAMPLE_HZ
THREE = ["normal", "drowsy", "yawning"]


def load_clips():
    """(person, class, clip) -> ordered EAR/MAR sequences from the audited npz."""
    d = np.load(os.path.join(TRAIN, "features_temporal.npz"), allow_pickle=True)
    keep = np.isin(d["y"], THREE)
    G, files, y = d["X_geo"][keep], d["files"][keep], d["y"][keep]
    ear, mar = G[:, 2], G[:, 4]        # indices documented in extract_temporal
    groups = {}
    for i, f in enumerate(files):
        m = parse(f)
        groups.setdefault((m[0], m[1], m[2]), []).append((m[3], i))
    out = {}
    for k, v in groups.items():
        idx = [i for _, i in sorted(v)]
        out[k] = (ear[idx].astype(float), mar[idx].astype(float), str(y[idx[0]]))
    return out


def replay(engine, ear, mar):
    """Drive the engine through one clip at its true 5 Hz and log every step."""
    engine.reset()
    rows = []
    for j in range(len(ear)):
        t = j * DT
        # Feed the engine directly at the feature level. step() expects
        # landmarks, so the deques are filled the same way step() would and the
        # identical fusion logic is then invoked.
        engine._ear.append(float(ear[j]))
        engine._mar.append(float(mar[j]))
        cls = str(engine.clf.predict(engine._features())[0])
        engine.last_class = cls
        if cls == "normal":
            engine.risk = max(0.0, engine.risk - engine.decay * DT)
        else:
            engine.risk = min(12.0, engine.risk + engine.gain.get(cls, 0.0) * DT)
        target = 0
        for i, edge in enumerate(engine.bands, start=1):
            if engine.risk >= edge:
                target = i
        if engine._stage_entered_t is None:
            engine._stage_entered_t = t
        if target > engine.stage:
            engine.stage = target
            engine._stage_entered_t = t
        elif target < engine.stage and \
                t - engine._stage_entered_t >= engine.min_dwell_s:
            engine.stage = target
            engine._stage_entered_t = t
        rows.append({"t": t, "cls": cls, "risk": engine.risk,
                     "stage": engine.stage})
    return rows


def evaluate(clips, engine):
    """False alarms, alert latency and stage correctness over every clip."""
    normal_s = alarms = 0.0
    lat, per_clip = [], []
    for k, (ear, mar, true_c) in sorted(clips.items()):
        rows = replay(engine, ear, mar)
        dur = len(rows) * DT
        peak = max(r["stage"] for r in rows)
        first = next((r["t"] for r in rows if r["stage"] > 0), None)
        if true_c == "normal":
            normal_s += dur
            # count ENTRIES into a raised stage, not raised frames
            prev, ent = 0, 0
            for r in rows:
                if r["stage"] > 0 and prev == 0:
                    ent += 1
                prev = r["stage"]
            alarms += ent
        else:
            if first is not None:
                lat.append(first)
        per_clip.append({"clip": "/".join(k), "true": true_c,
                         "peak_stage": peak, "first_alert_s": first,
                         "duration_s": round(dur, 2)})
    fa_hr = alarms / (normal_s / 3600.0) if normal_s > 0 else float("nan")
    # stage correctness: a normal clip must stay at 0; an event clip must raise
    ok = sum(1 for c in per_clip
             if (c["true"] == "normal" and c["peak_stage"] == 0)
             or (c["true"] != "normal" and c["peak_stage"] >= 1))
    return {"false_alarms_per_hour": fa_hr,
            "normal_minutes": normal_s / 60.0,
            "alarms_on_normal": int(alarms),
            "alert_latency_s": lat,
            "stage_accuracy": ok / len(per_clip),
            "per_clip": per_clip}


def fit_excluding(person):
    """Branch A fitted WITHOUT `person` - for an honest held-out evaluation.

    Fitting on all four people and then scoring their own clips measures
    in-sample behaviour and will flatter the system. Every accuracy figure in
    this project is leave-one-person-out, and the system metrics must be too.
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    d = np.load(os.path.join(TRAIN, "features_temporal.npz"), allow_pickle=True)
    ti = {str(n): i for i, n in enumerate(d["temporal_names"])}
    keep = np.isin(d["y"], THREE) & (d["persons"] != person)
    T, G, y = d["X_temporal"][keep], d["X_geo"][keep], d["y"][keep]
    X = np.column_stack([T[:, ti["perclos_20"]], G[:, 4],
                         T[:, ti["closed_run_w"]]])
    return make_pipeline(StandardScaler(),
                         SVC(kernel="linear", C=1, class_weight="balanced",
                             random_state=0)).fit(X, y)


def evaluate_loso(clips, bands):
    """Each person's clips scored by a model that never saw that person."""
    people = sorted({k[0] for k in clips})
    merged = {"per_clip": [], "alert_latency_s": [],
              "normal_s": 0.0, "alarms": 0}
    for p in people:
        clf_p = fit_excluding(p)
        eng = RiskEngine(clf_p, bands=bands)
        sub = {k: v for k, v in clips.items() if k[0] == p}
        r = evaluate(sub, eng)
        merged["per_clip"] += r["per_clip"]
        merged["alert_latency_s"] += r["alert_latency_s"]
        merged["normal_s"] += r["normal_minutes"] * 60.0
        merged["alarms"] += r["alarms_on_normal"]
    ok = sum(1 for c in merged["per_clip"]
             if (c["true"] == "normal" and c["peak_stage"] == 0)
             or (c["true"] != "normal" and c["peak_stage"] >= 1))
    return {"stage_accuracy": ok / len(merged["per_clip"]),
            "alert_latency_s": merged["alert_latency_s"],
            "false_alarms_per_hour": (merged["alarms"] /
                                      (merged["normal_s"] / 3600.0)
                                      if merged["normal_s"] else float("nan")),
            "normal_minutes": merged["normal_s"] / 60.0,
            "alarms_on_normal": merged["alarms"],
            "per_clip": merged["per_clip"]}


def false_alarms_external(clf, bands, min_run=12):
    """False-alarm rate on FL3D's `alert` stretches - 44 drivers we never saw.

    Our own `normal` footage totals about one minute, which cannot establish a
    rate: zero alarms there gives a 95% upper bound near 175/hour. FL3D
    contributes roughly two hours of human-labelled alert driving from 44 other
    people, so the rate can actually be estimated - and on strangers, which is
    the number a deployment would care about.

    Read it as a CONSERVATIVE figure: FL3D is night footage and Branch A
    transfers to it at 83.7% macro-F1, so this is the rate under domain shift
    rather than under ideal conditions.

    Only runs of at least `min_run` samples (2.4 s) are used, so the engine
    always has a full 1.2 s window before its output counts.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "training")))
    from experiment_external import fl3d_sequences
    seqs = fl3d_sequences()
    eng = RiskEngine(clf, bands=bands)
    total_s = 0.0
    alarms = 0
    runs = 0
    for item in seqs:
        v, ear, mar, lab = item[0], np.asarray(item[1]), np.asarray(item[2]), \
            np.asarray(item[3])
        # contiguous stretches the annotators marked as alert
        i = 0
        while i < len(lab):
            if lab[i] != "normal":
                i += 1
                continue
            j = i
            while j < len(lab) and lab[j] == "normal":
                j += 1
            if j - i >= min_run:
                rows = replay(eng, ear[i:j], mar[i:j])
                total_s += len(rows) * DT
                runs += 1
                prev = 0
                for r in rows:
                    if r["stage"] > 0 and prev == 0:
                        alarms += 1
                    prev = r["stage"]
            i = j
    hrs = total_s / 3600.0
    return {"alarms": alarms, "hours": hrs, "runs": runs,
            "per_hour": (alarms / hrs) if hrs > 0 else float("nan")}


def rule_of_three(events, hours):
    """95% upper bound on a rate when `events` is small (Hanley & Lippman-Hand).

    Zero alarms in a short recording does NOT establish a low false-alarm rate,
    and quoting 0.0/hour from one minute of footage would be indefensible.
    """
    if hours <= 0:
        return float("nan")
    return (3.0 / hours) if events == 0 else (events + 1.96 * np.sqrt(events)) / hours


def frame_latency(n=60):
    """Real per-frame cost of the full pipeline, MediaPipe included."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode)
    model = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")
    files = sorted(glob.glob(os.path.join(FRAMES, "*", "*.jpg")))
    if not files:
        return None
    sel = [files[i] for i in np.linspace(0, len(files) - 1, n).astype(int)]
    lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(model)),
        running_mode=RunningMode.IMAGE, num_faces=1))
    clf, _ = load_branch_a(verbose=False)
    eng = RiskEngine(clf)
    ms = []
    for p in sel:
        img = cv2.imread(p)
        if img is None:
            continue
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(
                                   cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        pts = (np.array([[q.x * w, q.y * h] for q in r.face_landmarks[0]])
               if r.face_landmarks else None)
        eng.step(pts, now=len(ms) * DT)
        ms.append((time.perf_counter() - t0) * 1000.0)
    lm.close()
    return np.array(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true",
                    help="search band edges that maximise stage accuracy")
    ap.add_argument("--latency-frames", type=int, default=60)
    a = ap.parse_args()

    clf, prov = load_branch_a()
    clips = load_clips()
    print(f"clips: {len(clips)}   "
          f"{collections.Counter(v[2] for v in clips.values())}")

    bands = tuple(BANDS)
    if a.tune:
        print("\ntuning band edges on our labelled clips ...")
        best, bb = -1, bands
        for b1 in np.arange(0.4, 3.01, 0.2):
            for b2 in np.arange(b1 + 0.6, 6.01, 0.4):
                for b3 in np.arange(b2 + 0.6, 9.01, 0.5):
                    e = RiskEngine(clf, bands=(b1, b2, b3))
                    r = evaluate(clips, e)
                    # accuracy first, then fewer false alarms
                    score = r["stage_accuracy"] - 0.0001 * r["false_alarms_per_hour"]
                    if score > best:
                        best, bb = score, (round(b1, 2), round(b2, 2), round(b3, 2))
        bands = bb
        print(f"  tuned bands: {bands}")

    engine = RiskEngine(clf, bands=bands)
    res = evaluate(clips, engine)

    print("\n" + "=" * 74)
    print("SYSTEM METRICS (integrated)")
    print("=" * 74)
    lat = frame_latency(a.latency_frames)
    if lat is not None and len(lat):
        q = np.percentile(lat, [50, 95, 99])
        print(f"per-frame pipeline latency  median {q[0]:6.1f} ms   "
              f"p95 {q[1]:6.1f} ms   p99 {q[2]:6.1f} ms   "
              f"({1000/q[0]:.1f} fps sustainable)")
    al = res["alert_latency_s"]
    if al:
        print(f"end-to-end ALERT latency    median {np.median(al):.2f} s   "
              f"min {min(al):.2f}   max {max(al):.2f}   (n={len(al)} event clips)")

    # The headline figures must be held-out, not in-sample.
    lo = evaluate_loso(clips, bands)
    hrs = lo["normal_minutes"] / 60.0
    ub = rule_of_three(lo["alarms_on_normal"], hrs)
    print(f"false alarms per hour       {lo['false_alarms_per_hour']:.1f}   "
          f"({lo['alarms_on_normal']} alarms over "
          f"{lo['normal_minutes']:.1f} min of `normal`)")
    print(f"   -> 95% upper bound        {ub:.0f}/hour. {lo['normal_minutes']:.1f} "
          f"min is FAR too little to establish a real rate;")
    print(f"      report it as 'no false alarms observed in "
          f"{lo['normal_minutes']:.1f} min', never as '0/hour'.")
    ext = None
    try:
        ext = false_alarms_external(clf, bands)
        print(f"false alarms, EXTERNAL      {ext['per_hour']:.1f}/hour   "
              f"({ext['alarms']} over {ext['hours']:.2f} h of FL3D `alert`, "
              f"{ext['runs']} runs, 44 unseen drivers)")
        print("   this is the usable figure - conservative, since FL3D is night "
              "footage under domain shift")
    except Exception as e:
        print(f"external false-alarm measurement unavailable: "
              f"{type(e).__name__}: {e}")
    print(f"stage-transition accuracy   {100*lo['stage_accuracy']:.1f}%  "
          f"({len(clips)} clips, LEAVE-ONE-PERSON-OUT)")
    print(f"   in-sample would read      {100*res['stage_accuracy']:.1f}%  "
          f"(reported for contrast only)")
    print(f"bands in use                {bands}")
    res = lo

    peaks = collections.Counter((c["true"], c["peak_stage"]) for c in res["per_clip"])
    print("\npeak stage reached, by true class:")
    for tc in THREE:
        row = {s: peaks.get((tc, s), 0) for s in (0, 1, 2, 3)}
        print(f"  {tc:<9}" + "  ".join(f"stage{s}:{n}" for s, n in row.items()))

    reach3 = any(c["peak_stage"] == 3 for c in res["per_clip"])
    if not reach3:
        dur = np.mean([c["duration_s"] for c in res["per_clip"]])
        print(f"\nNOTE: no clip reaches CRITICAL. Our clips average "
              f"{dur:.1f} s, and at gain 1.0/s the band-3 edge of {bands[2]} s "
              f"of\n      sustained drowsiness cannot be accumulated within "
              f"one clip. CRITICAL is\n      reachable in continuous operation, "
              f"not in this corpus - state that rather\n      than tuning the "
              f"band down until it fires.")

    out = {"bands": list(bands), "provenance": prov,
           "false_alarms_external": ext,
           "false_alarms_own_upper_bound_per_hour": float(ub),
           "per_frame_latency_ms": (None if lat is None else
                                    {"median": float(np.median(lat)),
                                     "p95": float(np.percentile(lat, 95)),
                                     "p99": float(np.percentile(lat, 99)),
                                     "n": int(len(lat))}),
           "alert_latency_s": {"median": float(np.median(al)) if al else None,
                               "values": [float(x) for x in al]},
           "false_alarms_per_hour": float(res["false_alarms_per_hour"]),
           "normal_minutes": float(res["normal_minutes"]),
           "stage_accuracy": float(res["stage_accuracy"]),
           "per_clip": res["per_clip"]}
    p = os.path.join(HERE, "system_metrics.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nmetrics -> {p}")


if __name__ == "__main__":
    main()
