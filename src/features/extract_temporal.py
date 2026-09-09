"""Temporal features for Branch A - the signal we were throwing away.

Branch A scores 90.3% but its only weak class is `drowsy` (recall 80.3%), and
**62% of all its errors are missed drowsy frames** (27 called normal, 21 called
yawning). The cause is structural, not tuning: a single frame cannot express
drowsiness.

  A closed eye in ONE frame is a blink.
  A closed eye sustained over a second is drowsiness.

The live detector already knows this - it requires EYE_AR_CONSEC_FRAMES = 15
before flagging a micro-sleep - but the classifier we score has never been
given that information. When that constant was audited, run-length separated
the classes **perfectly**: normal clips reached at most 10 consecutive closed
video-frames, drowsy clips a median of 142, with 0% overlap. Branch A is blind
to the one signal already shown to separate cleanly.

Two deliberate design choices:

CAUSAL WINDOWS ONLY. Each frame's window looks BACKWARD only. A centred window
would use future frames, which a real-time system can never have; the offline
number would then overstate what the deployed detector can do.

MULTI-THRESHOLD PERCLOS, NOT CALIBRATION. Measured open-eye EAR spans
0.232 to 0.318 across participants, so any single "closed" cutoff is wrong for
somebody. Per-driver calibration was already tested and HURT the SVM
(90.3 -> 88.0), because it discarded information the model was using. So
PERCLOS is computed at several fixed thresholds and the SVM decides how to
weight them - no calibration, no arbitrary single cutoff.

Windows never cross a clip boundary: frames are grouped by (person, class,
clip) and ordered by frame index.

    python extract_temporal.py                 # writes features_temporal.npz
    python extract_temporal.py --window 8
"""
import os
import re
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# indices into X_geo, from extract_features.py:
#   0 EAR_L, 1 EAR_R, 2 EAR_mean, 3 |EAR_L-R|, 4 MAR, 5 MAR/EAR, 6 face h/w
EAR_I, MAR_I = 2, 4

# PERCLOS thresholds. Deliberately several: no single value suits every driver.
PERCLOS_T = (0.15, 0.20, 0.25)
RUN_T = 0.20            # same as the audited live-detector EYE_AR_THRESH

NAMES = ["perclos_15", "perclos_20", "perclos_25",
         "ear_min_w", "ear_mean_w", "ear_std_w", "ear_slope_w",
         "closed_run_w", "mar_max_w", "mar_mean_w", "mar_std_w",
         "window_frac",
         # --- self-normalising variants -------------------------------------
         # perclos_20 counts frames below an ABSOLUTE EAR of 0.20, a cutoff
         # taken from four young men in one room. Open-eye EAR already spans
         # 0.232-0.318 WITHIN those four, so the threshold cannot survive
         # transfer to strangers - and external validation showed exactly that:
         # yawning transferred at precision 1.000 (MAR is a self-normalising
         # ratio) while drowsy precision collapsed to 0.329.
         #
         # These express closure as a fraction of THIS driver's own open-eye
         # aperture, estimated causally from an expanding window, so no
         # enrolment step and no future frames are required.
         "perclos_rel60", "perclos_rel70", "perclos_rel80",
         "ear_over_base", "closed_run_rel"]


def parse(fname):
    """p01_normal_1_f00035.jpg -> ('p01', 'normal', '1', 35)"""
    b = os.path.splitext(os.path.basename(str(fname)))[0]
    m = re.match(r"^(.+?)_(normal|drowsy|yawning|phone)_(\w+?)_f(\d+)$", b)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), int(m.group(4))


BASE_PCTL = 90          # the eye is open most of the time, so a high
BASE_MIN = 6            # percentile of recent EAR approximates "open"


def _baselines(ear):
    """Causal per-frame estimate of this driver's OPEN-eye EAR.

    Expanding window: baseline at frame i uses only frames 0..i, so it is
    usable in real time and cannot see the future. Before BASE_MIN frames the
    estimate is unreliable, so the sequence maximum so far is used instead of a
    percentile of almost nothing.
    """
    n = len(ear)
    b = np.empty(n, np.float32)
    for i in range(n):
        seg = ear[:i + 1]
        b[i] = seg.max() if len(seg) < BASE_MIN else np.percentile(seg, BASE_PCTL)
    return np.maximum(b, 1e-6)


def temporal_block(ear, mar, w):
    """Causal window features for one clip's ordered (ear, mar) sequences."""
    n = len(ear)
    base = _baselines(ear)
    out = np.zeros((n, len(NAMES)), np.float32)
    for i in range(n):
        lo = max(0, i - w + 1)          # trailing window, inclusive of i
        e = ear[lo:i + 1]
        m = mar[lo:i + 1]
        k = len(e)
        col = []
        for t in PERCLOS_T:
            col.append(float((e < t).mean()))
        col.append(float(e.min()))
        col.append(float(e.mean()))
        col.append(float(e.std()) if k > 1 else 0.0)
        # slope over the window: positive = eyes opening
        if k > 1:
            col.append(float(np.polyfit(np.arange(k), e, 1)[0]))
        else:
            col.append(0.0)
        # longest run of consecutive closed frames ending anywhere in window
        best = cur = 0
        for v in e:
            cur = cur + 1 if v < RUN_T else 0
            best = max(best, cur)
        col.append(float(best) / max(w, 1))          # normalised 0..1
        col.append(float(m.max()))
        col.append(float(m.mean()))
        col.append(float(m.std()) if k > 1 else 0.0)
        # how full the window is - lets the model discount early frames
        col.append(float(k) / max(w, 1))

        # ---- self-normalising: closure relative to THIS driver's baseline ---
        bb = base[lo:i + 1]
        ratio = e / bb                      # 1.0 = fully open for this person
        for frac in (0.60, 0.70, 0.80):
            col.append(float((ratio < frac).mean()))
        col.append(float(ratio[-1]))        # instantaneous relative aperture
        best_r = cur_r = 0
        for v in ratio:
            cur_r = cur_r + 1 if v < 0.70 else 0
            best_r = max(best_r, cur_r)
        col.append(float(best_r) / max(w, 1))
        out[i] = col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6,
                    help="causal window length in SAMPLED frames. Frames were "
                         "kept every 5th from 25 fps video, so 6 sampled "
                         "frames span 30 video frames = 1.2 s")
    ap.add_argument("--out", default=os.path.join(HERE, "features_temporal.npz"))
    a = ap.parse_args()

    d = np.load(os.path.join(HERE, "features.npz"), allow_pickle=True)
    X, y, persons, files = d["X_geo"], d["y"], d["persons"], d["files"]
    n = len(y)
    print(f"loaded {n} frames, {X.shape[1]} geometric features")

    meta = [parse(f) for f in files]
    bad = [f for f, m in zip(files, meta) if m is None]
    if bad:
        raise SystemExit(f"could not parse {len(bad)} filenames, e.g. {bad[:3]}")

    clips = collections.defaultdict(list)
    for i, (p, c, clip, fi) in enumerate(meta):
        clips[(p, c, clip)].append((fi, i))

    T = np.zeros((n, len(NAMES)), np.float32)
    lens = []
    for key, lst in clips.items():
        lst.sort()                                  # temporal order
        idx = np.array([i for _, i in lst])
        lens.append(len(idx))
        T[idx] = temporal_block(X[idx, EAR_I], X[idx, MAR_I], a.window)

    print(f"clips: {len(clips)}   frames/clip min {min(lens)} "
          f"median {int(np.median(lens))} max {max(lens)}")
    print(f"window {a.window} sampled frames "
          f"= {a.window*5} video frames = {a.window*5/25:.1f}s at 25 fps")
    if a.window > min(lens):
        print(f"  NOTE: window exceeds the shortest clip ({min(lens)} frames); "
              f"those frames get a partial window, flagged by window_frac")

    XT = np.hstack([X, T]).astype(np.float32)
    np.savez_compressed(a.out, X_geo=X, X_temporal=T, X_all=XT,
                        y=y, persons=persons, files=files,
                        temporal_names=np.array(NAMES), window=a.window)
    print(f"\nsaved {a.out}")
    print(f"  X_geo      {X.shape}")
    print(f"  X_temporal {T.shape}")
    print(f"  X_all      {XT.shape}")

    # sanity: do the new features actually separate drowsy from normal?
    print(f"\n{'feature':<14}{'normal':>10}{'drowsy':>10}{'yawning':>10}"
          f"{'d(drowsy,normal)':>18}")
    print("-" * 62)
    for j, nm in enumerate(NAMES):
        v = {c: T[y == c, j] for c in ("normal", "drowsy", "yawning")}
        a_, b_ = v["drowsy"], v["normal"]
        s = np.sqrt(((len(a_)-1)*a_.std(ddof=1)**2 + (len(b_)-1)*b_.std(ddof=1)**2)
                    / (len(a_)+len(b_)-2)) if len(a_) > 1 and len(b_) > 1 else 0
        dd = abs(a_.mean()-b_.mean())/s if s else 0.0
        print(f"{nm:<14}{v['normal'].mean():>10.3f}{v['drowsy'].mean():>10.3f}"
              f"{v['yawning'].mean():>10.3f}{dd:>18.2f}")


if __name__ == "__main__":
    main()
