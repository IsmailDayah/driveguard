"""Fill-light test: does adding frontal light recover the night failure?

The night result cost 16.3 points (96.3% -> 80.0%), traced to a specific
physical cause: the EYE REGION was 69% darker (89.6 -> 27.7) with 53% less fine
detail, even though whole-frame brightness barely moved (119.6 -> 114.9).
Overhead light plus brow and spectacle frames shadowed the eyes while
auto-exposure balanced for the whole face. Closed-eye EAR then doubled
(0.101 -> 0.202), crossed the 0.20 PERCLOS cutoff, and drowsy recall halved.

Five pre-processing methods failed to recover it (best +0.015 of a 0.26 gap),
which said the detail was never captured - a sensor problem, not a software one.

This tests that diagnosis directly by fixing the ILLUMINATION instead.

PREDICTIONS, recorded before running so they can be checked rather than
rationalised afterwards:

  1. eye-region brightness rises from 27.7 back toward the daytime 89.6
  2. eye-region fine detail rises from 1220 back toward 2609
  3. closed-eye EAR falls from 0.202 back toward 0.101
  4. drowsy recall rises from 48.4% back toward daytime
  5. accuracy recovers most of the 16.3-point loss

If 1-2 improve but 3-5 do not, the diagnosis is wrong and something other than
eye luminance is responsible. If nothing improves, the light was mispositioned.

FAILURE MODE TO WATCH: a bare phone LED aimed at the eyes makes the subject
squint, which lowers EAR by itself. That would look like "the fix worked" for
the wrong reason, so squint is checked for explicitly - `normal` EAR must NOT
fall relative to night.

    python experiment_fill.py
"""
import os
import sys
import glob
import json

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_features import ear_of, mar_of, LEFT_EYE, RIGHT_EYE, MOUTH
from extract_temporal import temporal_block, NAMES, parse
from experiment_night import scan, featurise, day_training, SETS

FRAMES = os.path.join(HERE, "..", "..", "data", "frames")
MODEL = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")
QC = os.path.join(HERE, "qc")
THREE = ["normal", "drowsy", "yawning"]
BG, TXT = "#0A1628", "#F1F5F9"
CONDS = [("day", "P1"), ("night", "P1_night"), ("fill", "P1_fill")]


def eye_region_stats(person):
    """Brightness / contrast / detail measured in the EYE REGION only.

    A whole-frame average hides a dark patch sitting exactly where the signal
    lives - that is what made the night failure invisible at frame level.
    """
    lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL)),
        running_mode=RunningMode.IMAGE, num_faces=1))
    rows = {}
    for cls in THREE:
        vals = []
        for p in sorted(glob.glob(os.path.join(FRAMES, cls, "*.jpg"))):
            m = parse(os.path.basename(p))
            if not m or m[0] != person:
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(
                                       cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            if not r.face_landmarks:
                continue
            q = np.array([[t.x * w, t.y * h] for t in r.face_landmarks[0]])
            lc, rc = q[LEFT_EYE].mean(axis=0), q[RIGHT_EYE].mean(axis=0)
            iod = float(np.hypot(*(lc - rc)))
            cx, cy = (lc + rc) / 2.0
            x0 = int(max(0, cx - 1.3*iod)); x1 = int(min(w, cx + 1.3*iod))
            y0 = int(max(0, cy - 0.45*iod)); y1 = int(min(h, cy + 0.45*iod))
            roi = img[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            vals.append((g.mean(), g.std(),
                         cv2.Laplacian(g, cv2.CV_64F).var(),
                         float((g > 220).mean()),
                         (ear_of(q, LEFT_EYE) + ear_of(q, RIGHT_EYE)) / 2))
        if vals:
            rows[cls] = np.array(vals)
    lm.close()
    return rows


def main():
    have = {}
    for label, person in CONDS:
        n = len(glob.glob(os.path.join(FRAMES, "*", f"{person}_*.jpg")))
        if label == "day":
            n = len([f for f in glob.glob(os.path.join(FRAMES, "*", "P1_*.jpg"))
                     if "_night_" not in f and "_fill_" not in f])
        have[label] = n
    print("frames available: " + "  ".join(f"{k} {v}" for k, v in have.items()))
    if have["fill"] == 0:
        raise SystemExit(
            "\nNo 'P1_fill' frames yet. Record with a lamp/phone bounced at\n"
            "camera height (NOT aimed into the eyes - squinting would corrupt\n"
            "the EAR measurement), then:\n"
            "  cd data\n"
            "  python record_clips.py --person P1_fill --only normal  --count 2\n"
            "  python record_clips.py --person P1_fill --only drowsy  --count 2\n"
            "  python record_clips.py --person P1_fill --only yawning --count 2\n"
            "  (then re-run this script)")

    print("\n" + "=" * 84)
    print("1. EYE-REGION OPTICS  (the measured cause of the night failure)")
    print("=" * 84)
    stats = {}
    print(f"{'cond':<7}{'class':<9}{'bright':>9}{'contrast':>10}{'detail':>10}"
          f"{'glare%':>9}{'EAR':>8}")
    print("-" * 62)
    for label, person in CONDS:
        st = eye_region_stats(person)
        stats[label] = st
        for cls in THREE:
            if cls in st:
                v = st[cls]
                print(f"{label:<7}{cls:<9}{v[:,0].mean():>9.1f}{v[:,1].mean():>10.1f}"
                      f"{v[:,2].mean():>10.0f}{100*v[:,3].mean():>8.2f}%"
                      f"{v[:,4].mean():>8.3f}")
        print()

    print("=" * 84)
    print("2. DID THE PREDICTIONS HOLD?")
    print("=" * 84)
    def g(cond, cls, col):
        return stats[cond][cls][:, col].mean() if cls in stats.get(cond, {}) else float("nan")
    checks = [
        ("eye brightness (drowsy)", 0, "up"),
        ("eye detail (drowsy)", 2, "up"),
        ("closed-eye EAR (drowsy)", 4, "down"),
    ]
    for name, col, want in checks:
        d, nn, f = g("day", "drowsy", col), g("night", "drowsy", col), g("fill", "drowsy", col)
        moved = f - nn
        ok = (moved > 0) if want == "up" else (moved < 0)
        frac = abs(moved) / max(abs(d - nn), 1e-9)
        print(f"  {name:<26} day {d:>8.3f}  night {nn:>8.3f}  fill {f:>8.3f}   "
              f"{'RECOVERED' if ok else 'no change/worse':<16} "
              f"({100*frac:.0f}% of the gap)")

    # squint check - a light in the eyes lowers OPEN-eye EAR too
    n_open, f_open = g("night", "normal", 4), g("fill", "normal", 4)
    print(f"\n  squint check: normal EAR night {n_open:.3f} -> fill {f_open:.3f}")
    if f_open < n_open - 0.02:
        print("  !! OPEN-eye EAR dropped - the subject was squinting into the")
        print("     light. Any drowsy-EAR improvement is confounded. Re-record")
        print("     with the light bounced off a wall instead of aimed at the face.")
    else:
        print("  ok - open-eye EAR held, so the light did not cause squinting")

    print("\n" + "=" * 84)
    print("3. ACCURACY")
    print("=" * 84)
    Xtr, ytr = day_training("absolute")
    ctl_tr, ctl_y = day_training("absolute", exclude="P1")
    for label, person in CONDS:
        clips, seen, lost, _ = scan(person)
        X, yv, _ = featurise(clips, "absolute")
        if X is None:
            continue
        tr_X, tr_y = (ctl_tr, ctl_y) if label == "day" else (Xtr, ytr)
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="linear", C=1, class_weight="balanced",
                                random_state=0)).fit(tr_X, tr_y)
        pred = clf.predict(X)
        acc = accuracy_score(yv, pred)
        _, rcs, _, _ = precision_recall_fscore_support(
            yv, pred, labels=THREE, zero_division=0)
        print(f"  {label:<7}n={len(yv):<5} accuracy {100*acc:>5.1f}%   "
              f"drowsy recall {100*rcs[1]:>5.1f}%   yawning {100*rcs[2]:>5.1f}%"
              f"   face-loss {100*lost/max(seen,1):.1f}%")

    json.dump({k: {c: v.mean(axis=0).tolist() for c, v in st.items()}
               for k, st in stats.items()},
              open(os.path.join(HERE, "fill_light.json"), "w"), indent=1)
    print("\nresults -> fill_light.json")


if __name__ == "__main__":
    main()
