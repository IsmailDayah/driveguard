"""Can pre-processing recover the night failure?

Measured cause: the EYE REGION is 69% darker at night (89.6 -> 27.7) with 53%
less fine detail, even though whole-frame brightness barely moved (119.6 ->
114.9). Overhead light plus brow and spectacle frames put the eyes in shadow
while auto-exposure balances for the whole face. MediaPipe then cannot resolve
the eyelid crease and EAR is overestimated (0.101 -> 0.202), which pushes
closed eyes above the 0.20 PERCLOS cutoff and collapses drowsy recall to 48.4%.

If that diagnosis is right, restoring local contrast BEFORE landmark detection
should recover EAR. Note the existing CLAHE in extract_features is applied to
the eye strip for HOG only - EAR is computed from landmarks on the raw frame,
so it never touched this.

Candidates, cheapest first (all real-time viable):
  raw        baseline
  clahe      CLAHE on the luminance channel
  gamma      gamma < 1 lifts shadows without blowing highlights
  clahe+g    both
  eq         plain global histogram equalisation, as a control

Judged on the thing that matters - does closed-eye EAR come back down, and does
the day result stay intact? A fix that repairs night by breaking day is not a
fix.

    python experiment_preprocess.py
"""
import os
import sys
import glob
import json

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_features import ear_of, mar_of, LEFT_EYE, RIGHT_EYE, MOUTH

FRAMES = os.path.join(HERE, "..", "..", "data", "frames")
MODEL = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_LUT = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], np.uint8)


def pp_raw(img):
    return img


def pp_clahe(img):
    """CLAHE on L only, so colour is preserved for the landmarker."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def pp_gamma(img):
    return cv2.LUT(img, _LUT)


def pp_clahe_gamma(img):
    return pp_gamma(pp_clahe(img))


def pp_eq(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


METHODS = [("raw", pp_raw), ("clahe", pp_clahe), ("gamma", pp_gamma),
           ("clahe+gamma", pp_clahe_gamma), ("hist-eq", pp_eq)]


def measure(paths, fn, lm):
    ears, mars, lost = [], [], 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = fn(img)
        h, w = img.shape[:2]
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(
                                   cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        if not r.face_landmarks:
            lost += 1
            continue
        q = np.array([[t.x * w, t.y * h] for t in r.face_landmarks[0]])
        ears.append((ear_of(q, LEFT_EYE) + ear_of(q, RIGHT_EYE)) / 2)
        mars.append(mar_of(q, MOUTH))
    return np.array(ears), np.array(mars), lost


def main():
    lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL)),
        running_mode=RunningMode.IMAGE, num_faces=1))

    sets = {}
    for cond, pat in (("day", "P1_{c}_*.jpg"),
                      ("night", "P1_night_{c}_*.jpg")):
        for cls in ("normal", "drowsy", "yawning"):
            fs = sorted(glob.glob(os.path.join(FRAMES, cls,
                                               pat.format(c=cls))))
            if cond == "day":
                fs = [f for f in fs if "_night_" not in os.path.basename(f)]
            sets[(cond, cls)] = fs

    print("target: night drowsy EAR should fall back toward the day value")
    print("        (day 0.101) WITHOUT disturbing day, and normal must stay high\n")
    print(f"{'method':<14}{'cond':<7}{'normal EAR':>12}{'drowsy EAR':>12}"
          f"{'separation':>12}{'lost':>6}")
    print("-" * 66)
    out = {}
    for name, fn in METHODS:
        row = {}
        for cond in ("day", "night"):
            en, _, l1 = measure(sets[(cond, "normal")], fn, lm)
            ed, _, l2 = measure(sets[(cond, "drowsy")], fn, lm)
            # normalised gap between open and closed - the quantity PERCLOS needs
            sep = (en.mean() - ed.mean()) / max(en.mean(), 1e-9)
            row[cond] = dict(normal=float(en.mean()), drowsy=float(ed.mean()),
                             sep=float(sep), lost=int(l1 + l2))
            print(f"{name:<14}{cond:<7}{en.mean():>12.3f}{ed.mean():>12.3f}"
                  f"{sep:>12.3f}{l1+l2:>6}")
        out[name] = row
        print()

    print("=" * 66)
    print("Which method restores the OPEN/CLOSED gap at night without")
    print("damaging day? (higher separation is better; day must not drop)")
    print("=" * 66)
    base_day = out["raw"]["day"]["sep"]
    print(f"{'method':<14}{'day sep':>10}{'night sep':>11}{'day change':>12}"
          f"{'night change':>13}")
    print("-" * 62)
    best, bestv = None, -9
    for name, _ in METHODS:
        d, n = out[name]["day"]["sep"], out[name]["night"]["sep"]
        dd = d - base_day
        nn = n - out["raw"]["night"]["sep"]
        print(f"{name:<14}{d:>10.3f}{n:>11.3f}{dd:>+12.3f}{nn:>+13.3f}")
        # only count it if day is not materially harmed
        if dd > -0.03 and n > bestv:
            best, bestv = name, n
    print(f"\nbest night separation without harming day: {best} ({bestv:.3f})")
    print(f"raw night separation was {out['raw']['night']['sep']:.3f}, "
          f"day {base_day:.3f}")
    lm.close()
    with open(os.path.join(HERE, "preprocess_night.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("results -> preprocess_night.json")


if __name__ == "__main__":
    main()
