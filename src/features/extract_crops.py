"""DriveGuard AI — cache image crops for Branch B (the deep-learning branch).

Saves two aligned crops per frame so the CNN can be compared fairly against
Branch A:

  eye  (96x32)  - the SAME region Branch A sees, so any difference in score is
                  due to the model, not the input
  face (96x96)  - the wider region, to measure what the CNN gains from context

Both use the inter-ocular anchor from extract_features.eye_strip, so crop
geometry does not encode the label (verified: 1.02x spread across classes).

    python extract_crops.py
"""
import os
import sys
import glob
import argparse

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_features import LEFT_EYE, RIGHT_EYE, eye_strip, clahe

FRAMES = os.path.join(HERE, "..", "..", "data", "frames")
MODEL = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")
OUT = os.path.join(HERE, "crops.npz")

CLASSES = ["normal", "drowsy", "yawning", "phone"]
FACE_CROP = 96          # default; overridden by --size

# Measured over an 81-frame sample spanning every person and class: the region
# face_crop() resizes is 294x294 px median (min 238, max 339), and 100.0% of it
# is genuine pixels rather than replicated border. So 96 keeps only 0.33x of the
# available detail and 224 (MobileNetV2's native ImageNet size) is still real
# image at 0.76x. Anything past ~294 would be upsampling.
NATIVE_CROP_PX = 294


def face_crop(gray, pts):
    """Square face crop anchored on inter-ocular distance (scale-invariant)."""
    lc, rc = pts[LEFT_EYE].mean(axis=0), pts[RIGHT_EYE].mean(axis=0)
    iod = float(np.hypot(*(lc - rc)))
    if iod < 8:
        return None
    cx, cy = (lc + rc) / 2.0
    half = 1.9 * iod                     # covers brow to chin at typical ratios
    cy = cy + 0.45 * iod                 # shift down so the mouth is included
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - gray.shape[1]), max(0, y1 - gray.shape[0])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    c = gray[y0:y1, x0:x1]
    if pl or pr or pt or pb:
        c = cv2.copyMakeBorder(c, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return cv2.resize(clahe.apply(c), (FACE_CROP, FACE_CROP))


def main():
    global FACE_CROP
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=96,
                    help=f"face crop side in px; the source region is "
                         f"~{NATIVE_CROP_PX}px so anything above that upsamples")
    ap.add_argument("--out", default="",
                    help="output .npz (default crops.npz, or crops_<size>.npz "
                         "when --size is not 96)")
    a = ap.parse_args()
    FACE_CROP = a.size
    out = a.out or os.path.join(
        HERE, "crops.npz" if a.size == 96 else f"crops_{a.size}.npz")
    if a.size > NATIVE_CROP_PX:
        print(f"WARNING: --size {a.size} exceeds the ~{NATIVE_CROP_PX}px source "
              f"region; this upsamples and adds no information.")
    print(f"face crop {a.size}x{a.size}  "
          f"({a.size/NATIVE_CROP_PX:.2f}x the native source region)")

    lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL)),
        running_mode=RunningMode.IMAGE, num_faces=1))

    eyes, faces, y, persons = [], [], [], []
    skipped = 0
    for cls in CLASSES:
        paths = sorted(glob.glob(os.path.join(FRAMES, cls, "*.jpg")))
        got = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                skipped += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            res = lm.detect(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            if not res.face_landmarks:
                skipped += 1
                continue
            pts = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]])
            e = eye_strip(gray, pts)
            f = face_crop(gray, pts)
            if e is None or f is None:
                skipped += 1
                continue
            eyes.append(e); faces.append(f)
            y.append(cls); persons.append(os.path.basename(p).split("_")[0])
            got += 1
        print(f"{cls:<10}{got:>5} crops")
    lm.close()

    eyes = np.array(eyes, np.uint8)
    faces = np.array(faces, np.uint8)
    np.savez_compressed(out, eyes=eyes, faces=faces,
                        y=np.array(y), persons=np.array(persons))
    print(f"\nsaved {out}")
    print(f"  eye crops  {eyes.shape}")
    print(f"  face crops {faces.shape}")
    print(f"  skipped    {skipped}")


if __name__ == "__main__":
    main()
