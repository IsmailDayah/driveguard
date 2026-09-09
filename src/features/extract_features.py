"""DriveGuard AI — Branch A feature extraction (the traditional-ML branch).

For every extracted frame this computes the hand-crafted features and writes them to a single .npz for the SVM/KNN stage:

    geometric : EAR (left, right, mean), MAR, eye/mouth ratios
    appearance: HOG descriptor of the CLAHE-equalised eye strip

CLAHE is applied to the eye crop before HOG because the recorded faces are
underexposed relative to the background (measured face/bg 0.46-0.77); CLAHE was
verified to recover ~2.3x more detail, which is exactly the gradient
information HOG depends on.

    python extract_features.py
"""
import os
import sys
import glob

import cv2
import numpy as np
from scipy.spatial import distance as dist
from skimage.feature import hog

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "..", "..", "data", "frames")
MODEL = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")
OUT = os.path.join(HERE, "features.npz")

CLASSES = ["normal", "drowsy", "yawning", "phone"]

# Same landmark indices the live system uses (driveguard_live.py) so the trained
# model consumes exactly the same features at inference time.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

EYE_CROP = (96, 32)          # w, h of the two-eye strip fed to HOG
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def ear_of(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    d = 2.0 * dist.euclidean(p1, p4)
    return (dist.euclidean(p2, p6) + dist.euclidean(p3, p5)) / d if d else 0.0


def mar_of(pts, idx):
    top, bottom, left, right = (pts[i] for i in idx)
    d = dist.euclidean(left, right)
    return dist.euclidean(top, bottom) / d if d else 0.0


def eye_strip(gray, pts):
    """Crop both eyes using an INTER-OCULAR anchor, equalise, resize.

    The crop must not depend on how open the eyes are. Padding by a multiple of
    the eye height did exactly that: closed eyes gave a 47px-tall crop and open
    eyes 69px (1.46x spread, aspect 3.6-5.6), so after resizing to a fixed size
    the amount of stretch encoded the label and a classifier could read the
    class off the geometry instead of the appearance.

    Inter-ocular distance (eye centre to eye centre) is unaffected by blinking,
    so anchoring to it gives a geometrically identical crop for every class.
    """
    lc = pts[LEFT_EYE].mean(axis=0)
    rc = pts[RIGHT_EYE].mean(axis=0)
    iod = float(np.hypot(*(lc - rc)))
    if iod < 8:
        return None
    cx, cy = (lc + rc) / 2.0
    aspect = EYE_CROP[0] / EYE_CROP[1]          # keep output aspect => no distortion
    half_w = 1.30 * iod
    half_h = half_w / aspect
    x0 = int(round(cx - half_w)); x1 = int(round(cx + half_w))
    y0 = int(round(cy - half_h)); y1 = int(round(cy + half_h))
    # pad rather than clamp, so a face near the edge keeps the same scale
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r = max(0, x1 - gray.shape[1]); pad_b = max(0, y1 - gray.shape[0])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 10 or y1 - y0 < 6:
        return None
    crop = gray[y0:y1, x0:x1]
    if pad_l or pad_r or pad_t or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_REPLICATE)
    crop = clahe.apply(crop)
    return cv2.resize(crop, EYE_CROP)


def main():
    landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL)),
        running_mode=RunningMode.IMAGE, num_faces=1))

    X_geo, X_hog, y, persons, files = [], [], [], [], []
    skipped = 0

    for cls in CLASSES:
        paths = sorted(glob.glob(os.path.join(FRAMES, cls, "*.jpg")))
        print(f"{cls:<10} {len(paths):>5} frames", end="", flush=True)
        got = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                skipped += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=np.ascontiguousarray(rgb)))
            if not res.face_landmarks:
                skipped += 1
                continue
            pts = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]])

            el, er = ear_of(pts, LEFT_EYE), ear_of(pts, RIGHT_EYE)
            mar = mar_of(pts, MOUTH)
            # normalise by face size so the features do not depend on distance
            fw = float(np.ptp(pts[:, 0])) or 1.0
            fh = float(np.ptp(pts[:, 1])) or 1.0
            geo = [el, er, (el + er) / 2.0, abs(el - er), mar,
                   mar / max(el + er, 1e-6), fh / fw]

            strip = eye_strip(gray, pts)
            if strip is None:
                skipped += 1
                continue
            hg = hog(strip, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), block_norm="L2-Hys",
                     feature_vector=True)

            X_geo.append(geo)
            X_hog.append(hg)
            y.append(cls)
            persons.append(os.path.basename(p).split("_")[0])
            files.append(os.path.basename(p))
            got += 1
        print(f"  -> {got} usable")

    landmarker.close()
    X_geo = np.array(X_geo, np.float32)
    X_hog = np.array(X_hog, np.float32)
    y = np.array(y)
    persons = np.array(persons)

    np.savez_compressed(OUT, X_geo=X_geo, X_hog=X_hog, y=y,
                        persons=persons, files=np.array(files))
    print(f"\nsaved {OUT}")
    print(f"  geometric features : {X_geo.shape}")
    print(f"  HOG features       : {X_hog.shape}")
    print(f"  skipped (no face)  : {skipped}")
    print(f"  people             : {sorted(set(persons.tolist()))}")


if __name__ == "__main__":
    main()
