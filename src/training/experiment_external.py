"""External validation: train on OUR 4 people, test on FL3D's subjects.

Every number in this project so far is leave-one-person-out across four people,
one webcam, two rooms, daytime. That measures generalisation to a new *person*
in the *same* setup. It does not measure generalisation at all in the sense a
reviewer means.

FL3D is a genuinely different distribution: 44 in-vehicle videos, different
cameras, lighting, demographics, real driving rather than simulated states, and
human per-frame labels for alert / microsleep / yawning.

Protocol - deliberately the hardest honest version:
  train on ALL 1,039 of our frames, test on FL3D. Nothing from FL3D touches
  training, and no per-dataset calibration is applied.

To keep the temporal features identical to ours, FL3D frames are sampled every
5th (our frames were kept every 5th from 25 fps video), so one window of 6
sampled frames is 1.2 s in both datasets.

Both outcomes are results. If it transfers, the 3-feature model has external
validity. If it collapses, drowsiness features are setup-specific - which our
own per-person variance already predicts, and which is worth reporting.

    python experiment_external.py --limit-videos 8     # quick check
    python experiment_external.py                      # full
"""
import os
import sys
import json
import glob
import time
import argparse
import collections

import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_features import ear_of, mar_of, LEFT_EYE, RIGHT_EYE, MOUTH
from extract_temporal import temporal_block, NAMES, parse

FL3D = os.path.join(HERE, "datasets", "fl3d", "classification_frames")
MODEL = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")
THREE = ["normal", "drowsy", "yawning"]
MAP = {"alert": "normal", "microsleep": "drowsy", "yawning": "yawning"}
STRIDE = 5          # match our own sampling so the window means the same thing
WINDOW = 6


# Which temporal columns each variant uses. MAR is always taken from the
# geometric block because it is already a self-normalising ratio.
FEATURE_SETS = {
    "absolute": ["perclos_20", "closed_run_w"],
    "relative": ["perclos_rel70", "closed_run_rel"],
    "both": ["perclos_20", "closed_run_w", "perclos_rel70", "closed_run_rel"],
}


def our_features(which="absolute"):
    """Our 1,039 frames -> the selected columns (+ MAR)."""
    d = np.load(os.path.join(HERE, "features_temporal.npz"), allow_pickle=True)
    G, T, y = d["X_geo"], d["X_temporal"], d["y"]
    ti = {n: i for i, n in enumerate([str(x) for x in d["temporal_names"]])}
    cols = [T[:, ti[n]] for n in FEATURE_SETS[which]] + [G[:, 4]]
    X = np.column_stack(cols)
    keep = np.isin(y, THREE)
    return X[keep], y[keep]


CACHE = os.path.join(HERE, "fl3d_earmar_cache.npz")


def fl3d_sequences(limit_videos=0, cap_frames=0):
    """Per-video (ear, mar, labels). Cached: the landmark pass is the expensive
    part and does not depend on which features we later build from it."""
    if os.path.exists(CACHE) and not limit_videos and not cap_frames:
        z = np.load(CACHE, allow_pickle=True)
        print(f"using cached landmarks: {CACHE}")
        return list(z["seqs"])
    ann = json.load(open(os.path.join(FL3D, "annotations_all.json"),
                        encoding="utf-8"))
    byvid = collections.defaultdict(list)
    for k, v in ann.items():
        st = v.get("driver_state")
        if st not in MAP:
            continue
        p = k.split("/")
        n = "".join(c for c in os.path.splitext(p[-1])[0] if c.isdigit())
        byvid[p[-2]].append((int(n) if n else 0, p[-1], MAP[st]))

    vids = sorted(byvid)
    if limit_videos:
        vids = vids[:limit_videos]
    print(f"FL3D videos: {len(vids)}  "
          f"(frames sampled every {STRIDE}th to match our protocol)")

    lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL)),
        running_mode=RunningMode.IMAGE, num_faces=1))

    Xs, ys, vs = [], [], []
    t0 = time.time()
    nf = 0
    for vi, v in enumerate(vids):
        seq = sorted(byvid[v])[::STRIDE]
        if cap_frames:
            seq = seq[:cap_frames]
        ear, mar, lab = [], [], []
        for idx, fname, cls in seq:
            # colour, not grayscale - see experiment_night.py. Reading grey
            # cost 16 points on a control fold, so the first FL3D run
            # (85.8% acc / 82.3% macro-F1) understated the true transfer.
            img = cv2.imread(os.path.join(FL3D, v, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(
                                       cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            if not r.face_landmarks:
                nf += 1
                continue
            q = np.array([[t.x * w, t.y * h] for t in r.face_landmarks[0]])
            ear.append((ear_of(q, LEFT_EYE) + ear_of(q, RIGHT_EYE)) / 2)
            mar.append(mar_of(q, MOUTH))
            lab.append(cls)
        if len(ear) < WINDOW:
            continue
        Xs.append((v, np.array(ear, np.float32), np.array(mar, np.float32),
                   np.array(lab)))
        if (vi + 1) % 5 == 0:
            print(f"  {vi+1}/{len(vids)} videos, "
                  f"{sum(len(x[1]) for x in Xs)} frames "
                  f"({time.time()-t0:.0f}s)", flush=True)
    lm.close()
    print(f"  faces not found: {nf}")
    if not limit_videos and not cap_frames:
        np.savez_compressed(CACHE, seqs=np.array(Xs, dtype=object))
        print(f"cached -> {CACHE}")
    return Xs


def fl3d_features(which="absolute", limit_videos=0, cap_frames=0):
    seqs = fl3d_sequences(limit_videos, cap_frames)
    ti = {n: i for i, n in enumerate(NAMES)}
    Xs, ys, vs = [], [], []
    for v, ear, mar, lab in seqs:
        T = temporal_block(ear, mar, WINDOW)
        cols = [T[:, ti[n]] for n in FEATURE_SETS[which]] + [mar]
        Xs.append(np.column_stack(cols))
        ys += list(lab)
        vs += [v] * len(lab)
    return np.vstack(Xs), np.array(ys), np.array(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--cap-frames", type=int, default=0)
    ap.add_argument("--features", default="absolute",
                    choices=["absolute", "relative", "both"])
    a = ap.parse_args()

    print(f'feature set: {a.features}')
    Xo, yo = our_features(a.features)
    print(f"our data: {len(yo)} frames  "
          f"{dict(collections.Counter(yo.tolist()))}\n")

    Xf, yf, vf = fl3d_features(a.features, a.limit_videos,
                               a.cap_frames)
    print(f"\nFL3D: {len(yf)} frames from {len(set(vf))} videos  "
          f"{dict(collections.Counter(yf.tolist()))}")

    # train on ours, test on FL3D. no FL3D data in training, no calibration.
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel="linear", C=1, class_weight="balanced",
                            random_state=0))
    clf.fit(Xo, yo)
    pred = clf.predict(Xf)

    acc = accuracy_score(yf, pred)
    _, _, f1, _ = precision_recall_fscore_support(
        yf, pred, average="macro", zero_division=0)
    print(f"\n{'='*74}")
    print(f"EXTERNAL VALIDATION  train=ours(4 people)  test=FL3D({len(set(vf))} videos)")
    print(f"{'='*74}")
    print(f"accuracy {100*acc:.1f}%   macro-F1 {100*f1:.1f}%")
    print(f"(our own LOSO was 95.0% - the gap is the domain shift)\n")
    print(classification_report(yf, pred, labels=THREE, digits=3,
                                zero_division=0))
    cm = confusion_matrix(yf, pred, labels=THREE)
    print("confusion (rows=true):")
    print(f"{'':<10}" + "".join(f"{c:>10}" for c in THREE))
    for i, c in enumerate(THREE):
        print(f"{c:<10}" + "".join(f"{v:>10}" for v in cm[i]))

    # per-video, so one bad clip cannot hide behind a good average
    per = {}
    for v in sorted(set(vf)):
        m = vf == v
        per[v] = float(accuracy_score(yf[m], pred[m]))
    vals = np.array(list(per.values()))
    print(f"\nper-video accuracy: mean {100*vals.mean():.1f}%  "
          f"median {100*np.median(vals):.1f}%  "
          f"min {100*vals.min():.1f}%  max {100*vals.max():.1f}%")
    worst = sorted(per.items(), key=lambda kv: kv[1])[:5]
    print("  worst: " + ", ".join(f"{k.replace('_720','')} {100*v:.0f}%"
                                  for k, v in worst))

    # FL3D is 73% `alert`, so ACCURACY is the wrong yardstick: a
    # predict-everything-normal model scores 72.9% accuracy with zero
    # utility. Judge on metrics that imbalance cannot inflate. An earlier
    # version compared accuracy against majority-class ACCURACY and wrongly
    # logged "DOES NOT TRANSFER" for a model sitting +54 macro-F1 above
    # that baseline.
    from sklearn.metrics import balanced_accuracy_score
    cnt = collections.Counter(yf.tolist())
    maj = max(cnt, key=cnt.get)
    dummy = np.array([maj] * len(yf))
    _, _, f1_maj, _ = precision_recall_fscore_support(
        yf, dummy, average='macro', zero_division=0)
    bal = balanced_accuracy_score(yf, pred)
    bal_maj = balanced_accuracy_score(yf, dummy)
    acc_maj = cnt[maj] / len(yf)
    print('')
    print(f'{chr(109)+chr(101)+chr(116)+chr(114)+chr(105)+chr(99):<22}{chr(109)+chr(111)+chr(100)+chr(101)+chr(108):>11}{chr(109)+chr(97)+chr(106):>11}{chr(100)+chr(101)+chr(108)+chr(116)+chr(97):>10}')
    print('-' * 54)
    for nm, x, z in (('accuracy', acc, acc_maj),
                     ('balanced accuracy', bal, bal_maj),
                     ('macro-F1', f1, f1_maj)):
        print(f'{nm:<22}{100*x:>10.1f}%{100*z:>10.1f}%{100*(x-z):>+9.1f}')
    ok = (f1 - f1_maj) > 0.20 and (bal - bal_maj) > 0.20
    print('')
    print('verdict: ' + ('TRANSFERS' if ok else 'DOES NOT TRANSFER'))
    json.dump({"accuracy": float(acc), "f1": float(f1),
               "balanced_accuracy": float(bal),
               "f1_majority": float(f1_maj),
               "acc_majority": float(acc_maj),
               "balanced_accuracy": float(bal),
               "f1_majority": float(f1_maj), "acc_majority": float(acc_maj),
               "n_frames": int(len(yf)), "n_videos": len(set(vf)),
               "per_video": per, 
               "confusion": cm.tolist(), "classes": THREE},
              open(os.path.join(HERE,
                   f"external_validation_{a.features}.json"), "w"),
              indent=1)
    print("results -> external_validation.json")


if __name__ == "__main__":
    main()
