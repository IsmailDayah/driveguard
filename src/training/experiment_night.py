"""Night test: does the model survive an illumination shift?

Every frame in every dataset used so far - ours and FL3D's 44 videos - is
daytime or in-car. Low-light performance is completely untested, and the system
is scoped to daytime use, so this is an exploratory probe rather than a headline
result.

The design keeps ONE variable free. Same person, same room, same seat, same
camera, same actions; only the lighting changed. So a drop cannot be blamed on
a new face or a new setup - it is illumination or nothing.

Three comparisons, in increasing strictness:

  1. P1 DAY   (existing LOSO fold, trained on the other three) - the control
  2. P1 NIGHT trained on all four people's day data              - the test
  3. absolute vs relative PERCLOS on the night frames                - the question

(3) matters because low light makes EAR estimates jitter, and an ABSOLUTE 0.20
cutoff has no way to adapt while a relative one does. Relative already lost on
FL3D; if it wins here, that identifies precisely when self-normalisation earns
its keep.

    python experiment_night.py                 # after recording
    python experiment_night.py --person P1_night
"""
import os
import sys
import glob
import json
import argparse
import collections

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_features import ear_of, mar_of, LEFT_EYE, RIGHT_EYE, MOUTH
from extract_temporal import temporal_block, NAMES, parse

FRAMES = os.path.join(HERE, "..", "..", "data", "frames")
MODEL = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")
QC = os.path.join(HERE, "qc")
THREE = ["normal", "drowsy", "yawning"]
WINDOW = 6
BG, TXT = "#0A1628", "#F1F5F9"

SETS = {"absolute": ["perclos_20", "closed_run_w"],
        "relative": ["perclos_rel70", "closed_run_rel"]}


def scan(person_filter):
    """EAR/MAR/brightness per frame for the matching person, grouped by clip."""
    lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL)),
        running_mode=RunningMode.IMAGE, num_faces=1))
    clips = collections.defaultdict(list)
    seen = nofound = 0
    bright = []
    for cls in THREE:
        for p in sorted(glob.glob(os.path.join(FRAMES, cls, "*.jpg"))):
            m = parse(os.path.basename(p))
            if not m or m[0] != person_filter:
                continue
            seen += 1
            # MUST read colour. MediaPipe's landmarker uses colour
            # information; feeding it grey-replicated-to-RGB degrades landmark
            # precision enough to move P1's day fold from 96% to 80.4%.
            # extract_features.py (which produced every headline number) reads
            # colour, so anything compared against it must do the same.
            img = cv2.imread(p)
            if img is None:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bright.append((g.mean(), g.std()))
            h, w = g.shape
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=np.ascontiguousarray(
                                       cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            if not r.face_landmarks:
                nofound += 1
                continue
            q = np.array([[t.x * w, t.y * h] for t in r.face_landmarks[0]])
            clips[(m[1], m[2])].append(
                (m[3],
                 (ear_of(q, LEFT_EYE) + ear_of(q, RIGHT_EYE)) / 2,
                 mar_of(q, MOUTH), p))
    lm.close()
    return clips, seen, nofound, np.array(bright) if bright else np.zeros((0, 2))


def featurise(clips, which):
    ti = {n: i for i, n in enumerate(NAMES)}
    Xs, ys, ps = [], [], []
    for (cls, _), lst in sorted(clips.items()):
        lst.sort()
        ear = np.array([e for _, e, _, _ in lst], np.float32)
        mar = np.array([m for _, _, m, _ in lst], np.float32)
        if len(ear) < WINDOW:
            continue
        T = temporal_block(ear, mar, WINDOW)
        cols = [T[:, ti[n]] for n in SETS[which]] + [mar]
        Xs.append(np.column_stack(cols))
        ys += [cls] * len(ear)
        ps += [p for _, _, _, p in lst]
    if not Xs:
        return None, None, None
    return np.vstack(Xs), np.array(ys), ps


def day_training(which, exclude=None):
    """Our daytime frames -> (X, y). `exclude` drops one person for a control."""
    d = np.load(os.path.join(HERE, "features_temporal.npz"), allow_pickle=True)
    G, T, y, persons = d["X_geo"], d["X_temporal"], d["y"], d["persons"]
    ti = {n: i for i, n in enumerate([str(x) for x in d["temporal_names"]])}
    X = np.column_stack([T[:, ti[n]] for n in SETS[which]] + [G[:, 4]])
    keep = np.isin(y, THREE)
    if exclude:
        keep &= persons != exclude
    return X[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="P1_night")
    ap.add_argument("--day-person", default="P1",
                    help="the same person's daytime clips, used as the control")
    a = ap.parse_args()

    print(f"scanning night frames for '{a.person}' ...")
    nclips, nseen, nlost, nbright = scan(a.person)
    if nseen == 0:
        raise SystemExit(
            f"no frames found for person '{a.person}'.\n"
            f"Record first:\n"
            f"  cd data\n"
            f"  python record_clips.py --person {a.person} --only normal  --count 2\n"
            f"  python record_clips.py --person {a.person} --only drowsy  --count 2\n"
            f"  python record_clips.py --person {a.person} --only yawning --count 2")

    print(f"scanning day frames for '{a.day_person}' (control) ...")
    dclips, dseen, dlost, dbright = scan(a.day_person)

    print(f"\n{'='*76}\n1. CAN MEDIAPIPE SEE YOU AT ALL?\n{'='*76}")
    print(f"{'condition':<10}{'frames':>8}{'no face':>9}{'loss %':>9}"
          f"{'brightness':>12}{'contrast':>10}")
    for nm, seen, lost, br in (("day", dseen, dlost, dbright),
                               ("night", nseen, nlost, nbright)):
        pct = 100 * lost / max(seen, 1)
        print(f"{nm:<10}{seen:>8}{lost:>9}{pct:>8.1f}%"
              f"{br[:, 0].mean() if len(br) else 0:>12.1f}"
              f"{br[:, 1].mean() if len(br) else 0:>10.1f}")
    if nseen and 100 * nlost / nseen > 25:
        print("\n  WARNING: >25% of night frames have no detectable face.")
        print("  Any accuracy below is conditioned on the frames that worked.")

    print(f"\n{'='*76}\n2. DID THE FEATURES SHIFT?\n{'='*76}")
    def stats(clips, label):
        rows = {}
        for (cls, _), lst in clips.items():
            for _, e, m, _ in lst:
                rows.setdefault(cls, []).append((e, m))
        for c in THREE:
            v = np.array(rows.get(c, []))
            if len(v):
                print(f"  {label:<6}{c:<9}EAR {v[:,0].mean():.3f}+-{v[:,0].std():.3f}"
                      f"   MAR {v[:,1].mean():.3f}+-{v[:,1].std():.3f}   n={len(v)}")
    stats(dclips, "day"); print()
    stats(nclips, "night")

    print(f"\n{'='*76}\n3. ACCURACY\n{'='*76}")
    out = {}
    for which in ("absolute", "relative"):
        Xn, yn, paths = featurise(nclips, which)
        if Xn is None:
            print(f"  {which}: too few usable night frames"); continue
        # control: this person's DAY fold, model trained on the other three
        Xd, yd, _ = featurise(dclips, which)
        Xtr_ctl, ytr_ctl = day_training(which, exclude=a.day_person)
        ctl = make_pipeline(StandardScaler(),
                            SVC(kernel="linear", C=1, class_weight="balanced",
                                random_state=0)).fit(Xtr_ctl, ytr_ctl)
        acc_day = accuracy_score(yd, ctl.predict(Xd)) if Xd is not None else float("nan")
        # test: night, model trained on ALL four people's day data
        Xtr, ytr = day_training(which)
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="linear", C=1, class_weight="balanced",
                                random_state=0)).fit(Xtr, ytr)
        pred = clf.predict(Xn)
        acc = accuracy_score(yn, pred)
        bal = balanced_accuracy_score(yn, pred)
        _, _, f1, _ = precision_recall_fscore_support(
            yn, pred, average="macro", zero_division=0)
        print(f"\n  --- {which} ---")
        print(f"  control  {a.day_person} DAY  (trained without them): "
              f"{100*acc_day:.1f}%")
        print(f"  test     {a.person} NIGHT (trained on all 4 day):  "
              f"{100*acc:.1f}%   balanced {100*bal:.1f}%   macro-F1 {100*f1:.1f}%")
        print(f"  illumination cost: {100*(acc-acc_day):+.1f} points")
        print(classification_report(yn, pred, labels=THREE, digits=3,
                                    zero_division=0))
        cm = confusion_matrix(yn, pred, labels=THREE)
        print("  confusion (rows=true):")
        print("  " + f"{'':<10}" + "".join(f"{c:>10}" for c in THREE))
        for i, c in enumerate(THREE):
            print("  " + f"{c:<10}" + "".join(f"{v:>10}" for v in cm[i]))
        out[which] = {"night_accuracy": float(acc), "night_balanced": float(bal),
                      "night_f1": float(f1), "day_control": float(acc_day),
                      "confusion": cm.tolist(),
                      "n_frames": int(len(yn))}
        # keep the worst frames for eyeballing
        if which == "absolute":
            wrong = [p for p, t, q in zip(paths, yn, pred) if t != q]
            out["examples_wrong"] = wrong[:12]

    if len(out.get("absolute", {})) and len(out.get("relative", {})):
        d_, r_ = out["absolute"]["night_f1"], out["relative"]["night_f1"]
        print(f"\n{'='*76}")
        print(f"absolute macro-F1 {100*d_:.1f}%   relative {100*r_:.1f}%   "
              f"-> {'relative wins at night' if r_ > d_ else 'absolute still wins'}")
        print("(relative lost on FL3D; a win here would show self-normalisation")
        print(" pays off specifically when the SENSOR degrades, not when the")
        print(" PERSON changes)")

    # ---- visual: night frames the model got wrong ----
    wrong = out.get("examples_wrong", [])
    if wrong:
        n = min(12, len(wrong)); cols = 6
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.3*cols, 2.4*rows),
                                 facecolor=BG, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, p in zip(axes.ravel(), wrong[:n]):
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
                ax.set_title(os.path.basename(p).split("_")[2], color=TXT,
                             fontsize=8)
        fig.suptitle("Night frames the model got wrong", color=TXT,
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pth = os.path.join(QC, "night_errors.png")
        fig.savefig(pth, dpi=130, facecolor=BG); plt.close(fig)
        print(f"\nwrong-frame contact sheet -> {pth}")

    with open(os.path.join(HERE, "night_test.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("results -> night_test.json")


if __name__ == "__main__":
    main()
