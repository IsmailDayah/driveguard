"""DriveGuard AI — Branch A ablation study.

The first Branch A run scored 68.0% leave-one-person-out, and the confusion
matrix showed two specific, explainable failures:

  1. 'phone' has no facial signature. Holding a phone leaves the eyes open and
     the mouth closed, i.e. geometrically identical to 'normal'. The phone
     itself lies outside the eye/mouth region these features measure, so no
     amount of tuning can fix it - it is a job for the detector in Branch B.

  2. Per-person accuracy ranged 43.8% - 94.7%, because absolute EAR is
     person-specific (measured open-eye baselines: P4 0.264, P1 0.299,
     P3 0.234, P2 0.232). A single decision boundary cannot serve all.

This script tests both hypotheses instead of assuming them:

  A  4-class, raw features                 (the baseline)
  B  3-class, raw features                 (phone handed to Branch B)
  C  4-class, per-driver calibrated
  D  3-class, per-driver calibrated        (the proposed configuration)

Calibration is not test-set peeking: it uses ONLY the person's 'normal' frames,
mirroring a real driver-monitoring system that asks the driver to look at the
camera for a few seconds when they sit down. Labels for drowsy/yawning/phone are
never touched. Both calibrated and uncalibrated numbers are reported.

    python experiment_branch_a.py
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
QC = os.path.join(HERE, "qc")
os.makedirs(QC, exist_ok=True)

BG, TXT, MUTED = "#0A1628", "#F1F5F9", "#93A6BD"
FULL = ["normal", "drowsy", "yawning", "phone"]
THREE = ["normal", "drowsy", "yawning"]

# indices into X_geo: 0 EAR_L, 1 EAR_R, 2 EAR_mean, 3 |EAR_L-R|,
#                     4 MAR, 5 MAR/EAR, 6 face h/w
EAR_IDX = [0, 1, 2, 3]
MAR_IDX = [4]


def calibrate(X, y, persons):
    """Scale each person's EAR/MAR by their own 'normal' baseline."""
    Xc = X.copy()
    for p in set(persons.tolist()):
        m = persons == p
        base = X[m & (y == "normal")]
        if len(base) == 0:
            continue
        ear0 = max(float(np.mean(base[:, 2])), 1e-6)
        mar0 = max(float(np.mean(base[:, 4])), 1e-6)
        Xc[np.ix_(np.where(m)[0], EAR_IDX)] = X[np.ix_(np.where(m)[0], EAR_IDX)] / ear0
        Xc[np.ix_(np.where(m)[0], MAR_IDX)] = X[np.ix_(np.where(m)[0], MAR_IDX)] / mar0
    return Xc


def loso(X, y, persons, classes):
    keep = np.isin(y, classes)
    X, y, persons = X[keep], y[keep], persons[keep]
    yt, yp, per_p = [], [], {}
    for p in sorted(set(persons.tolist())):
        te = persons == p
        m = make_pipeline(StandardScaler(),
                          SVC(kernel="linear", C=1, class_weight="balanced",
                              random_state=0))
        m.fit(X[~te], y[~te])
        pred = m.predict(X[te])
        per_p[p] = accuracy_score(y[te], pred)
        yt.append(y[te]); yp.append(pred)
    return np.concatenate(yt), np.concatenate(yp), per_p


def plot_cm(cm, classes, title, sub, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="cividis", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{100*cmn[i, j]:.0f}%", ha="center",
                    va="center", fontsize=11, fontweight="bold",
                    color="white" if cmn[i, j] < 0.55 else BG)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color=TXT)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=TXT)
    ax.set_xlabel("Predicted", color=TXT, fontweight="bold")
    ax.set_ylabel("True", color=TXT, fontweight="bold")
    ax.set_title(f"{title}\n{sub}", color=TXT, fontsize=12,
                 fontweight="bold", pad=12)
    for s in ax.spines.values():
        s.set_color("#2B4263")
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(colors=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=170, facecolor=BG); plt.close(fig)


def main():
    d = np.load(os.path.join(HERE, "features.npz"), allow_pickle=True)
    X, y, persons = d["X_geo"], d["y"], d["persons"]
    Xc = calibrate(X, y, persons)

    variants = [
        ("A  4-class, raw",        X,  FULL),
        ("B  3-class, raw",        X,  THREE),
        ("C  4-class, calibrated", Xc, FULL),
        ("D  3-class, calibrated", Xc, THREE),
    ]

    print(f"{'variant':<24}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}   per-person")
    print("-" * 84)
    out, best = [], None
    for name, Xv, classes in variants:
        yt, yp, per_p = loso(Xv, y, persons, classes)
        acc = accuracy_score(yt, yp)
        pr, rc, f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0)
        spread = f"{100*min(per_p.values()):.0f}-{100*max(per_p.values()):.0f}%"
        print(f"{name:<24}{100*acc:>7.1f}%{100*pr:>7.1f}%{100*rc:>7.1f}%"
              f"{100*f1:>7.1f}%   {spread}")
        rec = {"variant": name, "accuracy": acc, "precision": pr,
               "recall": rc, "f1": f1,
               "per_person": {k: float(v) for k, v in per_p.items()},
               "classes": classes}
        out.append(rec)
        if best is None or acc > best["accuracy"]:
            best = dict(rec, y_true=yt, y_pred=yp)

    print("\n" + "=" * 84)
    print(f"BEST: {best['variant']}  ->  {100*best['accuracy']:.1f}% "
          f"accuracy, macro-F1 {100*best['f1']:.1f}%")
    print("=" * 84)
    print("\nper-person (model never trained on that person):")
    for p, a in sorted(best["per_person"].items()):
        print(f"  {p:<10}{100*a:>7.1f}%")
    print("\n" + classification_report(best["y_true"], best["y_pred"],
                                       labels=best["classes"],
                                       zero_division=0, digits=3))

    cm = confusion_matrix(best["y_true"], best["y_pred"], labels=best["classes"])
    p = os.path.join(QC, "branch_a_best_confusion.png")
    plot_cm(cm, best["classes"], f"Branch A - {best['variant']}",
            f"leave-one-person-out accuracy {100*best['accuracy']:.1f}%", p)
    print("confusion matrix ->", p)

    with open(os.path.join(HERE, "branch_a_ablation.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("results ->", os.path.join(HERE, "branch_a_ablation.json"))


if __name__ == "__main__":
    main()
