"""Small physics-motivated feature subsets, not all 19 concatenated.

The full 19-feature set OVERFITS: geometric-only 90.3%, temporal-only 90.7%,
both together 88.3% on ~595 training frames per fold. Same lesson the 224px
resolution run taught - more real information is not automatically better when
the training set is small.

The two feature families also fail in complementary ways:

  geometric only  drowsy recall 80.3%   yawning recall 92.5%
  temporal only   drowsy recall 90.2%   yawning recall 83.0%

which says exactly what the physics says: **drowsiness is temporal, a yawn is
instantaneous.** A yawn is one large mouth opening; drowsiness is sustained eye
closure. Neither family carries both, and concatenating everything drowns the
signal in dimensions.

So the subsets below are written down BEFORE running, each justified by a
mechanism rather than chosen after seeing scores. All are reported, including
the losers, so this is not a search for the luckiest combination.

    python experiment_subsets.py
"""
import os
import json
import itertools

import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

from experiment_branch_a import loso, THREE, plot_cm

HERE = os.path.dirname(os.path.abspath(__file__))
QC = os.path.join(HERE, "qc")

GEO = ["EAR_L", "EAR_R", "EAR_mean", "EAR_asym", "MAR", "MAR_over_EAR", "face_hw"]


def main():
    d = np.load(os.path.join(HERE, "features_temporal.npz"), allow_pickle=True)
    G, T = d["X_geo"], d["X_temporal"]
    y, persons = d["y"], d["persons"]
    tnames = [str(x) for x in d["temporal_names"]]
    names = GEO + tnames
    X = np.hstack([G, T])
    ix = {n: i for i, n in enumerate(names)}

    # ---- subsets defined a priori, each with a stated mechanism ----
    SUBSETS = [
        ("A  geometric only (baseline)", GEO),
        ("B  temporal only", tnames),
        ("C  everything (19)", names),
        # minimal: one temporal drowsiness cue + one instantaneous yawn cue
        ("D  perclos_20 + MAR", ["perclos_20", "MAR"]),
        # add sustained-closure duration, which distinguishes a blink
        ("E  D + closed_run", ["perclos_20", "MAR", "closed_run_w"]),
        # add windowed mouth max so a yawn is seen even between frames
        ("F  E + mar_max_w", ["perclos_20", "MAR", "closed_run_w", "mar_max_w"]),
        # add instantaneous eye state: catches the onset before PERCLOS moves
        ("G  F + EAR_mean", ["perclos_20", "MAR", "closed_run_w", "mar_max_w",
                             "EAR_mean"]),
        # windowed eye level instead of the raw frame value
        ("H  G + ear_mean_w", ["perclos_20", "MAR", "closed_run_w", "mar_max_w",
                               "EAR_mean", "ear_mean_w"]),
        # the full geometric family plus only the two strongest temporal cues
        ("I  geometric + perclos_20 + closed_run",
         GEO + ["perclos_20", "closed_run_w"]),
        # geometric plus every eye-temporal cue, no mouth-temporal
        ("J  geometric + all eye-temporal",
         GEO + ["perclos_15", "perclos_20", "perclos_25", "closed_run_w",
                "ear_mean_w", "ear_min_w"]),
    ]

    print(f"{'subset':<42}{'n':>3}{'acc':>8}{'F1':>8}{'drowsy':>8}{'yawn':>7}"
          f"   per-person")
    print("-" * 104)
    rows = []
    for name, feats in SUBSETS:
        cols = [ix[f] for f in feats]
        yt, yp, pp = loso(X[:, cols], y, persons, THREE)
        acc = accuracy_score(yt, yp)
        _, _, f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0)
        _, rcs, _, _ = precision_recall_fscore_support(
            yt, yp, labels=THREE, zero_division=0)
        spread = " ".join(f"{k[:3]} {100*v:.0f}" for k, v in sorted(pp.items()))
        print(f"{name:<42}{len(cols):>3}{100*acc:>7.1f}%{100*f1:>7.1f}%"
              f"{100*rcs[1]:>7.1f}%{100*rcs[2]:>6.1f}%   {spread}")
        rows.append({"subset": name, "features": feats, "n": len(cols),
                     "accuracy": float(acc), "f1": float(f1),
                     "drowsy_recall": float(rcs[1]),
                     "yawning_recall": float(rcs[2]),
                     "per_person": {k: float(v) for k, v in pp.items()},
                     "y_true": yt, "y_pred": yp})

    base = rows[0]
    best = max(rows, key=lambda r: r["accuracy"])
    print(f"\n{'='*104}")
    print(f"baseline  A  {100*base['accuracy']:.1f}%")
    print(f"best      {best['subset']}  {100*best['accuracy']:.1f}%  "
          f"({100*(best['accuracy']-base['accuracy']):+.1f} points)")

    people = sorted(base["per_person"])
    diff = np.array([100*(best["per_person"][p]-base["per_person"][p])
                     for p in people])
    md, sd = diff.mean(), diff.std(ddof=1)
    t = md/(sd/np.sqrt(len(diff))) if sd else float("inf")
    print(f"\npaired vs baseline: " +
          "  ".join(f"{p} {v:+.1f}" for p, v in zip(people, diff)))
    print(f"mean {md:+.1f}  SD {sd:.1f}  t {t:.2f} (crit 3.182) -> "
          f"{'SEPARABLE' if abs(t) > 3.182 else 'NOT separable'}   "
          f"wins {int((diff>0).sum())}/{len(diff)}")

    print(f"\n{'='*104}")
    print(classification_report(best["y_true"], best["y_pred"], labels=THREE,
                                digits=3, zero_division=0))
    cm = confusion_matrix(best["y_true"], best["y_pred"], labels=THREE)
    print("confusion (rows=true):")
    print(f"{'':<10}" + "".join(f"{c:>10}" for c in THREE))
    for i, c in enumerate(THREE):
        print(f"{c:<10}" + "".join(f"{v:>10}" for v in cm[i]))
    p = os.path.join(QC, "branch_a_subset_confusion.png")
    plot_cm(cm, THREE, f"Branch A - {best['subset']}",
            f"leave-one-person-out {100*best['accuracy']:.1f}%", p)
    print("\nconfusion ->", p)

    out = [{k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
           for r in rows]
    with open(os.path.join(HERE, "branch_a_subsets.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("results -> branch_a_subsets.json")
    print("\nNOTE: 10 subsets were compared on the same folds. The winner's")
    print("margin is therefore optimistic; the honest claim is the mechanism")
    print("(temporal for drowsy, instantaneous for yawning), not the decimal.")


if __name__ == "__main__":
    main()
