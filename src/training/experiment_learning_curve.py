"""Does the CNN lose because of the METHOD, or because of the DATA?

Track 2A equalises temporal context. This equalises nothing and instead varies
the one thing we suspect is decisive: how much training data each branch gets.

The report's conclusion hinges on it. "Traditional ML beat deep learning" is a
weak claim. "Deep learning lost at this data scale, and here is the curve
showing where it would stop losing" is a finding - and it is falsifiable, which
is the point.

PREDICTIONS, recorded before running:

  1. Branch A is FLAT. Three geometric features with a linear SVM saturate
     within a few hundred frames; more frames of the same four faces add
     almost nothing.
  2. The CNN RISES monotonically and has not plateaued at 100%. 2.2M
     parameters on ~595 images is the overparameterised regime.
  3. Extrapolating the CNN's trend gives the data volume at which it would
     reach Branch A - a concrete, quotable number for the report.

If instead the CNN is also flat, the data-scale explanation is WRONG and the
honest conclusion becomes that hand-crafted PERCLOS is simply better suited to
this problem. Either outcome is publishable; only the unfalsifiable version is
not.

DESIGN

Outer loop is the same leave-one-person-out used everywhere else. Within each
fold the training pool (3 persons) is subsampled to a fraction f, and BOTH
branches are retrained on that identical subset and scored on the held-out
person. Same rows, same folds, same classifier for Branch A.

Subsampling is stratified by (person, class) and drawn at CLIP granularity
where possible: consecutive frames are near-duplicates, so removing random
frames would shrink the count without shrinking the information, and the curve
would look flatter than the truth.

    python experiment_learning_curve.py --probe
    python experiment_learning_curve.py --device cuda
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "training")))
from experiment_cnn_temporal import (train_cnn, encode, groups_of, svm,
                                     THREE, WINDOW, DROWSY)
from extract_temporal import parse

FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SEEDS = (0, 1, 2)          # repeats at each fraction; variance is large at 0.25


def subsample(persons, y, files, keep_frac, rng):
    """Keep a contiguous PREFIX of fraction `keep_frac` from every clip.

    Clip-granularity dropping was the obvious design and it does not work here:
    11 of the 12 (person, class) groups contain exactly TWO clips, so the only
    achievable fractions would be 0.50 and 1.00. Requesting 0.25/0.50/0.75/1.00
    would silently collapse to two distinct points while still plotting four -
    a curve that looks valid and is not.

    Contiguous prefixes give fine-grained control and stay self-consistent for
    BOTH branches. Branch A's temporal features are causal: a frame at position
    j uses only j-5..j, all of which are inside the prefix. A random frame
    subsample would instead leave each surviving frame's window referring to
    frames excluded from training.

    HONEST LIMITATION, to state in the report: frames within a clip are
    correlated, so halving a clip removes less information than dropping a
    whole clip would. The curve is therefore optimistic at small fractions,
    and the x-axis is reported as raw frame count rather than as an effective
    sample size.

    A plain prefix is deterministic, which would make every seed draw the SAME
    subset and reduce the repeats to measuring CNN-initialisation variance
    only. So the per-clip fraction is jittered around the target with mean
    keep_frac, giving genuinely different subsets per seed while every kept
    span remains a prefix. A random segment START would also vary the subset
    but is rejected: frames near the start would carry windows referring to
    excluded frames.
    """
    if keep_frac >= 1.0:
        return np.ones(len(y), bool)
    by = {}
    for i, f in enumerate(files):
        m = parse(f)
        by.setdefault((m[0], m[1], m[2]), []).append((m[3], i))
    mask = np.zeros(len(y), bool)
    jit = 0.12
    for key in sorted(by):
        idx = [i for _, i in sorted(by[key])]
        f = float(np.clip(keep_frac + rng.uniform(-jit, jit), 0.05, 1.0))
        k = max(1, int(round(len(idx) * f)))
        mask[idx[:k]] = True
    return mask


def branch_a_matrix(files):
    """The winning Branch A subset E, aligned to the crop rows."""
    d = np.load(os.path.join(HERE, "features_temporal.npz"), allow_pickle=True)
    ti = {str(n): i for i, n in enumerate(d["temporal_names"])}
    keep = np.isin(d["y"], THREE)
    if not np.array_equal(d["files"][keep], files):
        raise SystemExit("features_temporal.npz rows do not match the crops")
    T, G = d["X_temporal"][keep], d["X_geo"][keep]
    return np.column_stack([T[:, ti["perclos_20"]], G[:, 4],
                            T[:, ti["closed_run_w"]]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    # MEASURED on this laptop (16 logical / 10 physical, i7-13620H), 1 epoch
    # on 609 images: 14 threads + 4 workers ~20s, 1 thread ~14.3s,
    # 4 threads + 0 workers **5.1s**. Oversubscribing torch threads across
    # dataloader workers left the machine at 19% CPU load - 59 threads
    # contending for 10 physical cores. Overridable via DG_THREADS.
    torch.set_num_threads(int(os.environ.get("DG_THREADS", "4")))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device

    d = np.load(os.path.join(HERE, "crops_aligned.npz"), allow_pickle=True)
    keep = np.isin(d["y"], THREE)
    X, y = d["faces"][keep], d["y"][keep]
    persons, files = d["persons"][keep], d["files"][keep]
    people = sorted(set(persons.tolist()))
    XA = branch_a_matrix(files)
    seeds = SEEDS[:max(1, a.seeds)]

    n_train = sum(1 for _ in people) and int((persons != people[0]).sum())
    print(f"n={len(y)}  people={people}  device={dev}")
    print(f"fractions={FRACTIONS}  seeds={seeds}  "
          f"=> {len(people)*len(FRACTIONS)*len(seeds)} CNN trainings")

    if a.probe:
        t0 = time.time()
        te = persons == people[0]
        train_cnn(X[~te], y[~te], 1, a.batch, a.lr, a.workers, dev)
        dt = time.time() - t0
        total = sum(f for f in FRACTIONS) * len(people) * len(seeds)
        print(f"\n1 epoch on {int((~te).sum())} images: {dt:.1f}s")
        print(f"estimated full run: {dt*a.epochs*total/60:.0f} min on {dev}")
        return

    t0 = time.time()
    rows = []
    for frac in FRACTIONS:
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + int(frac * 100))
            ya, pa, yb, pb, ntr = [], [], [], [], []
            for p in people:
                te = persons == p
                pool = ~te
                sub = subsample(persons, y, files, frac, rng) & pool
                if len(set(y[sub].tolist())) < 3:
                    continue
                ntr.append(int(sub.sum()))
                # Branch A on exactly the same subset
                pred_a = svm().fit(XA[sub], y[sub]).predict(XA[te])
                ya.append(y[te]); pa.append(pred_a)
                # CNN on exactly the same subset
                m = train_cnn(X[sub], y[sub], a.epochs, a.batch, a.lr,
                              a.workers, dev, seed=seed)
                _, P = encode(m, X, dev, a.workers)
                pred_b = np.array([THREE[i] for i in P[te].argmax(1)])
                yb.append(y[te]); pb.append(pred_b)
                del m
            ya, pa = np.concatenate(ya), np.concatenate(pa)
            yb, pb = np.concatenate(yb), np.concatenate(pb)
            r = {"fraction": frac, "seed": seed,
                 "mean_train_frames": float(np.mean(ntr)),
                 "branch_a_acc": float(accuracy_score(ya, pa)),
                 "cnn_acc": float(accuracy_score(yb, pb)),
                 "branch_a_f1": float(precision_recall_fscore_support(
                     ya, pa, labels=THREE, average="macro", zero_division=0)[2]),
                 "cnn_f1": float(precision_recall_fscore_support(
                     yb, pb, labels=THREE, average="macro", zero_division=0)[2])}
            rows.append(r)
            print(f"  frac {frac:.2f} seed {seed}  n_train {r['mean_train_frames']:.0f}"
                  f"   BranchA {100*r['branch_a_acc']:.1f}%   "
                  f"CNN {100*r['cnn_acc']:.1f}%   ({time.time()-t0:.0f}s)",
                  flush=True)
            # Write after EVERY point, not at the end. A pod died 8 points into
            # this stage on 14 Aug and destroyed an hour of completed work,
            # because the JSON was only produced once the whole stage finished.
            # Stage-level snapshots do not protect a long single stage.
            with open(os.path.join(HERE, "learning_curve.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"epochs": a.epochs, "fractions": list(FRACTIONS),
                           "seeds": list(seeds), "rows": rows,
                           "aggregate": [], "partial": True,
                           "minutes": (time.time() - t0) / 60}, f, indent=1)

    # ---------------------------------------------------------- aggregate
    print("\n" + "=" * 78)
    print(f"{'frac':>6}{'n_train':>10}{'BranchA':>12}{'CNN':>12}{'gap':>10}")
    print("-" * 78)
    agg = []
    for frac in FRACTIONS:
        g = [r for r in rows if r["fraction"] == frac]
        if not g:
            continue
        a_m = np.mean([r["branch_a_acc"] for r in g])
        b_m = np.mean([r["cnn_acc"] for r in g])
        a_s = np.std([r["branch_a_acc"] for r in g])
        b_s = np.std([r["cnn_acc"] for r in g])
        nt = np.mean([r["mean_train_frames"] for r in g])
        agg.append({"fraction": frac, "n_train": float(nt),
                    "branch_a": float(a_m), "branch_a_sd": float(a_s),
                    "cnn": float(b_m), "cnn_sd": float(b_s)})
        print(f"{frac:>6.2f}{nt:>10.0f}{100*a_m:>10.1f}+-{100*a_s:<3.1f}"
              f"{100*b_m:>10.1f}+-{100*b_s:<3.1f}{100*(a_m-b_m):>9.1f}")
    print("=" * 78)

    if len(agg) >= 2:
        da = agg[-1]["branch_a"] - agg[0]["branch_a"]
        db = agg[-1]["cnn"] - agg[0]["cnn"]
        print(f"\nslope over the measured range "
              f"({agg[0]['n_train']:.0f} -> {agg[-1]['n_train']:.0f} frames):")
        print(f"  Branch A {100*da:+.1f} points   CNN {100*db:+.1f} points")
        print("  prediction 1 (Branch A flat)        : "
              f"{'HELD' if abs(da) < 0.03 else 'REFUTED'}")
        print("  prediction 2 (CNN still rising)     : "
              f"{'HELD' if db > 0.03 else 'REFUTED'}")
        # log-linear extrapolation to the crossing point
        if db > 0.01 and agg[-1]["cnn"] < agg[-1]["branch_a"]:
            xs = np.log([r["n_train"] for r in agg])
            ys = np.array([r["cnn"] for r in agg])
            k, c = np.polyfit(xs, ys, 1)
            if k > 0:
                need = float(np.exp((agg[-1]["branch_a"] - c) / k))
                print(f"  log-linear extrapolation: the CNN would reach "
                      f"Branch A's {100*agg[-1]['branch_a']:.1f}% at roughly "
                      f"{need:,.0f} training frames "
                      f"({need/agg[-1]['n_train']:.1f}x what we have).")
                print("  (extrapolation, not measurement - state it as such)")

    out = os.path.join(HERE, "learning_curve.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"epochs": a.epochs, "fractions": list(FRACTIONS),
                   "seeds": list(seeds), "rows": rows, "aggregate": agg,
                   "minutes": (time.time() - t0) / 60}, f, indent=1)
    print(f"\nresults -> {out}   ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
