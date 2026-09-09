"""Causal prediction smoothing - closing the gap between how we score and how
the system actually runs.

Every number so far treats each frame as an independent decision. The deployed
detector does not: it requires EYE_AR_CONSEC_FRAMES = 15 before flagging a
micro-sleep, because no driver-monitoring system alarms on a single frame.

That mismatch costs us precisely where we are weakest. On FL3D the model raises
976 false `drowsy` flags on alert frames, and drowsy precision is 0.579 while
recall is 0.802. If those false flags are ISOLATED frames while true drowsiness
is SUSTAINED - and we measured true episodes at a median 142 consecutive closed
video-frames - then a causal majority filter should remove most false alarms
and almost no true ones.

Strictly causal: the label at frame i is the mode of predictions [i-k+1 .. i].
No future frames, so the offline number stays achievable in real time. Windows
never cross a clip or video boundary.

    python experiment_smoothing.py
"""
import os
import sys
import json
import collections

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_temporal import temporal_block, NAMES, parse
from experiment_external import fl3d_sequences, FEATURE_SETS

THREE = ["normal", "drowsy", "yawning"]
WINDOW = 6
WIDTHS = (1, 3, 5, 7, 9, 11)


def smooth(pred, k):
    """Causal mode filter over the last k predictions."""
    if k <= 1:
        return pred.copy()
    out = np.empty_like(pred)
    for i in range(len(pred)):
        seg = pred[max(0, i - k + 1):i + 1]
        vals, cnt = np.unique(seg, return_counts=True)
        # ties resolved toward the most recent value, so the filter cannot
        # invent a class that is not currently being observed
        best = vals[cnt == cnt.max()]
        out[i] = pred[i] if pred[i] in best else best[0]
    return out


def our_data(which="absolute"):
    d = np.load(os.path.join(HERE, "features_temporal.npz"), allow_pickle=True)
    G, T, y, persons, files = (d["X_geo"], d["X_temporal"], d["y"],
                               d["persons"], d["files"])
    ti = {n: i for i, n in enumerate([str(x) for x in d["temporal_names"]])}
    X = np.column_stack([T[:, ti[n]] for n in FEATURE_SETS[which]] + [G[:, 4]])
    keep = np.isin(y, THREE)
    meta = [parse(f) for f in files]
    grp = np.array([f"{m[0]}|{m[1]}|{m[2]}" for m in meta])
    order = np.array([m[3] for m in meta])
    return X[keep], y[keep], persons[keep], grp[keep], order[keep]


def report(tag, yt, yp, groups, order):
    print(f"\n{tag}")
    print(f"{'k':>3}{'acc':>8}{'bal':>8}{'macroF1':>9}"
          f"{'drowsy P':>10}{'drowsy R':>10}{'false alarms':>14}")
    print("-" * 62)
    rows = []
    for k in WIDTHS:
        sm = np.empty_like(yp)
        for g in np.unique(groups):
            m = groups == g
            idx = np.where(m)[0][np.argsort(order[m])]
            sm[idx] = smooth(yp[idx], k)
        acc = accuracy_score(yt, sm)
        bal = balanced_accuracy_score(yt, sm)
        pr, rc, f1, _ = precision_recall_fscore_support(
            yt, sm, labels=THREE, zero_division=0)
        mf = f1.mean()
        fa = int(((yt == "normal") & (sm == "drowsy")).sum())
        print(f"{k:>3}{100*acc:>7.1f}%{100*bal:>7.1f}%{100*mf:>8.1f}%"
              f"{100*pr[1]:>9.1f}%{100*rc[1]:>9.1f}%{fa:>14}")
        rows.append({"k": k, "accuracy": float(acc), "balanced": float(bal),
                     "macro_f1": float(mf), "drowsy_p": float(pr[1]),
                     "drowsy_r": float(rc[1]), "false_alarms": fa})
    return rows


def main():
    out = {}

    # ---------- our data, leave-one-person-out ----------
    X, y, persons, grp, order = our_data()
    yt, yp, gt, ot = [], [], [], []
    for p in sorted(set(persons.tolist())):
        te = persons == p
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="linear", C=1, class_weight="balanced",
                                random_state=0)).fit(X[~te], y[~te])
        yt.append(y[te]); yp.append(clf.predict(X[te]))
        gt.append(grp[te]); ot.append(order[te])
    out["ours"] = report("OUR DATA - leave-one-person-out",
                         np.concatenate(yt), np.concatenate(yp),
                         np.concatenate(gt), np.concatenate(ot))

    # ---------- FL3D, trained on all of ours ----------
    seqs = fl3d_sequences()
    ti = {n: i for i, n in enumerate(NAMES)}
    Xf, yf, gf, of = [], [], [], []
    for v, ear, mar, lab in seqs:
        T = temporal_block(ear, mar, WINDOW)
        Xf.append(np.column_stack(
            [T[:, ti[n]] for n in FEATURE_SETS["absolute"]] + [mar]))
        yf += list(lab); gf += [v] * len(lab); of += list(range(len(lab)))
    Xf = np.vstack(Xf); yf = np.array(yf)
    gf = np.array(gf); of = np.array(of)
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel="linear", C=1, class_weight="balanced",
                            random_state=0)).fit(X, y)
    out["fl3d"] = report(
        "FL3D - trained on ours, tested on 44 unseen drivers",
        yf, clf.predict(Xf), gf, of)

    print(f"\n{'='*70}")
    for name in ("ours", "fl3d"):
        base = out[name][0]
        best = max(out[name], key=lambda r: r["macro_f1"])
        print(f"{name:<6} k=1 -> macroF1 {100*base['macro_f1']:.1f}%, "
              f"{base['false_alarms']} false alarms")
        print(f"{'':<6} k={best['k']} -> macroF1 {100*best['macro_f1']:.1f}% "
              f"({100*(best['macro_f1']-base['macro_f1']):+.1f}), "
              f"{best['false_alarms']} false alarms "
              f"({100*(best['false_alarms']-base['false_alarms'])/max(base['false_alarms'],1):+.0f}%)")
    print("\nSmoothing is free: no new data, no new features, and it makes the")
    print("evaluation match how the detector actually runs.")
    with open(os.path.join(HERE, "smoothing.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("results -> smoothing.json")


if __name__ == "__main__":
    main()
