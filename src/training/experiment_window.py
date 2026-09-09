"""How long should the PERCLOS window be?

1.2 s was chosen because our clips are only ~6.2 s, not because the method
wants it. The PERCLOS literature (Wierwille) uses minute-scale windows: the
whole point of the metric is that drowsiness accumulates over time. Our
recording protocol, not the physics, set that number - so sweep it.

Ceiling: a window cannot exceed the clip, and a window approaching clip length
stops measuring "recent eye state" and starts measuring "this clip's average",
which is a proxy for the clip label. Where accuracy keeps climbing as the
window approaches clip length, that is the warning sign, not a result - so the
window/clip ratio is printed alongside.

    python experiment_window.py
"""
import os
import json
import collections

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from experiment_branch_a import loso, THREE
from extract_temporal import parse, temporal_block, NAMES, EAR_I, MAR_I

HERE = os.path.dirname(os.path.abspath(__file__))
# subset E, the winner: temporal drowsiness cue + instantaneous yawn cue + run
WINNER = ["perclos_20", "MAR", "closed_run_w"]


def build(X, meta, window):
    clips = collections.defaultdict(list)
    for i, m in enumerate(meta):
        clips[(m[0], m[1], m[2])].append((m[3], i))
    T = np.zeros((len(meta), len(NAMES)), np.float32)
    for _, lst in clips.items():
        lst.sort()
        idx = np.array([i for _, i in lst])
        T[idx] = temporal_block(X[idx, EAR_I], X[idx, MAR_I], window)
    return T, clips


def main():
    d = np.load(os.path.join(HERE, "features.npz"), allow_pickle=True)
    X, y, persons, files = d["X_geo"], d["y"], d["persons"], d["files"]
    meta = [parse(f) for f in files]
    _, clips = build(X, meta, 6)
    lens = [len(v) for v in clips.values()]
    med = int(np.median(lens))
    print(f"{len(clips)} clips, median {med} sampled frames "
          f"({med*5/25:.1f}s at 25 fps, frames kept every 5th)\n")

    gi = {"MAR": 4}
    ti = {n: i for i, n in enumerate(NAMES)}

    print(f"{'window':>7}{'seconds':>9}{'w/clip':>8}{'acc':>8}{'F1':>8}"
          f"{'drowsy':>8}{'yawn':>7}   per-person")
    print("-" * 92)
    rows = []
    for w in (2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30):
        T, _ = build(X, meta, w)
        cols = np.column_stack([T[:, ti["perclos_20"]],
                                X[:, gi["MAR"]],
                                T[:, ti["closed_run_w"]]])
        yt, yp, pp = loso(cols, y, persons, THREE)
        acc = accuracy_score(yt, yp)
        _, _, f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0)
        _, rcs, _, _ = precision_recall_fscore_support(
            yt, yp, labels=THREE, zero_division=0)
        spread = " ".join(f"{k[:3]} {100*v:.0f}" for k, v in sorted(pp.items()))
        ratio = w / med
        flag = "  <- window ~= clip" if ratio > 0.6 else ""
        print(f"{w:>7}{w*5/25:>8.1f}s{ratio:>8.2f}{100*acc:>7.1f}%{100*f1:>7.1f}%"
              f"{100*rcs[1]:>7.1f}%{100*rcs[2]:>6.1f}%   {spread}{flag}")
        rows.append({"window": w, "seconds": w*5/25, "clip_ratio": ratio,
                     "accuracy": float(acc), "f1": float(f1),
                     "drowsy_recall": float(rcs[1]),
                     "yawning_recall": float(rcs[2]),
                     "per_person": {k: float(v) for k, v in pp.items()}})

    safe = [r for r in rows if r["clip_ratio"] <= 0.6]
    best = max(safe, key=lambda r: r["accuracy"])
    print(f"\nbest window within the safe range (<=60% of clip): "
          f"{best['window']} frames = {best['seconds']:.1f}s -> "
          f"{100*best['accuracy']:.1f}%")
    big = max(rows, key=lambda r: r["accuracy"])
    if big["clip_ratio"] > 0.6:
        print(f"NOTE: {big['window']} frames scores higher "
              f"({100*big['accuracy']:.1f}%) but spans "
              f"{100*big['clip_ratio']:.0f}% of a clip - that is measuring the "
              f"clip, not the driver. Not used.")
    with open(os.path.join(HERE, "branch_a_window_sweep.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("results -> branch_a_window_sweep.json")


if __name__ == "__main__":
    main()
