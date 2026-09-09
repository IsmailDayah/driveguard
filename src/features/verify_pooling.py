"""Verify the Track 2A pooling before spending 16 CNN trainings on it.

A pooling bug would not crash. It would quietly produce a plausible number -
and if a future frame leaked into a window, that number would be BETTER than
the truth, which is the worst kind of wrong. So every claim the pooling makes
is checked here, mathematically where possible and visually where not.

  1  grouping     - clips are the same (person, class, clip) units Branch A
                    uses, contiguous and frame-ordered
  2  causality    - perturbing frame j+1 must leave row j bit-identical. This
                    is the decisive test: a centred or forward-looking window
                    fails it immediately.
  3  window span  - window_frac must equal k/w with k = min(j+1, w), which
                    pins the window to exactly the trailing frames
  4  arithmetic   - pooled means/std/min/max recomputed independently
  5  neural       - the PERCLOS analogue must equal the fraction of the window
     PERCLOS       above the threshold, by direct count
  6  visual       - render the frames of real windows so a human can confirm a
                    "1.2s window" is 6 consecutive frames of one clip

    python verify_pooling.py
"""
import os
import sys
import collections

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "training")))
from experiment_cnn_temporal import (groups_of, pool_prob, pool_emb,
                                     PROB_T, RUN_T, DROWSY, THREE, WINDOW)
from extract_temporal import parse

QC = os.path.join(HERE, "qc")
os.makedirs(QC, exist_ok=True)
BG, TXT, MUTED = "#0A1628", "#F1F5F9", "#93A6BD"
W = WINDOW
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


d = np.load(os.path.join(HERE, "crops_aligned.npz"), allow_pickle=True)
keep = np.isin(d["y"], THREE)
faces, y = d["faces"][keep], d["y"][keep]
persons, files = d["persons"][keep], d["files"][keep]
n = len(y)
groups = groups_of(files)
rng = np.random.default_rng(0)

print(f"n={n}  clips={len(groups)}  window={W} frames = {W/5:.1f}s @5fps\n")

# ---------------------------------------------------------------- 1 grouping
print("1. GROUPING")
sizes = [len(v) for v in groups.values()]
check("every row belongs to exactly one clip",
      sum(sizes) == n and len({i for v in groups.values() for i in v}) == n,
      f"{sum(sizes)} rows over {len(groups)} clips")
bad_order = 0
mixed = 0
for k, idx in groups.items():
    fr = [parse(files[i])[3] for i in idx]
    if fr != sorted(fr):
        bad_order += 1
    meta = {(parse(files[i])[0], parse(files[i])[1], parse(files[i])[2]) for i in idx}
    if len(meta) != 1 or next(iter(meta)) != k:
        mixed += 1
check("clips are frame-ordered", bad_order == 0, f"{bad_order} unsorted")
check("no clip mixes person/class/clip", mixed == 0, f"{mixed} mixed")
check("one person per clip",
      all(len({persons[i] for i in idx}) == 1 for idx in groups.values()))
print(f"   clip length: min {min(sizes)}  median {int(np.median(sizes))}  "
      f"max {max(sizes)}")

# --------------------------------------------------------------- 2 causality
print("\n2. CAUSALITY  (the decisive test)")
P = rng.random((n, 3)).astype(np.float32)
P /= P.sum(1, keepdims=True)
E = rng.random((n, 16)).astype(np.float32)
big = max(groups.items(), key=lambda kv: len(kv[1]))
idx = big[1]
j = len(idx) // 2
base_p = pool_prob(P, idx, W)
base_e = pool_emb(E, idx, W, ["mean", "std", "min", "max"])
P2, E2 = P.copy(), E.copy()
for t in idx[j + 1:]:
    P2[t] = rng.random(3); P2[t] /= P2[t].sum()
    E2[t] = rng.random(16)
alt_p = pool_prob(P2, idx, W)
alt_e = pool_emb(E2, idx, W, ["mean", "std", "min", "max"])
check("prob rows 0..j unchanged when the FUTURE is rewritten",
      np.array_equal(base_p[:j + 1], alt_p[:j + 1]),
      f"clip {big[0]}, j={j} of {len(idx)}")
check("emb  rows 0..j unchanged when the FUTURE is rewritten",
      np.array_equal(base_e[:j + 1], alt_e[:j + 1]))
check("rows after j DO change (perturbation was real)",
      not np.array_equal(base_p[j + 1:], alt_p[j + 1:]))
# and the converse: rewriting the PAST must change row j
P3 = P.copy()
for t in idx[:j]:
    P3[t] = rng.random(3); P3[t] /= P3[t].sum()
check("row j changes when the PAST is rewritten (window is not width-1)",
      not np.array_equal(pool_prob(P3, idx, W)[j], base_p[j]))

# ------------------------------------------------------------- 3 window span
print("\n3. WINDOW SPAN")
span_ok = True
for k, ix in list(groups.items())[:40]:
    pp = pool_prob(P, ix, W)
    for j2 in range(len(ix)):
        expect = min(j2 + 1, W) / W
        if abs(pp[j2, 12] - expect) > 1e-6:
            span_ok = False
check("window_frac == min(j+1, w)/w for every row", span_ok,
      f"checked {sum(len(v) for v in list(groups.values())[:40])} rows")
check("first row of a clip has a 1-frame window",
      abs(pool_prob(P, idx, W)[0, 12] - 1.0 / W) < 1e-6)
check("row w-1 onward has a full window",
      abs(pool_prob(P, idx, W)[W - 1, 12] - 1.0) < 1e-6)

# --------------------------------------------------------------- 4 arithmetic
print("\n4. ARITHMETIC  (independent recomputation)")
man_ok = True
sub = E[idx]
pooled = pool_emb(E, idx, W, ["mean", "std", "min", "max"])
for j2 in rng.choice(len(idx), min(25, len(idx)), replace=False):
    seg = sub[max(0, j2 - W + 1):j2 + 1]
    ref = np.concatenate([seg.mean(0),
                          seg.std(0) if len(seg) > 1 else np.zeros(16, np.float32),
                          seg.min(0), seg.max(0)])
    if not np.allclose(pooled[j2], ref, atol=1e-6):
        man_ok = False
check("mean/std/min/max match a direct recomputation", man_ok)

# ----------------------------------------------------------- 5 neural PERCLOS
print("\n5. NEURAL PERCLOS  (must be a literal fraction-of-window)")
pp = pool_prob(P, idx, W)
per_ok = run_ok = True
dvals = P[idx, DROWSY]
for j2 in range(len(idx)):
    seg = dvals[max(0, j2 - W + 1):j2 + 1]
    for c, t in enumerate(PROB_T):
        if abs(pp[j2, c] - (seg > t).mean()) > 1e-6:
            per_ok = False
    best = cur = 0
    for v in seg:
        cur = cur + 1 if v > RUN_T else 0
        best = max(best, cur)
    if abs(pp[j2, 7] - best / W) > 1e-6:
        run_ok = False
check("fraction above each threshold is exact", per_ok,
      f"thresholds {PROB_T}")
check("longest run above 0.5 is exact and normalised by w", run_ok)
check("all pooled prob features are finite", np.isfinite(pp).all())

# ------------------------------------------------------------------ 6 visual
print("\n6. VISUAL  (what a 1.2s window actually contains)")
picks = []
for cls in THREE:
    cand = [(k, v) for k, v in groups.items() if k[1] == cls and len(v) > W + 4]
    k, ix = cand[rng.integers(len(cand))]
    j2 = int(rng.integers(W - 1, len(ix)))
    picks.append((cls, k, ix, j2))

fig, axes = plt.subplots(3, W, figsize=(1.55 * W, 5.6), facecolor=BG,
                         squeeze=False)
for r, (cls, k, ix, j2) in enumerate(picks):
    lo = max(0, j2 - W + 1)
    win = ix[lo:j2 + 1]
    for c in range(W):
        ax = axes[r][c]
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        if c < len(win):
            i = win[c]
            ax.imshow(faces[i], cmap="gray", vmin=0, vmax=255)
            fr = parse(files[i])[3]
            last = (c == len(win) - 1)
            ax.set_title(f"f{fr:05d}" + ("  <- now" if last else ""),
                         color="#FFD166" if last else MUTED, fontsize=8,
                         fontweight="bold" if last else "normal")
            for s in ax.spines.values():
                s.set_color("#FFD166" if last else "#2B4263")
                s.set_linewidth(2.0 if last else 0.8)
        else:
            ax.axis("off")
    axes[r][0].set_ylabel(f"{k[0]}\n{cls}\nclip {k[2]}", color=TXT,
                          fontsize=9, fontweight="bold", rotation=0,
                          labelpad=38, va="center")
fig.suptitle(f"One causal {W/5:.1f}s window ({W} frames @5fps) per class - "
             f"trailing only, never crossing a clip",
             color=TXT, fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
p1 = os.path.join(QC, "pooling_windows.png")
fig.savefig(p1, dpi=150, facecolor=BG); plt.close(fig)
print(f"   -> {p1}")

# frame-index continuity inside the rendered windows
cont = all(
    [parse(files[i])[3] for i in ix[max(0, j2 - W + 1):j2 + 1]] ==
    list(range(parse(files[ix[max(0, j2 - W + 1)]])[3],
               parse(files[ix[j2]])[3] + 1, 5))
    for _, _, ix, j2 in picks)
check("rendered windows are consecutive 5-frame-stride samples", cont)

print("\n" + "=" * 66)
print("POOLING VERIFIED - safe to train" if ok_all
      else "POOLING HAS A DEFECT - do not run the experiment")
print("=" * 66)
sys.exit(0 if ok_all else 1)
