"""Aggregate the multi-seed Track 2A runs into mean +- sd.

Why this exists: the single-seed runs disagreed with each other about the SIGN
of the temporal effect (+1.5 points under one seeding, -5.7 under another).
A conclusion that flips with the random seed is not a conclusion, so every 2A
number is reported as a mean over seeds with its spread, and any claim whose
effect is smaller than the spread is explicitly marked unresolved.

Branch A is deterministic (a linear SVM on fixed features) and must therefore
appear with sd = 0.000 in every run. That is used as a correctness check: a
non-zero Branch A spread would mean the seeds are not scoring identical rows,
and the whole comparison would be invalid.

    python aggregate_seeds.py --glob "cnn_temporal_seed*.json"
"""
import os
import glob
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
QC = os.path.join(HERE, "qc")
BG, TXT, MUTED = "#0A1628", "#F1F5F9", "#93A6BD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="cnn_temporal_seed*.json")
    ap.add_argument("--dir", default=HERE)
    a = ap.parse_args()
    os.makedirs(QC, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.dir, a.glob)))
    if not paths:
        raise SystemExit(f"no files matching {a.glob} in {a.dir}")
    print(f"aggregating {len(paths)} seed runs:")
    for p in paths:
        print(f"   {os.path.basename(p)}")

    runs = [json.load(open(p, encoding="utf-8")) for p in paths]
    by = {}
    for r in runs:
        for row in r["results"]:
            by.setdefault(row["tag"], []).append(row)

    print("\n" + "=" * 86)
    print(f"{'model':<36}{'acc mean':>10}{'sd':>8}{'macroF1':>10}{'sd':>8}"
          f"{'n':>4}{'range':>12}")
    print("-" * 86)
    agg = []
    for tag, rows in by.items():
        acc = np.array([x["accuracy"] for x in rows])
        f1 = np.array([x["macro_f1"] for x in rows])
        agg.append({"tag": tag, "n": len(rows),
                    "acc_mean": float(acc.mean()), "acc_sd": float(acc.std(ddof=0)),
                    "acc_min": float(acc.min()), "acc_max": float(acc.max()),
                    "f1_mean": float(f1.mean()), "f1_sd": float(f1.std(ddof=0))})
    agg.sort(key=lambda r: -r["acc_mean"])
    for r in agg:
        print(f"{r['tag']:<36}{100*r['acc_mean']:>9.1f}%{100*r['acc_sd']:>8.1f}"
              f"{100*r['f1_mean']:>9.1f}%{100*r['f1_sd']:>8.1f}{r['n']:>4}"
              f"{100*r['acc_min']:>6.1f}-{100*r['acc_max']:<5.1f}")
    print("=" * 86)

    # ---- correctness check: Branch A must be deterministic ----------------
    ba = [r for r in agg if r["tag"].startswith("BranchA")]
    if ba:
        sd = ba[0]["acc_sd"]
        ok = sd < 1e-9
        print(f"\n[{'PASS' if ok else 'FAIL'}] Branch A sd = {sd:.2e} "
              f"(must be 0 - it is a deterministic SVM on fixed features)")
        if not ok:
            print("      Non-zero spread means the seeds are NOT scoring the")
            print("      same rows. Do not trust any comparison in this table.")

    # ---- is the temporal effect bigger than the seed noise? ---------------
    print("\n" + "=" * 86)
    print("IS THE TEMPORAL EFFECT RESOLVABLE ABOVE SEED NOISE?")
    print("=" * 86)
    # PAIRED analysis. The two arms share a seed AND an encoder - only the
    # window differs - so comparing their independent spreads throws away the
    # pairing and badly understates power. The per-seed difference is the right
    # statistic, and a consistent SIGN across seeds is stronger evidence than
    # any mean at n=3.
    print("paired per-seed differences (w6 - w1); sign consistency is the test")
    for rep in ("prob", "emb"):
        for scheme in ("nested", "flat"):
            t1 = f"CNN {rep} + no window [{scheme}]"
            t6 = f"CNN {rep} + 1.2s [{scheme}]"
            if t1 not in by or t6 not in by:
                continue
            a1 = {}, {}
            d1 = {r.get("seed_idx", i): r["accuracy"] for i, r in enumerate(by[t1])}
            d6 = {r.get("seed_idx", i): r["accuracy"] for i, r in enumerate(by[t6])}
            ks = sorted(set(d1) & set(d6))
            diffs = np.array([d6[k] - d1[k] for k in ks])
            if len(diffs) == 0:
                continue
            same_sign = bool(np.all(diffs > 0) or np.all(diffs < 0))
            verdict = ("CONSISTENT " + ("+" if diffs.mean() > 0 else "-")
                       if same_sign else "SIGN FLIPS - no effect")
            per = "  ".join(f"{100*x:+.1f}" for x in diffs)
            print(f"  {rep:<6} [{scheme:<6}] per-seed [{per}]  "
                  f"mean {100*diffs.mean():+.1f}   **{verdict}**")
    if len(runs) < 3:
        print(f"\n  NOTE: only {len(runs)} seeds - a spread from {len(runs)} "
              f"points is a weak estimate. Treat verdicts as provisional.")

    # ---- does anything beat Branch A? -------------------------------------
    if ba:
        ref = ba[0]["acc_mean"]
        print("\n" + "=" * 86)
        print(f"ANYTHING BEATING BRANCH A ({100*ref:.1f}%)?")
        print("=" * 86)
        beat = [r for r in agg if r["acc_mean"] > ref and not r["tag"].startswith("BranchA")]
        if beat:
            for r in beat:
                margin = r["acc_mean"] - ref
                sure = "beyond seed noise" if margin > r["acc_sd"] else "WITHIN seed noise"
                print(f"  {r['tag']:<36}{100*r['acc_mean']:.1f}% "
                      f"({100*margin:+.1f}, {sure})")
        else:
            print("  Nothing. Branch A remains the best model on our data.")

        # Branch A is a CONSTANT here (deterministic SVM, sd verified 0), so
        # "is arm X below Branch A?" is a one-sample t-test on the per-seed
        # gaps, not a two-sample comparison. That is the correct test and it
        # uses the pairing, but with n=3 it has very little power - a
        # non-significant result means "not shown", never "no difference".
        best = max((r for r in agg if not r["tag"].startswith("BranchA")),
                   key=lambda r: r["acc_mean"])
        gaps = np.array([ref - x["accuracy"] for x in by[best["tag"]]])
        n = len(gaps)
        print(f"\n  closest arm: {best['tag']}")
        print(f"    {100*best['acc_mean']:.1f}% +-{100*best['acc_sd']:.1f}   "
              f"gap to Branch A {100*gaps.mean():.1f} points")
        if n >= 2 and gaps.std(ddof=1) > 0:
            se = gaps.std(ddof=1) / np.sqrt(n)
            t = gaps.mean() / se
            # one-tailed critical values, df = n-1
            crit = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015}.get(n - 1, 1.96)
            sig = abs(t) > crit
            print(f"    per-seed gaps: "
                  f"[{'  '.join(f'{100*g:+.1f}' for g in gaps)}]")
            print(f"    one-sample t = {t:.2f} (df={n-1}, one-tailed crit "
                  f"{crit:.3f}) -> {'SIGNIFICANT' if sig else 'not significant'}")
            if not sig:
                print("    'not significant' at n=%d means NOT SHOWN, not "
                      "'no difference'." % n)
        else:
            print("    too few seeds for a test")

    # ------------------------------------------------------------- figure
    show = [r for r in agg if r["n"] >= 1][:12]
    show.sort(key=lambda r: r["acc_mean"])
    fig, ax = plt.subplots(figsize=(11, 0.52 * len(show) + 2.2), facecolor=BG)
    ax.set_facecolor("#12263F")
    ys = np.arange(len(show))
    cols = ["#22D3EE" if not r["tag"].startswith("BranchA") else "#F59E0B"
            for r in show]
    ax.barh(ys, [100 * r["acc_mean"] for r in show],
            xerr=[100 * r["acc_sd"] for r in show],
            color=cols, height=0.62,
            error_kw=dict(ecolor="#F1F5F9", capsize=4, lw=1.3))
    for i, r in enumerate(show):
        ax.text(100 * r["acc_mean"] + 100 * r["acc_sd"] + 0.8, i,
                f"{100*r['acc_mean']:.1f}", va="center", color=TXT,
                fontsize=9, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["tag"] for r in show], color=TXT, fontsize=9)
    ax.set_xlabel("LOSO accuracy (%)  -  error bars are sd over seeds",
                  color=TXT, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#2B4263")
    ax.set_title("Track 2A - mean over seeds, with spread\n"
                 "an effect smaller than its error bar is not a result",
                 color=TXT, fontsize=12.5, fontweight="bold", pad=12)
    fig.tight_layout()
    p = os.path.join(QC, "cnn_temporal_seeds.png")
    fig.savefig(p, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"\nfigure -> {p}")

    out = os.path.join(HERE, "cnn_temporal_aggregate.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": len(paths), "aggregate": agg}, f, indent=1)
    print(f"metrics -> {out}")


if __name__ == "__main__":
    main()
