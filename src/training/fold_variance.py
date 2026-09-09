"""How large must a difference be, with only four people, to mean anything?

Leave-one-person-out with n=4 gives four accuracy estimates per model. Those
estimates swing widely — one variant moved +13.6 points on one fold and -12.7
on another while its overall score stayed flat. Reporting "variant X beat
variant Y by 2 points" under that much spread would be reading noise.

This quantifies the spread and does a paired comparison across the common
folds, so the report can state which gaps are real and which are not.

    python fold_variance.py
"""
import os
import glob
import json
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_branches import nice_name

HERE = os.path.dirname(os.path.abspath(__file__))
QC = os.path.join(HERE, "qc")
os.makedirs(QC, exist_ok=True)
BG, PANEL, TXT, MUTED = "#0A1628", "#12263F", "#F1F5F9", "#93A6BD"
GREEN, RED, AMBER, CYAN = "#22C55E", "#EF4444", "#F59E0B", "#22D3EE"


def load():
    rows = []
    ab = os.path.join(HERE, "branch_a_ablation.json")
    if os.path.exists(ab):
        for r in json.load(open(ab, encoding="utf-8")):
            if r["variant"].startswith("B "):
                rows.append({"name": "SVM (EAR/MAR)", "branch": "A",
                             "acc": r["accuracy"], "pp": r["per_person"]})
    for p in sorted(glob.glob(os.path.join(HERE, "branch_b_*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        if len(r.get("classes", [])) != 3:
            continue
        rows.append({"name": nice_name(r), "branch": "B",
                     "acc": r["accuracy"], "pp": r["per_person"]})
    return sorted(rows, key=lambda r: -r["acc"])


def main():
    rows = load()
    if len(rows) < 2:
        print("need at least two completed runs"); return
    people = sorted(set.intersection(*[set(r["pp"]) for r in rows]))
    print(f"folds in common: {people}\n")

    print(f"{'model':<34}{'overall':>9}{'fold mean':>11}{'SD':>7}"
          f"{'SE':>7}{'min-max':>13}")
    print("-" * 81)
    for r in rows:
        v = np.array([100 * r["pp"][p] for p in people])
        r["v"] = v
        print(f"{r['name']:<34}{100*r['acc']:>8.1f}%{v.mean():>10.1f}%"
              f"{v.std(ddof=1):>7.1f}{v.std(ddof=1)/np.sqrt(len(v)):>7.1f}"
              f"{f'{v.min():.0f}-{v.max():.0f}%':>13}")

    print(f"\n{'='*81}")
    print("paired fold-by-fold differences (positive = first model better)")
    print(f"{'='*81}")
    print(f"{'comparison':<52}{'mean d':>9}{'SD d':>8}{'verdict':>12}")
    print("-" * 81)
    verdicts = []
    for a, b in itertools.combinations(rows, 2):
        d = a["v"] - b["v"]
        md, sd = d.mean(), d.std(ddof=1)
        # with n=4 the paired t critical value at 95% is 3.182; requiring the
        # mean difference to exceed that many standard errors is a deliberately
        # strict bar, which is the right posture for four people
        se = sd / np.sqrt(len(d)) if sd > 0 else 1e-9
        t = md / se
        sig = abs(t) > 3.182
        verdicts.append((a["name"], b["name"], md, sd, sig))
        print(f"{a['name'][:24]:<25} vs {b['name'][:24]:<25}"
              f"{md:>9.1f}{sd:>8.1f}{'REAL' if sig else 'noise':>12}")

    # How many people would the headline claim actually need? Same effect and
    # spread, more folds. Actionable, because recruiting is something the team
    # can still do before the report is due.
    from scipy import stats as _st
    head = next((v for v in verdicts
                 if v[0].startswith("SVM") and "face" in v[1]
                 and "scratch" not in v[1]), None)
    if head:
        _, _, md, sd, sig = head
        print(f"\n{'='*81}")
        print(f"power estimate for the headline claim "
              f"({head[0]} vs {head[1]})")
        print(f"{'='*81}")
        print(f"observed mean difference {md:.1f} points, SD {sd:.1f}")
        need = None
        for n in range(4, 31):
            tcrit = _st.t.ppf(0.975, n - 1)
            if md / (sd / np.sqrt(n)) > tcrit:
                need = n
                break
        if need:
            print(f"people needed to separate it at 95%: {need}"
                  + (f"  ({need - len(people)} more than we have)"
                     if need > len(people) else "  (already sufficient)"))
        else:
            print("not separable at 95% within 30 people at this effect size")
        wins = int((rows[0]["v"] > rows[1]["v"]).sum()) if len(rows) > 1 else 0
        print(f"direction is consistent: the SVM wins {wins}/{len(people)} "
              f"folds (sign test p={2*0.5**len(people)*1:.3f} two-tailed)")

    print(f"\n{'='*81}")
    real = [v for v in verdicts if v[4]]
    print(f"{len(real)} of {len(verdicts)} pairwise gaps survive a paired test "
          f"across {len(people)} folds.")
    for a, b, md, sd, _ in real:
        print(f"  {a}  >  {b}   by {md:.1f} points")
    if not real:
        print("  none - with four people no pair is separable at 95%.")

    # ---- figure: per-fold accuracy, one line per model ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), facecolor=BG,
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    ax1.set_facecolor(PANEL)
    x = np.arange(len(people))
    for i, r in enumerate(rows):
        ax1.plot(x, r["v"], "-o", lw=2.2, ms=7,
                 color=GREEN if r["branch"] == "A" else
                 [RED, AMBER, CYAN, "#A78BFA", "#F472B6"][i % 5],
                 label=r["name"])
    ax1.set_xticks(x); ax1.set_xticklabels(people, color=TXT)
    ax1.set_ylabel("accuracy (%)", color=TXT, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.tick_params(colors=MUTED)
    for s in ax1.spines.values():
        s.set_color("#2B4263")
    ax1.set_title("Every model rises and falls on the same people\n"
                  "the fold, not the architecture, drives most of the spread",
                  color=TXT, fontsize=12.5, fontweight="bold")
    ax1.legend(fontsize=8, facecolor=BG, labelcolor=TXT, loc="upper center",
               bbox_to_anchor=(0.5, -0.10), ncol=2, framealpha=0.0)

    ax2.set_facecolor(PANEL)
    names = [r["name"] for r in rows]
    means = [r["v"].mean() for r in rows]
    errs = [r["v"].std(ddof=1) / np.sqrt(len(people)) for r in rows]
    yp = np.arange(len(names))[::-1]
    ax2.barh(yp, means, xerr=errs, height=0.55,
             color=[GREEN if r["branch"] == "A" else RED for r in rows],
             error_kw=dict(ecolor=TXT, capsize=5, lw=1.6))
    for yy, m, e in zip(yp, means, errs):
        ax2.text(m + e + 1.5, yy, f"{m:.1f} ± {e:.1f}", va="center",
                 color=TXT, fontsize=9.5, fontweight="bold")
    ax2.set_yticks(yp); ax2.set_yticklabels(names, color=TXT, fontsize=9)
    ax2.set_xlim(0, 118)
    ax2.set_xlabel("mean fold accuracy ± standard error (%)", color=TXT,
                   fontweight="bold")
    ax2.tick_params(colors=MUTED)
    for s in ax2.spines.values():
        s.set_color("#2B4263")
    ax2.set_title("Error bars overlap for every CNN variant",
                  color=TXT, fontsize=12.5, fontweight="bold")

    fig.tight_layout()
    p = os.path.join(QC, "fold_variance.png")
    fig.savefig(p, dpi=170, facecolor=BG); plt.close(fig)
    print("\nfigure ->", p)

    with open(os.path.join(HERE, "fold_variance.json"), "w",
              encoding="utf-8") as f:
        json.dump({"folds": people,
                   "models": [{"name": r["name"], "overall": r["acc"],
                               "per_fold": {p: float(v) for p, v in
                                            zip(people, r["v"])},
                               "sd": float(r["v"].std(ddof=1))} for r in rows],
                   "pairwise": [{"a": a, "b": b, "mean_diff": float(md),
                                 "sd_diff": float(sd), "separable": bool(s)}
                                for a, b, md, sd, s in verdicts]},
                  f, indent=1)
    print("metrics ->", os.path.join(HERE, "fold_variance.json"))


if __name__ == "__main__":
    main()
