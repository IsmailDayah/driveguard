"""DriveGuard AI — the ML-vs-DL comparison at the heart of this project.

Collects every completed Branch A and Branch B result and renders one figure
plus one table. All numbers come from the SAME leave-one-person-out protocol on
the SAME frames, so the comparison is like-for-like.

    python compare_branches.py
"""
import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
QC = os.path.join(HERE, "qc")
os.makedirs(QC, exist_ok=True)
BG, PANEL, TXT, MUTED = "#0A1628", "#12263F", "#F1F5F9", "#93A6BD"
GREEN, RED, CYAN, AMBER = "#22C55E", "#EF4444", "#22D3EE", "#F59E0B"


def load():
    rows = []
    ab = os.path.join(HERE, "branch_a_ablation.json")
    if os.path.exists(ab):
        for r in json.load(open(ab, encoding="utf-8")):
            if r["variant"].startswith("B "):        # 3-class raw = chosen config
                rows.append({"branch": "A", "name": "SVM (EAR/MAR)",
                             "accuracy": r["accuracy"], "f1": r["f1"],
                             "per_person": r["per_person"]})
    for p in sorted(glob.glob(os.path.join(HERE, "branch_b_*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        if len(r.get("classes", [])) != 3:
            continue
        rows.append({"branch": "B", "name": nice_name(r),
                     "accuracy": r["accuracy"], "f1": r["f1"],
                     "per_person": r["per_person"]})
    return rows


def is_square(r):
    """Runs predating the explicit flag are identified from their tag.

    Tags look like `<backbone>-<input>[sq][-frzN]-<n>cls`, so the marker lives
    on the input field: 'eyesq' vs 'eye'.
    """
    if "square" in r:
        return bool(r["square"])
    parts = r.get("tag", "").split("-")
    return len(parts) > 1 and parts[1].endswith("sq")


def nice_name(r):
    """Readable label built from the run's own fields.

    Derived rather than looked up, so a new ablation variant appears in the
    chart the moment it finishes instead of showing up as a raw tag.
    """
    arch = ("MobileNetV2" if r.get("backbone") == "mobilenet_v2"
            else "from scratch")
    bits = [r.get("input", "?")]
    if is_square(r):
        bits.append("squared")
    if r.get("freeze"):
        bits.append(f"frozen-{r['freeze']}")
    return f"CNN {arch} ({', '.join(bits)})"


def main():
    rows = load()
    if not rows:
        print("no results yet"); return
    rows.sort(key=lambda r: -r["accuracy"])

    print(f"{'branch':<8}{'model':<34}{'accuracy':>10}{'macro-F1':>10}"
          f"{'per-person range':>20}")
    print("-" * 84)
    for r in rows:
        pp = r["per_person"].values()
        print(f"{r['branch']:<8}{r['name']:<34}{100*r['accuracy']:>9.1f}%"
              f"{100*r['f1']:>9.1f}%"
              f"{f'{100*min(pp):.0f}-{100*max(pp):.0f}%':>20}")

    names = [r["name"] for r in rows]
    accs = [100 * r["accuracy"] for r in rows]
    f1s = [100 * r["f1"] for r in rows]
    cols = [GREEN if r["branch"] == "A" else RED for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=BG,
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.set_facecolor(PANEL)
    ypos = np.arange(len(names))[::-1]
    bars = ax1.barh(ypos, accs, color=cols, height=0.6)
    for b, a_, f in zip(bars, accs, f1s):
        ax1.text(a_ + 1, b.get_y() + b.get_height() / 2,
                 f"{a_:.1f}%   (F1 {f:.1f})", va="center",
                 color=TXT, fontsize=10, fontweight="bold")
    ax1.set_yticks(ypos); ax1.set_yticklabels(names, color=TXT, fontsize=10)
    ax1.set_xlim(0, 108); ax1.set_xlabel("leave-one-person-out accuracy (%)",
                                         color=TXT, fontweight="bold")
    ax1.tick_params(colors=MUTED)
    for s in ax1.spines.values():
        s.set_color("#2B4263")
    ax1.set_title("Traditional ML  vs  Deep Learning\n"
                  "identical frames, identical validation protocol",
                  color=TXT, fontsize=12.5, fontweight="bold")
    # below the axes: inside the panel it lands on the shortest bar's label
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREEN),
                        plt.Rectangle((0, 0), 1, 1, color=RED)],
               labels=["Branch A - Traditional ML", "Branch B - Deep Learning"],
               loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
               facecolor=BG, labelcolor=TXT, fontsize=9, framealpha=0.0)

    ax2.set_facecolor(PANEL)
    people = sorted(rows[0]["per_person"])
    x = np.arange(len(people))
    w = 0.8 / len(rows)
    for i, r in enumerate(rows):
        vals = [100 * r["per_person"].get(p, 0) for p in people]
        ax2.bar(x + i * w - 0.4 + w / 2, vals, width=w * 0.92,
                label=r["name"],
                color=GREEN if r["branch"] == "A" else RED,
                alpha=1.0 - 0.18 * i)
    ax2.set_xticks(x); ax2.set_xticklabels(people, color=TXT)
    ax2.set_ylim(0, 105); ax2.set_ylabel("accuracy (%)", color=TXT)
    ax2.tick_params(colors=MUTED)
    for s in ax2.spines.values():
        s.set_color("#2B4263")
    ax2.set_title("Per-person (model never saw that face in training)",
                  color=TXT, fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, facecolor=BG, labelcolor=TXT, loc="upper center",
               bbox_to_anchor=(0.5, -0.09), ncol=2, framealpha=0.0)

    fig.tight_layout()
    p = os.path.join(QC, "branch_comparison.png")
    fig.savefig(p, dpi=170, facecolor=BG); plt.close(fig)
    print("\ncomparison figure ->", p)


if __name__ == "__main__":
    main()
