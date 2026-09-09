"""DriveGuard AI — the measured case for a two-branch design.

A landmark branch and a detector branch are both needed. That is easy to assert
and easy to challenge, so it is measured here on our own footage, one class at a
time.

Two independent failures of the landmark pipeline are quantified:

  1. PHONE — Branch A reads eye and mouth geometry. A phone held beside the
     face leaves both unchanged, so the class is invisible to it *by
     construction*, not by undertraining. An off-the-shelf COCO detector, with
     no training on our data at all, sees the object directly.

  2. PEAK YAWN — MediaPipe returns no landmarks on the frames where the head is
     pitched furthest back, which is the deepest part of the yawn: exactly the
     event the branch exists to catch. Those frames are absent from
     features.npz entirely, so Branch A is never even scored on them.

Both numbers come from the same leave-one-person-out protocol on the same
frames as every other result in the project.

    python phone_case_study.py
"""
import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from experiment_branch_a import loso, FULL

HERE = os.path.dirname(os.path.abspath(__file__))
QC = os.path.join(HERE, "qc")
os.makedirs(QC, exist_ok=True)
FRAMES = os.path.join(HERE, "..", "..", "data", "frames")
BG, PANEL, TXT, MUTED = "#0A1628", "#12263F", "#F1F5F9", "#93A6BD"
GREEN, RED, CYAN, AMBER = "#22C55E", "#EF4444", "#22D3EE", "#F59E0B"


def branch_a_per_class():
    """Per-class precision/recall for Branch A in its 4-class configuration."""
    d = np.load(os.path.join(HERE, "features.npz"), allow_pickle=True)
    yt, yp, _ = loso(d["X_geo"], d["y"], d["persons"], FULL)
    pr, rc, f1, sup = precision_recall_fscore_support(
        yt, yp, labels=FULL, zero_division=0)
    return {c: {"precision": float(pr[i]), "recall": float(rc[i]),
                "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(FULL)}


def yolo_phone():
    """Phone metrics from the most recent full eval_on_ours run."""
    hits = sorted(glob.glob(os.path.join(HERE, "yolo", "ownfootage_*.json")))
    if not hits:
        print("no eval_on_ours results yet - run yolo/eval_on_ours.py first")
        return None
    r = json.load(open(hits[-1], encoding="utf-8"))
    name = os.path.basename(hits[-1])
    if r.get("frames", 0) < 500:
        print(f"WARNING: {name} covers only {r['frames']} frames - that is a "
              f"--limit run, not the full evaluation. Ignoring it; re-run "
              f"eval_on_ours.py without --limit.")
        return None
    if "per_class" not in r or "phone" not in r.get("per_class", {}):
        print(f"WARNING: {name} predates per-class metrics (or was run with "
              f"--classes 3). Re-run eval_on_ours.py to include the phone row.")
        return None
    ph = r["per_class"]["phone"]
    return {"precision": ph["precision"], "recall": ph["recall"],
            "f1": ph["f1"], "support": ph["support"],
            "frames": r["frames"], "tag": r.get("tag", "yolo")}


def landmark_dropouts():
    """Frames MediaPipe could not landmark, grouped by class."""
    used = {os.path.basename(str(x)) for x in
            np.load(os.path.join(HERE, "features.npz"),
                    allow_pickle=True)["files"]}
    out = {}
    for p in glob.glob(os.path.join(FRAMES, "*", "*.jpg")):
        c = os.path.basename(os.path.dirname(p))
        b = os.path.basename(p)
        out.setdefault(c, {"total": 0, "dropped": []})
        out[c]["total"] += 1
        if b not in used:
            out[c]["dropped"].append(b)
    return out


def main():
    print("computing Branch A per-class metrics (4-class LOSO)...")
    a = branch_a_per_class()
    y = yolo_phone()
    drops = landmark_dropouts()

    print(f"\n{'='*74}")
    print("1. PHONE - can each branch see it?")
    print(f"{'='*74}")
    print(f"{'':<38}{'precision':>11}{'recall':>9}{'F1':>8}")
    print(f"{'Branch A (EAR/MAR geometry)':<38}"
          f"{100*a['phone']['precision']:>10.1f}%{100*a['phone']['recall']:>8.1f}%"
          f"{100*a['phone']['f1']:>7.1f}%")
    if y:
        print(f"{y['tag'] + ' COCO, zero-shot':<38}"
              f"{100*y['precision']:>10.1f}%"
              f"{100*y['recall']:>8.1f}%{100*y['f1']:>7.1f}%")
    print(f"\nBranch A's other classes, same run, for scale:")
    for c in ("normal", "drowsy", "yawning"):
        print(f"  {c:<36}{100*a[c]['precision']:>10.1f}%"
              f"{100*a[c]['recall']:>8.1f}%{100*a[c]['f1']:>7.1f}%")

    print(f"\n{'='*74}")
    print("2. PEAK YAWN - where MediaPipe returns nothing at all")
    print(f"{'='*74}")
    for c in sorted(drops):
        n = len(drops[c]["dropped"])
        print(f"  {c:<10}{n:>4} / {drops[c]['total']:<5} frames unusable "
              f"({100*n/drops[c]['total']:>4.1f}%)")
        for b in drops[c]["dropped"][:6]:
            print(f"        {b}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), facecolor=BG,
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # -- panel 1: per-class F1 for Branch A, with the detector beside it --
    # Grouped, never overlaid: drawing the detector bar on top of Branch A's
    # would read as a stacked bar summing to the detector's score.
    ax1.set_facecolor(PANEL)
    cls = ["normal", "drowsy", "yawning", "phone"]
    xs = np.arange(len(cls))
    vals = [100 * a[c]["f1"] for c in cls]

    def label(x, v):
        ax1.text(x, v + 1.5, f"{v:.1f}%", ha="center", color=TXT,
                 fontweight="bold", fontsize=10.5)

    # only the phone class has a detector counterpart, so only that pair is
    # split; the other three stay centred on their tick
    ax1.bar(xs[:3], vals[:3], color=CYAN, width=0.6,
            label="Branch A — EAR/MAR geometry")
    for x, v in zip(xs[:3], vals[:3]):
        label(x, v)

    w, off = (0.38, 0.19) if y else (0.6, 0.0)
    ax1.bar([xs[-1] - off], [vals[-1]], color=RED, width=w)
    label(xs[-1] - off, vals[-1])
    if y:
        ax1.bar([xs[-1] + off], [100 * y["f1"]], color=GREEN, width=w,
                label=f"{y['tag']} — zero-shot detector")
        label(xs[-1] + off, 100 * y["f1"])
        ax1.annotate("", xy=(xs[-1] + off, 100 * y["f1"] - 3),
                     xytext=(xs[-1] - off, vals[-1] + 3),
                     arrowprops=dict(arrowstyle="->", color=AMBER, lw=2))
        # beside the arrow, not on it
        ax1.text(xs[-1] - off - 0.12, (vals[-1] + 100 * y["f1"]) / 2 + 6,
                 f"+{100*y['f1'] - vals[-1]:.1f}", color=AMBER, fontsize=11.5,
                 fontweight="bold", ha="right", va="center")
        ax1.legend(facecolor=BG, labelcolor=TXT, fontsize=9, loc="upper left")
    ax1.set_xticks(xs); ax1.set_xticklabels(cls)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("per-class F1 (%)", color=TXT, fontweight="bold")
    ax1.tick_params(colors=MUTED)
    for s in ax1.spines.values():
        s.set_color("#2B4263")
    ax1.set_title("Branch A sees three classes well and one not at all\n"
                  "the phone gap is closed by the detector, not by tuning",
                  color=TXT, fontsize=12.5, fontweight="bold")

    # -- panel 2: landmark dropout by class --
    ax2.set_facecolor(PANEL)
    cls2 = sorted(drops)
    pct = [100 * len(drops[c]["dropped"]) / drops[c]["total"] for c in cls2]
    b2 = ax2.bar(cls2, pct, color=[AMBER if p > 0 else "#2B4263" for p in pct],
                 width=0.6)
    for b, c in zip(b2, cls2):
        n = len(drops[c]["dropped"])
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                 f"{n}/{drops[c]['total']}", ha="center", color=TXT,
                 fontweight="bold", fontsize=10.5)
    ax2.set_ylim(0, max(max(pct) * 1.45, 0.5))
    ax2.set_ylabel("frames with no landmarks (%)", color=TXT, fontweight="bold")
    ax2.tick_params(colors=MUTED)
    for s in ax2.spines.values():
        s.set_color("#2B4263")
    ax2.set_title("MediaPipe drops out only on yawning frames\n"
                  "the head-back pose at the peak of the yawn",
                  color=TXT, fontsize=12.5, fontweight="bold")

    fig.tight_layout()
    p = os.path.join(QC, "two_branch_justification.png")
    fig.savefig(p, dpi=170, facecolor=BG); plt.close(fig)
    print("\nfigure ->", p)

    res = {"branch_a_per_class": a,
           "yolo_zero_shot_phone": y,
           "landmark_dropouts": {c: {"total": v["total"],
                                     "dropped": v["dropped"]}
                                 for c, v in drops.items()}}
    o = os.path.join(HERE, "two_branch_justification.json")
    with open(o, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("metrics ->", o)


if __name__ == "__main__":
    main()
