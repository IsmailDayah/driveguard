"""Is YOLO26n genuinely worse at domain transfer, or just read at the wrong threshold?

The head-to-head on our own footage compared both detectors at ONE fixed
confidence (0.25). That is not obviously fair: YOLO26n is NMS-free and its
confidences are distributed differently - measured mean confidence 0.690 against
YOLO11n's 0.838, and it fired on 47.5% of frames against 60.3%. A model can look
much worse simply because the operating point suits the other one.

So this sweeps the threshold for BOTH models and compares each at its OWN best
operating point. If YOLO26n catches up, the gap was calibration. If it does not,
the transfer deficit is real.

Cost is one inference pass per model, not one per threshold: predictions are
collected once at a low floor and the metrics are recomputed offline.

    python threshold_sweep_ours.py
"""
import os
import sys
import glob
import json
import time
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_on_ours import OURS, FRAMES, QC, resolve_map, branch_frames, weights_tag

BG, TXT, MUTED = "#0A1628", "#F1F5F9", "#93A6BD"
COLS = ["#22D3EE", "#F59E0B"]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)


def collect(weights, files, classes, floor, imgsz):
    """One inference pass per model; keep every detection above `floor`.

    Metrics at any threshold >= floor can then be recomputed offline, so the
    sweep costs one pass rather than one pass per threshold.
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    cmap, _ = resolve_map(model.names, {}, classes)
    if not cmap:
        raise SystemExit(f"no class mapping for {weights}")
    per_frame, y_true, t0 = [], [], time.time()
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        res = model.predict(img, conf=floor, imgsz=imgsz, verbose=False)[0]
        dets = [(cmap[int(b.cls.item())], float(b.conf.item()))
                for b in res.boxes if int(b.cls.item()) in cmap]
        per_frame.append(dets)
        y_true.append(os.path.basename(os.path.dirname(f)))
    ms = 1000 * (time.time() - t0) / max(len(per_frame), 1)
    return per_frame, y_true, ms, sorted(set(cmap.values()))


def score_at(per_frame, y_true, thr, classes):
    """Prediction = highest-confidence detection at or above `thr`, else none."""
    y_pred = []
    for dets in per_frame:
        best, bc = "(none)", 0.0
        for c, cf in dets:
            if cf >= thr and cf > bc:
                best, bc = c, cf
        y_pred.append(best)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    det = float(np.mean([p != "(none)" for p in y_pred]))
    return {"thr": float(thr),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1.mean()),
            "detection_rate": det,
            "per_class_f1": {c: float(x) for c, x in zip(classes, f1)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+",
                    default=[os.path.join(HERE, "podweights", "yolo11n_best.pt"),
                             os.path.join(HERE, "podweights", "yolo26n_best.pt")])
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    os.makedirs(QC, exist_ok=True)

    # Exactly the frame set eval_on_ours uses, so these numbers sit in the same
    # table as the fixed-threshold ones and as Branch A / Branch B.
    classes = OURS[:3]
    files = sorted(glob.glob(os.path.join(FRAMES, "*", "*.jpg")))
    files = [f for f in files
             if os.path.basename(os.path.dirname(f)) in classes]
    keep = branch_frames()
    if keep is not None:
        files = [f for f in files if os.path.basename(f) in keep]
    print(f"frames: {len(files)}   classes: {classes}   floor: {a.floor}")

    out = {}
    for w in a.weights:
        tag = weights_tag(w)
        print(f"\n=== {tag} ===")
        per_frame, y_true, ms, mapped = collect(w, files, classes, a.floor, a.imgsz)
        rows = [score_at(per_frame, y_true, t, classes) for t in THRESHOLDS]
        best = max(rows, key=lambda r: r["macro_f1"])
        at25 = min(rows, key=lambda r: abs(r["thr"] - 0.25))
        out[tag] = {"rows": rows, "best": best, "at_0.25": at25,
                    "ms_per_frame_cpu": ms, "n_frames": len(y_true)}
        print(f"  {ms:.0f} ms/frame CPU")
        print(f"  at thr 0.25 : acc {100*at25['accuracy']:.1f}%  "
              f"macroF1 {100*at25['macro_f1']:.1f}%  "
              f"det {100*at25['detection_rate']:.1f}%")
        print(f"  BEST thr {best['thr']:.3f} : acc {100*best['accuracy']:.1f}%  "
              f"macroF1 {100*best['macro_f1']:.1f}%  "
              f"det {100*best['detection_rate']:.1f}%")
        print("  per-class F1 at best: " +
              "  ".join(f"{c} {100*v:.1f}%" for c, v in best["per_class_f1"].items()))

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("WAS THE FIXED-THRESHOLD COMPARISON UNFAIR?")
    print("=" * 78)
    tags = list(out)
    if len(tags) == 2:
        a_, b_ = out[tags[0]], out[tags[1]]
        d25 = 100 * (a_["at_0.25"]["macro_f1"] - b_["at_0.25"]["macro_f1"])
        dbest = 100 * (a_["best"]["macro_f1"] - b_["best"]["macro_f1"])
        print(f"  {tags[0]} - {tags[1]}")
        print(f"    at a common 0.25      : {d25:+.1f} macro-F1 points")
        print(f"    each at its own best  : {dbest:+.1f} macro-F1 points")
        shrink = abs(d25) - abs(dbest)
        print(f"\n  gap narrows by {shrink:+.1f} points once each model is read at")
        print("  its own operating point.")
        if abs(dbest) < 3.0:
            print("  -> The fixed-threshold comparison WAS misleading; the two")
            print("     detectors are close once calibrated.")
        else:
            print("  -> The difference SURVIVES recalibration, so it is a real")
            print("     transfer deficit rather than an operating-point artefact.")

    # -------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=BG)
    for ax, key, lab in ((axes[0], "macro_f1", "macro-F1"),
                         (axes[1], "detection_rate", "detection rate")):
        ax.set_facecolor("#12263F")
        for i, tag in enumerate(out):
            r = out[tag]["rows"]
            ax.plot([x["thr"] for x in r], [100 * x[key] for x in r],
                    color=COLS[i % 2], lw=2.4, label=tag)
            if key == "macro_f1":
                b = out[tag]["best"]
                ax.plot([b["thr"]], [100 * b["macro_f1"]], "o",
                        color=COLS[i % 2], ms=10, mec="white", mew=1.4)
        ax.axvline(0.25, color=MUTED, ls=":", lw=1.2)
        ax.text(0.255, ax.get_ylim()[1] * 0.05, "fixed 0.25 used\nin the first test",
                color=MUTED, fontsize=8)
        ax.set_xlabel("confidence threshold", color=TXT, fontweight="bold")
        ax.set_ylabel(f"{lab} (%)", color=TXT, fontweight="bold")
        ax.tick_params(colors=MUTED)
        ax.grid(alpha=0.14, color="#93A6BD")
        for s in ax.spines.values():
            s.set_color("#2B4263")
        leg = ax.legend(facecolor="#0F1E33", edgecolor="#2B4263")
        for t in leg.get_texts():
            t.set_color(TXT)
    fig.suptitle("Detector threshold sweep on OUR footage - each model at its own "
                 "operating point\n(circles mark best macro-F1)",
                 color=TXT, fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = os.path.join(QC, "detector_threshold_sweep_ours.png")
    fig.savefig(p, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"\nfigure -> {p}")

    jp = os.path.join(HERE, "threshold_sweep_ours.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"metrics -> {jp}")


if __name__ == "__main__":
    main()
