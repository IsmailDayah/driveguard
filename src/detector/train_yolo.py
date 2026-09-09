"""Train YOLOv8n and YOLOv11n on identical data for a fair head-to-head.

The project requires two detector variants so the project reports two
independent result sets. Everything except the architecture is held constant:
same dataset, same train/val split, same epochs, same image size, same seed.

    python train_yolo.py --data datasets/xxx/data.yaml --epochs 100
    python train_yolo.py --data ... --models yolov8n.pt yolo11n.pt yolov8s.pt
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, TXT, MUTED = "#0A1628", "#F1F5F9", "#93A6BD"
COLS = ["#22D3EE", "#A78BFA", "#F59E0B", "#22C55E"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo11n.pt"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=50,
                    help="stop if val mAP has not improved for N epochs. "
                         "ultralytics keeps best.pt by val mAP, so a longer "
                         "budget costs nothing in quality - patience only "
                         "stops it burning GPU time on a plateau.")
    ap.add_argument("--project", default="runs")
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    if dev == "cpu":
        print("WARNING: no CUDA. Detector training on CPU is very slow - "
              "run this on a CUDA machine.")

    results = []
    for name in a.models:
        tag = os.path.splitext(name)[0]
        print(f"\n{'='*70}\ntraining {tag}  ({a.epochs} epochs, imgsz {a.imgsz})\n{'='*70}")
        t0 = time.time()
        model = YOLO(name)
        model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz,
                    batch=a.batch, seed=a.seed, deterministic=True,
                    patience=a.patience,
                    project=a.project, name=tag, exist_ok=True, verbose=True)
        m = model.val(data=a.data, imgsz=a.imgsz, split="test", verbose=False)
        # inference speed on a single image, averaged
        speeds = m.speed  # ms per image: preprocess/inference/postprocess
        infer_ms = float(speeds.get("inference", 0.0))
        rec = {
            "model": tag,
            "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map),
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
            "inference_ms": infer_ms,
            "fps": (1000.0 / infer_ms) if infer_ms else None,
            "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6,
            "train_minutes": (time.time() - t0) / 60,
            "weights": os.path.join(a.project, tag, "weights", "best.pt"),
        }
        results.append(rec)
        print(f"\n{tag}: mAP@0.5 {100*rec['mAP50']:.1f}%  "
              f"mAP@0.5:0.95 {100*rec['mAP50_95']:.1f}%  "
              f"{rec['fps']:.0f} FPS  {rec['params_M']:.2f}M params")

    # ---- comparison table ----
    print(f"\n{'='*84}")
    print(f"{'model':<12}{'mAP@0.5':>10}{'mAP@.5:.95':>12}{'precision':>11}"
          f"{'recall':>9}{'FPS':>8}{'params(M)':>11}{'train(min)':>11}")
    print("-" * 84)
    for r in results:
        print(f"{r['model']:<12}{100*r['mAP50']:>9.1f}%{100*r['mAP50_95']:>11.1f}%"
              f"{100*r['precision']:>10.1f}%{100*r['recall']:>8.1f}%"
              f"{(r['fps'] or 0):>8.0f}{r['params_M']:>11.2f}{r['train_minutes']:>11.1f}")
    print("=" * 84)

    with open(os.path.join(a.project, "yolo_comparison.json"), "w") as f:
        json.dump(results, f, indent=1)

    # ---- comparison chart ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=BG)
    names = [r["model"] for r in results]
    panels = [("mAP@0.5", [100 * r["mAP50"] for r in results], "%"),
              ("mAP@0.5:0.95", [100 * r["mAP50_95"] for r in results], "%"),
              ("Inference speed", [r["fps"] or 0 for r in results], " FPS")]
    for ax, (title, vals, unit) in zip(axes, panels):
        ax.set_facecolor("#12263F")
        bars = ax.bar(names, vals, color=COLS[:len(names)], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}{unit}",
                    ha="center", va="bottom", color=TXT, fontweight="bold")
        ax.set_title(title, color=TXT, fontsize=12, fontweight="bold")
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_color("#2B4263")
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) else 1)
    fig.suptitle("Detector comparison on identical data and splits",
                 color=TXT, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(a.project, "yolo_comparison.png")
    fig.savefig(p, dpi=170, facecolor=BG)
    print("chart ->", p)


if __name__ == "__main__":
    main()
