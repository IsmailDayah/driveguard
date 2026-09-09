"""DriveGuard AI — Branch B CNN (the deep-learning branch).

Trains a CNN on the cached crops and evaluates it with the SAME
leave-one-person-out protocol as Branch A, so the ML-vs-DL comparison the
project makes is genuinely like-for-like.

Configurations
    backbone : mobilenet_v2 (ImageNet transfer learning) | scratch
    input    : face (96x96 context) | eye (96x32, identical region to Branch A)
    classes  : 3 (normal/drowsy/yawning) | 4 (adds phone)

    python train_branch_b_cnn.py --probe                 # time one epoch
    python train_branch_b_cnn.py --input face --epochs 40
    python train_branch_b_cnn.py --input eye  --classes 3
"""
import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
QC = os.path.join(HERE, "qc")
os.makedirs(QC, exist_ok=True)
BG, TXT, MUTED = "#0A1628", "#F1F5F9", "#93A6BD"
FULL = ["normal", "drowsy", "yawning", "phone"]
THREE = ["normal", "drowsy", "yawning"]

torch.manual_seed(0)
np.random.seed(0)


class CropDS(Dataset):
    """Grayscale crop -> 3-channel tensor, with training augmentation.

    `square` upsamples a wide crop (the 96x32 eye strip) to 96x96. MobileNetV2
    downsamples by 32x, so a 32px-tall input collapses to a 1px-tall feature
    map and every vertical detail is destroyed - which is fatal here, because
    eye openness IS a vertical measurement. Upsampling adds no information but
    preserves vertical structure through the conv stack. The distortion is
    identical for every class, so it cannot leak the label.
    """

    def __init__(self, imgs, labels, classes, train, square=False):
        if square and imgs.shape[1] != imgs.shape[2]:
            side = max(imgs.shape[1], imgs.shape[2])
            imgs = np.stack([cv2.resize(im, (side, side)) for im in imgs])
        self.imgs = imgs
        self.y = np.array([classes.index(c) for c in labels])
        self.train = train
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=7, translate=(0.06, 0.06),
                                    scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.35, contrast=0.35),
        ])
        self.erase = transforms.RandomErasing(p=0.25, scale=(0.02, 0.12))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        t = torch.from_numpy(self.imgs[i].astype(np.float32) / 255.0)[None]
        if self.train:
            t = self.aug(t)
        t = t.repeat(3, 1, 1)
        t = transforms.functional.normalize(
            t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.train:
            t = self.erase(t)
        return t, int(self.y[i])


class SmallCNN(nn.Module):
    """From-scratch baseline sized for ~1k training images."""

    def __init__(self, n):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128),
                               nn.AdaptiveAvgPool2d(1))
        self.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, n))

    def forward(self, x):
        return self.c(self.f(x))


def load_backbone(model, path):
    """Load a pretrained MobileNetV2 feature stack, ignoring its classifier.

    The pretraining task (DDD: drowsy / not-drowsy, 2 classes) has a different
    head width to ours, so only `features.*` is transferred. Reported rather
    than silently partial-loading, because a typo in the checkpoint path would
    otherwise look exactly like a successful run with ImageNet weights.
    """
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd)
    feat = {k: v for k, v in sd.items() if k.startswith("features.")}
    if not feat:
        raise SystemExit(f"{path} contains no 'features.*' tensors "
                         f"(keys start with: {sorted(sd)[:3]})")
    missing, unexpected = model.load_state_dict(feat, strict=False)
    loaded = len(feat)
    shape_ok = sum(1 for k, v in feat.items()
                   if k in dict(model.named_parameters())
                   or k in dict(model.named_buffers()))
    print(f"  init: loaded {loaded} backbone tensors from "
          f"{os.path.basename(path)} ({shape_ok} matched by name), "
          f"{len([m for m in missing if m.startswith('features.')])} "
          f"backbone tensors left at ImageNet init")
    if unexpected:
        print(f"  init: {len(unexpected)} unexpected keys ignored")
    return model


def build(backbone, n, freeze=0, init=""):
    """freeze: 0 = fine-tune everything, k>0 = freeze all but the last k blocks.

    Fine-tuning all 2.2M MobileNetV2 parameters on ~595 images per fold is
    heavily overparameterised. Freezing most of the ImageNet backbone and
    adapting only the final blocks is the standard remedy for small datasets,
    so it is tested rather than assumed to be unnecessary.
    """
    if backbone == "scratch":
        return SmallCNN(n)
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if init:
        m = load_backbone(m, init)
    m.classifier = nn.Sequential(nn.Dropout(0.3),
                                 nn.Linear(m.last_channel, n))
    if freeze > 0:
        for p in m.features.parameters():
            p.requires_grad = False
        for blk in m.features[-freeze:]:
            for p in blk.parameters():
                p.requires_grad = True
    return m


def run_fold(Xtr, ytr, Xte, yte, classes, backbone, epochs, bs, lr, workers,
             square=False, freeze=0, init="", dev="cpu"):
    tr = DataLoader(CropDS(Xtr, ytr, classes, True, square), batch_size=bs,
                    shuffle=True, num_workers=workers, drop_last=len(ytr) > bs)
    te = DataLoader(CropDS(Xte, yte, classes, False, square), batch_size=64,
                    shuffle=False, num_workers=workers)
    model = build(backbone, len(classes), freeze, init).to(dev)
    # class weights so the slightly larger 'normal' class cannot dominate
    cnt = np.array([np.sum(np.array(ytr) == c) for c in classes], np.float32)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) / len(classes),
                     dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(tr)), pct_start=0.3)

    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in te:
            preds.append(model(xb.to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def plot_cm(cm, classes, title, sub, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="cividis", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{100*cmn[i, j]:.0f}%", ha="center",
                    va="center", fontsize=11, fontweight="bold",
                    color="white" if cmn[i, j] < 0.55 else BG)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color=TXT)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=TXT)
    ax.set_xlabel("Predicted", color=TXT, fontweight="bold")
    ax.set_ylabel("True", color=TXT, fontweight="bold")
    ax.set_title(f"{title}\n{sub}", color=TXT, fontsize=12,
                 fontweight="bold", pad=12)
    for s in ax.spines.values():
        s.set_color("#2B4263")
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(colors=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=170, facecolor=BG); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="face", choices=["face", "eye"])
    ap.add_argument("--backbone", default="mobilenet_v2",
                    choices=["mobilenet_v2", "scratch"])
    ap.add_argument("--classes", type=int, default=3, choices=[3, 4])
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze all but the last N backbone blocks "
                         "(0 = fine-tune everything)")
    ap.add_argument("--crops", default="crops.npz",
                    help="cached crop file; crops_224.npz holds the "
                         "higher-resolution face crops")
    ap.add_argument("--init", default="",
                    help="pretrained MobileNetV2 backbone .pt; only "
                         "features.* are loaded")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--square", action="store_true",
                    help="upsample a wide crop to square before the backbone")
    ap.add_argument("--probe", action="store_true",
                    help="time a single 1-epoch fold and exit")
    a = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda is unavailable")
    crops = a.crops if os.path.isabs(a.crops) else os.path.join(HERE, a.crops)
    if not os.path.exists(crops):
        raise SystemExit(f"no such crop file: {crops}")
    d = np.load(crops, allow_pickle=True)
    X = d["eyes"] if a.input == "eye" else d["faces"]
    y, persons = d["y"], d["persons"]
    classes = THREE if a.classes == 3 else FULL
    keep = np.isin(y, classes)
    X, y, persons = X[keep], y[keep], persons[keep]

    print(f"input={a.input} {X.shape[1:]}  backbone={a.backbone}  "
          f"classes={len(classes)}  n={len(y)}  epochs={a.epochs}  "
          f"device={dev}  crops={os.path.basename(crops)}"
          + (f"  init={os.path.basename(a.init)}" if a.init else ""))

    if a.probe:
        p = sorted(set(persons.tolist()))[0]
        te = persons == p
        t0 = time.time()
        run_fold(X[~te], y[~te], X[te], y[te], classes, a.backbone,
                 1, a.batch, a.lr, a.workers, a.square, a.freeze, a.init, dev)
        dt = time.time() - t0
        print(f"\n1 epoch on 1 fold: {dt:.1f}s")
        print(f"estimated full run ({a.epochs} epochs x 4 folds): "
              f"{dt * a.epochs * 4 / 60:.1f} min")
        return

    t0 = time.time()
    yt, yp, per_p = [], [], {}
    for p in sorted(set(persons.tolist())):
        te = persons == p
        pred_idx = run_fold(X[~te], y[~te], X[te], y[te], classes,
                            a.backbone, a.epochs, a.batch, a.lr, a.workers,
                            a.square, a.freeze, a.init, dev)
        pred = np.array([classes[i] for i in pred_idx])
        per_p[p] = accuracy_score(y[te], pred)
        # flushed so a redirected log shows progress during the run, not only
        # once the process exits
        print(f"  fold {p:<10}{100*per_p[p]:>7.1f}%   "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
        yt.append(y[te]); yp.append(pred)
    yt, yp = np.concatenate(yt), np.concatenate(yp)

    acc = accuracy_score(yt, yp)
    pr, rc, f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0)
    # resolution and init belong in the tag: without them a 224 run would
    # overwrite the 96 run's JSON and confusion matrix
    px = X.shape[1] if a.input == "face" else X.shape[2]
    suffix = ("sq" if a.square else "") + (f"-frz{a.freeze}" if a.freeze else "")
    if px != 96:
        suffix += f"-{px}px"
    if a.init:
        suffix += "-" + os.path.splitext(os.path.basename(a.init))[0]
    tag = f"{a.backbone}-{a.input}{suffix}-{len(classes)}cls"
    print(f"\n{'='*66}")
    print(f"Branch B CNN [{tag}]  acc {100*acc:.1f}%  prec {100*pr:.1f}%  "
          f"rec {100*rc:.1f}%  F1 {100*f1:.1f}%")
    print(f"{'='*66}")
    print(classification_report(yt, yp, labels=classes,
                                zero_division=0, digits=3))

    cm = confusion_matrix(yt, yp, labels=classes)
    path = os.path.join(QC, f"branch_b_{tag}_confusion.png")
    plot_cm(cm, classes, f"Branch B CNN — {a.backbone} on {a.input} crop",
            f"leave-one-person-out accuracy {100*acc:.1f}%", path)
    print("confusion matrix ->", path)

    res = {"tag": tag, "input": a.input, "backbone": a.backbone,
           "freeze": a.freeze, "square": bool(a.square),
           "px": int(px), "init": a.init, "device": dev,
           "classes": classes, "epochs": a.epochs, "accuracy": acc,
           "precision": pr, "recall": rc, "f1": f1,
           "per_person": {k: float(v) for k, v in per_p.items()},
           "minutes": (time.time() - t0) / 60}
    out = os.path.join(HERE, f"branch_b_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print("metrics ->", out)


if __name__ == "__main__":
    main()

