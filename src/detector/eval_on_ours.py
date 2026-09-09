"""Test a trained detector on OUR OWN clips — the domain-gap experiment.

The detectors are trained on the FL3D + NITYMED detection set built
by build_detection_set.py. This script runs them on the
1,043 frames in `data/frames/`, i.e. **the exact frames Branch A and the
Branch B CNN were scored on**, so all three approaches are compared on identical
data with identical people held out.

Runs locally on CPU by design. `data/raw` and `data/frames` contain
identifiable faces; the PDPA 2010 commitment made to the participants is that
this footage is processed on-device and never uploaded, so the domain-gap test
comes to the data rather than the data going to a remote machine.

Our clips carry a class label per clip, not bounding boxes, so the detector is
scored as a classifier: each frame's prediction is its highest-confidence
detection. That splits the domain gap into two separate, reportable failures —
frames where the detector found nothing at all, and frames where it found
something and named it wrong.

    python eval_on_ours.py --weights runs/yolov8n/weights/best.pt
    python eval_on_ours.py --weights ... --map "awake=normal,yawn=yawning"
    python eval_on_ours.py --weights ... --classes 3      # drop phone
"""
import argparse
import glob
import json
import os
import time
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.abspath(os.path.join(HERE, "..", "..", "data", "frames"))
QC = os.path.join(HERE, "qc")
os.makedirs(QC, exist_ok=True)
BG, PANEL, TXT, MUTED = "#0A1628", "#12263F", "#F1F5F9", "#93A6BD"
OURS = ["normal", "drowsy", "yawning", "phone"]

# Public drowsiness sets use many spellings for the same four states. These are
# only *suggestions*: the resolved mapping is printed for eyeballing before the
# run, and anything unmatched is reported rather than quietly dropped.
SYNONYMS = {
    "normal":  ["normal", "awake", "alert", "active", "open", "eyes_open",
                "no_yawn", "nodrowsy", "non_drowsy", "safe_driving", "attentive"],
    "drowsy":  ["drowsy", "sleepy", "sleep", "closed", "eyes_closed",
                "microsleep", "fatigue", "tired", "drowsiness"],
    "yawning": ["yawning", "yawn", "mouth_open", "yawning_mouth"],
    "phone":   ["phone", "cell", "cellphone", "mobile", "texting", "talking",
                "phone_use", "using_phone", "distracted"],
}


NEG_WORDS = {"no", "not", "non", "without", "never", "un"}
NEG_PREFIX = ("non", "not", "no", "un")


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _syn_table():
    return {norm(a): ours for ours, alts in SYNONYMS.items() for a in alts}


def _loose(n, syn):
    """Substring match against the synonym table."""
    for k, v in syn.items():
        if k in n or n in k:
            return v
    return None


def _negated_state(raw, syn):
    """If `raw` is a NEGATED state name, return the state it negates.

    Datasets label the safe class as often as the unsafe one, and they do it
    with names like 'no_yawn', 'not drowsy', 'no phone'. Naive substring
    matching maps every one of those to the exact class they deny, which
    silently inverts the label for a whole class. Only accept a negation when
    the text after the negation marker really is a state, so 'normal' (which
    merely starts with 'no') is left alone.
    """
    toks = [t for t in "".join(c if c.isalnum() else " "
                              for c in raw.lower()).split() if t]
    if any(t in NEG_WORDS for t in toks):
        rest = norm("".join(t for t in toks if t not in NEG_WORDS))
        return syn.get(rest) or _loose(rest, syn) if rest else None
    n = norm(raw)                                   # glued: nondrowsy, noyawn
    for p in NEG_PREFIX:
        if n.startswith(p) and len(n) > len(p) + 2:
            rest = n[len(p):]
            hit = syn.get(rest) or _loose(rest, syn)
            if hit:
                return hit
    return None


def resolve_map(det_names, manual, classes):
    """detector class id -> one of our labels, or None if it has no counterpart."""
    manual_n = {norm(k): v for k, v in manual.items()}
    syn = _syn_table()

    out, unmapped = {}, []
    for i, raw in det_names.items():
        n = norm(raw)
        hit = manual_n.get(n) or syn.get(n)          # explicit override, then exact
        if hit is None:
            neg = _negated_state(raw, syn)           # 'not drowsy' -> normal
            if neg is not None:
                # negating an unsafe state means the safe state; negating
                # 'normal' is ambiguous (which unsafe state?) so it is dropped
                hit = "normal" if neg != "normal" else None
            else:
                hit = _loose(n, syn)
        if hit in classes:
            out[i] = hit
        else:
            unmapped.append((raw, hit))
    return out, unmapped


def weights_tag(w):
    """runs/yolov8n/weights/best.pt -> 'yolov8n';  yolov8n.pt -> 'yolov8n'."""
    parts = os.path.normpath(os.path.abspath(w)).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "weights":
        return parts[-3]
    return os.path.splitext(parts[-1])[0]


def branch_frames():
    """Basenames of the frames Branch A / Branch B actually scored on.

    MediaPipe fails to landmark a handful of frames, and both landmark-based
    branches silently drop those, so `data/frames` is a strict superset
    of what they were measured on. Scoring the detector on the superset would
    hand it four frames the others never had to answer, which is not a
    like-for-like comparison.
    """
    p = os.path.join(HERE, "..", "features.npz")
    if not os.path.exists(p):
        return None
    try:
        return {os.path.basename(str(x))
                for x in np.load(p, allow_pickle=True)["files"]}
    except Exception:
        return None


def stratified(files, limit, classes):
    """Take `limit` frames spread evenly over classes, not the first N.

    Sorted order puts one person's one class first, so a plain head() smoke
    test can report 0% on data where nothing was detectable anyway.
    """
    if not limit or limit >= len(files):
        return files
    per = max(1, limit // len(classes))
    out = []
    for c in classes:
        sub = [f for f in files if os.path.basename(os.path.dirname(f)) == c]
        if sub:
            idx = np.linspace(0, len(sub) - 1, min(per, len(sub))).astype(int)
            out += [sub[i] for i in idx]
    return sorted(out)


def plot_cm(cm, classes, title, sub, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=BG)
    ax.set_facecolor(BG)
    cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="cividis", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{100*cmn[i, j]:.0f}%", ha="center",
                    va="center", fontsize=10, fontweight="bold",
                    color="white" if cmn[i, j] < 0.55 else BG)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_xticklabels(classes + (["(none)"] if cm.shape[1] > cm.shape[0] else []),
                       color=TXT, rotation=20, ha="right")
    ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(classes, color=TXT)
    ax.set_xlabel("Predicted", color=TXT, fontweight="bold")
    ax.set_ylabel("True", color=TXT, fontweight="bold")
    ax.set_title(f"{title}\n{sub}", color=TXT, fontsize=12, fontweight="bold",
                 pad=12)
    for s in ax.spines.values():
        s.set_color("#2B4263")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(colors=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=170, facecolor=BG); plt.close(fig)


def contact_sheet(rows, path, title):
    """Annotated examples so the result is verified by eye, not only by number.

    The detection box is drawn on, because a grid of unannotated faces shows
    nothing about *what the detector actually found* — which is the only thing
    worth checking by eye here.
    """
    if not rows:
        return
    n = len(rows)
    cols = min(4, n)
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols, figsize=(3.5 * cols, 3.0 * r),
                             facecolor=BG, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off"); ax.set_facecolor(BG)
    for ax, (img, true_c, pred_c, conf, xyxy) in zip(axes.ravel(), rows):
        ok = true_c == pred_c
        col = "#22C55E" if ok else "#EF4444"
        if xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            bgr = tuple(int(col.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(img, f"{pred_c} {conf:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"true {true_c} / pred {pred_c}"
                     + (f"  {conf:.2f}" if conf else ""),
                     color=col, fontsize=9.5, fontweight="bold")
    fig.suptitle(title, color=TXT, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    # tight_layout does not reserve room for per-axes titles on a multi-row
    # grid, so row 2's labels land on top of row 1's images without this
    if r > 1:
        fig.subplots_adjust(hspace=0.30)
    fig.savefig(path, dpi=140, facecolor=BG); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--classes", type=int, default=4, choices=[3, 4],
                    help="3 drops 'phone' to match the Branch A/B headline runs")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--map", default="", help='e.g. "awake=normal,yawn=yawning"')
    ap.add_argument("--save-vis", type=int, default=8,
                    help="number of annotated example frames to render")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: N frames spread evenly over the classes")
    ap.add_argument("--all-frames", dest="match_branches", action="store_false",
                    help="score every frame on disk, including the ones "
                         "MediaPipe could not landmark (default: restrict to "
                         "the frames Branch A/B were scored on)")
    a = ap.parse_args()

    classes = OURS[:3] if a.classes == 3 else OURS
    files = sorted(glob.glob(os.path.join(a.frames, "*", "*.jpg")))
    files = [f for f in files
             if os.path.basename(os.path.dirname(f)) in classes]
    excluded = []
    if a.match_branches:
        keep = branch_frames()
        if keep is None:
            print("WARNING: features.npz not found - cannot align the frame set "
                  "with Branch A/B; numbers are on all frames.")
        else:
            excluded = [f for f in files if os.path.basename(f) not in keep]
            files = [f for f in files if os.path.basename(f) in keep]

    files = stratified(files, a.limit, classes)
    if not files:
        raise SystemExit(f"no frames under {a.frames}")

    from ultralytics import YOLO
    model = YOLO(a.weights)
    det_names = model.names
    manual = dict(kv.split("=", 1) for kv in a.map.split(",") if "=" in kv)
    cmap, unmapped = resolve_map(det_names, manual, classes)

    print(f"weights      : {a.weights}")
    print(f"frames       : {len(files)} from {a.frames}")
    print(f"our classes  : {classes}")
    if excluded:
        print(f"\naligned with Branch A/B: {len(excluded)} frame(s) excluded "
              f"because MediaPipe could not landmark them, so the "
              f"landmark-based branches were never scored on them:")
        for f in excluded[:10]:
            print(f"  {os.path.basename(f)}")
        if len(excluded) > 10:
            print(f"  (+{len(excluded)-10} more)")
        print("  re-run with --all-frames to include them.")
    # A COCO-pretrained model has 80 classes; listing all of them buries the
    # two lines that matter, so only the mappings actually in use are shown.
    print(f"\ndetector classes -> ours")
    for i, raw in det_names.items():
        if i in cmap:
            print(f"  [{i}] {raw:<24} -> {cmap[i]}")
    ambiguous = [(r, g) for r, g in unmapped if g is not None]
    unrelated = [r for r, g in unmapped if g is None]
    if ambiguous:
        print("\nNOTE: these resolved to a class outside the current set and "
              "are ignored; pass --map if that is wrong:")
        for raw, guess in ambiguous:
            print(f"  {raw}  (resolved to: {guess})")
    if unrelated:
        show = ", ".join(unrelated[:8])
        more = f", +{len(unrelated)-8} more" if len(unrelated) > 8 else ""
        print(f"\n{len(unrelated)} detector class(es) unrelated to ours, "
              f"ignored: {show}{more}")
    if not cmap:
        raise SystemExit("\nno detector class maps to ours - pass --map")
    missing = set(classes) - set(cmap.values())
    if missing:
        print(f"\nNOTE: {sorted(missing)} cannot be predicted by this detector "
              f"at all — recall for those classes is 0 by construction.")

    y_true, y_pred, confs, persons = [], [], [], []
    vis_pool = []
    t0 = time.time()
    for k, f in enumerate(files):
        base = os.path.basename(f)
        true_c = os.path.basename(os.path.dirname(f))
        img = cv2.imread(f)
        res = model.predict(img, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]

        best_cls, best_conf, best_box = None, 0.0, None
        for b in res.boxes:
            cid = int(b.cls.item())
            cf = float(b.conf.item())
            if cid in cmap and cf > best_conf:
                best_cls, best_conf = cmap[cid], cf
                best_box = b.xyxy[0].tolist()

        y_true.append(true_c)
        y_pred.append(best_cls if best_cls else "(none)")
        confs.append(best_conf)
        persons.append(base.split("_")[0])
        # every frame is a candidate example; capping this at the first N would
        # bias the sheet to whichever class sorts first
        vis_pool.append((f, true_c, y_pred[-1], best_conf, best_box))
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(files)} frames  ({time.time()-t0:.0f}s)",
                  flush=True)

    dt = time.time() - t0
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    persons = np.array(persons); confs = np.array(confs)

    detected = y_pred != "(none)"
    det_rate = float(detected.mean())
    acc_all = accuracy_score(y_true, y_pred)
    acc_det = (accuracy_score(y_true[detected], y_pred[detected])
               if detected.any() else 0.0)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0)
    cpr, crc, cf1, csup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    per_class = {c: {"precision": float(cpr[i]), "recall": float(crc[i]),
                     "f1": float(cf1[i]), "support": int(csup[i])}
                 for i, c in enumerate(classes)}

    tag = weights_tag(a.weights)
    print(f"\n{'='*72}")
    print(f"Domain-gap test on our own footage - {tag}")
    print(f"{'='*72}")
    print(f"frames                    : {len(files)}")
    print(f"detection rate            : {100*det_rate:.1f}%  "
          f"({int((~detected).sum())} frames with no usable detection)")
    print(f"accuracy, all frames      : {100*acc_all:.1f}%   <- headline, "
          f"comparable to Branch A / CNN")
    print(f"accuracy, detected frames : {100*acc_det:.1f}%   <- classification "
          f"quality once it fires")
    print(f"macro precision/recall/F1 : {100*pr:.1f}% / {100*rc:.1f}% / "
          f"{100*f1:.1f}%")
    print(f"mean confidence           : {confs[detected].mean():.3f}"
          if detected.any() else "mean confidence           : n/a")
    print(f"CPU speed                 : {1000*dt/len(files):.0f} ms/frame  "
          f"({len(files)/dt:.1f} FPS)")
    print()
    print(classification_report(y_true, y_pred, labels=classes,
                                zero_division=0, digits=3))

    print("per-person accuracy (all frames)")
    per_p = {}
    for p in sorted(set(persons.tolist())):
        m = persons == p
        per_p[p] = float(accuracy_score(y_true[m], y_pred[m]))
        print(f"  {p:<10}{100*per_p[p]:>7.1f}%   "
              f"(detected {100*detected[m].mean():>5.1f}%)")

    cols = classes + (["(none)"] if (~detected).any() else [])
    cm = confusion_matrix(y_true, y_pred, labels=cols)[:len(classes)]
    cm_path = os.path.join(QC, f"yolo_{tag}_ownfootage_confusion.png")
    plot_cm(cm, classes, f"{tag} on our own footage (never trained on it)",
            f"accuracy {100*acc_all:.1f}%  ·  detection rate "
            f"{100*det_rate:.1f}%", cm_path)
    print("\nconfusion matrix ->", cm_path)

    if a.save_vis:
        rng = np.random.default_rng(0)
        wrong = [v for v in vis_pool if v[1] != v[2]]
        right = [v for v in vis_pool if v[1] == v[2]]
        pick = []
        for pool in (right, wrong):
            if pool:
                idx = rng.choice(len(pool), min(a.save_vis // 2, len(pool)),
                                 replace=False)
                pick += [pool[i] for i in idx]
        rows = [(cv2.imread(f), t, p, c, bb) for f, t, p, c, bb in pick]
        vis_path = os.path.join(QC, f"yolo_{tag}_ownfootage_examples.png")
        contact_sheet(rows, vis_path,
                      f"{tag} on our footage — correct (green) vs wrong (red)")
        print("examples ->", vis_path)

    res = {"weights": a.weights, "tag": tag, "classes": classes,
           "frames": len(files), "conf": a.conf, "imgsz": a.imgsz,
           "aligned_with_branches": bool(a.match_branches),
           "excluded_frames": [os.path.basename(f) for f in excluded],
           "detection_rate": det_rate, "accuracy_all": float(acc_all),
           "accuracy_detected": float(acc_det), "precision": float(pr),
           "recall": float(rc), "f1": float(f1),
           "per_class": per_class,
           "mean_confidence": float(confs[detected].mean()) if detected.any() else 0.0,
           "ms_per_frame_cpu": 1000 * dt / len(files),
           "class_map": {det_names[i]: v for i, v in cmap.items()},
           "unmapped_detector_classes": [r for r, _ in unmapped],
           "per_person": per_p,
           "pred_distribution": dict(Counter(y_pred.tolist()))}
    out = os.path.join(HERE, f"ownfootage_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("metrics ->", out)


if __name__ == "__main__":
    main()
