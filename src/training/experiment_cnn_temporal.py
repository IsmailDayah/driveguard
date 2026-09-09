"""Track 2A - the fair fight: give the CNN the same 1.2s context Branch A has.

Branch A scores 95.0% and the CNN 79.1%, and that 15.9-point headline is not a
like-for-like comparison. Branch A's winning features are PERCLOS over a 1.2s
causal window; the CNN was scored on ONE isolated 96x96 frame. Temporal
features alone moved Branch A 90.3 -> 95.0, so the comparison pits 1.2 seconds
of evidence against 40 milliseconds. An examiner could dismantle it, and so
could we - so we test it instead of defending it.

This holds the temporal context CONSTANT and varies only the feature source:

    Branch A   EAR/MAR  -> causal 1.2s window -> linear SVM
    Track 2A   CNN      -> causal 1.2s window -> linear SVM   (same window,
                                                               same folds,
                                                               same classifier)

THE LEAKAGE TRAP, and why this script trains 16 CNNs instead of 4.

A CNN's outputs on its OWN training images are in-sample: near one-hot and far
cleaner than anything it produces on a stranger. Pool those into windows, feed
them to an SVM, and the SVM learns a rule calibrated to a confidence level it
will never see at test time. The result would be pessimistic for a reason that
has nothing to do with deep learning.

So embeddings are generated NESTED. For outer fold p:

    person p        <- CNN trained on the other three   (never saw p)
    each other q    <- CNN trained on the remaining two (never saw q)

Every row the SVM sees - train and test alike - is therefore out-of-sample for
the network that produced it. 4 outer folds x (1 + 3) = 16 trainings, and the
four "trained on the other three" models are shared with the per-frame control.

The cheap non-nested variant is ALSO measured (scheme `flat`), where each
person is encoded by the CNN trained on the other three and the SVM then runs
plain LOSO on top. There the held-out person influenced the encoder that
produced the training rows. Reporting both quantifies that optimism instead of
asserting it is negligible.

REPRESENTATIONS, all pooled over the identical causal window (trailing frames
only, never crossing a clip boundary):

  frame      per-frame CNN argmax, no pooling - control, must reproduce 79.1%
  prob       3 softmax outputs -> window stats including a "neural PERCLOS":
             the fraction of the window with P(drowsy) above a threshold, which
             is exactly what PERCLOS is, with a learned closure detector in
             place of the EAR ratio. 13-d, comparable to Branch A's 3-d winner.
  emb        1280-d penultimate, mean-pooled
  embstat    1280-d, mean+std+min+max pooled (5120-d - overparameterised on
             ~595 training rows by design, to measure what that costs)
  embpca     PCA-32 (FIT ON TRAINING ROWS ONLY) then mean+std+min+max

Branch A is recomputed inside this script on the same rows and the same folds
rather than quoted, so the two numbers are guaranteed comparable.

    python experiment_cnn_temporal.py --probe        # time one CNN training
    python experiment_cnn_temporal.py
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support,
                             confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_branch_b_cnn import CropDS, build
from extract_temporal import parse

THREE = ["normal", "drowsy", "yawning"]
WINDOW = 6                      # 25 fps / stride 5 = 5 fps -> 6 frames = 1.2 s
PROB_T = (0.30, 0.50, 0.70)     # neural-PERCLOS thresholds on P(drowsy)
RUN_T = 0.50
DROWSY = THREE.index("drowsy")
REPS = ["prob", "emb", "embstat", "embpca", "fusion"]


# --------------------------------------------------------------------------
# CNN training / embedding extraction
# --------------------------------------------------------------------------
def train_cnn(X, y, epochs, bs, lr, workers, dev, seed=0):
    """Train MobileNetV2 exactly as train_branch_b_cnn.run_fold does.

    Same augmentation, class weights, label smoothing, AdamW and OneCycleLR, so
    the per-frame control reproduces the published 79.1% and any later
    difference is attributable to pooling rather than to a changed recipe.

    `seed=None` deliberately does NOT reseed. train_branch_b_cnn.py seeds once
    at module import, so its folds continue one RNG stream and each fold gets a
    different initialisation. Reseeding per fold gives every fold the SAME
    init, which is a real behavioural difference and is what the --seed-mode
    switch exists to measure.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    tr = DataLoader(CropDS(X, y, THREE, True), batch_size=bs, shuffle=True,
                    num_workers=workers, drop_last=len(y) > bs)
    model = build("mobilenet_v2", len(THREE)).to(dev)
    cnt = np.array([np.sum(np.array(y) == c) for c in THREE], np.float32)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) / len(THREE),
                     dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=0.05)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(tr)), pct_start=0.3)
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def encode(model, X, dev, workers=0):
    """Penultimate 1280-d embedding and 3-d softmax for every row.

    Eval transform only - no augmentation - through the same tensor pipeline
    the network trained under, so the embedding is exactly what the classifier
    head consumes.
    """
    model.eval()
    dl = DataLoader(CropDS(X, [THREE[0]] * len(X), THREE, False),
                    batch_size=64, shuffle=False, num_workers=workers)
    embs, probs = [], []
    for xb, _ in dl:
        f = model.features(xb.to(dev))
        e = torch.flatten(nn.functional.adaptive_avg_pool2d(f, (1, 1)), 1)
        embs.append(e.cpu().numpy())
        probs.append(torch.softmax(model.classifier(e), 1).cpu().numpy())
    return (np.vstack(embs).astype(np.float32),
            np.vstack(probs).astype(np.float32))


# --------------------------------------------------------------------------
# causal pooling - identical window semantics to extract_temporal.temporal_block
# --------------------------------------------------------------------------
def groups_of(files):
    """(person, class, clip) -> row indices ordered by frame number.

    The same grouping key extract_temporal.py uses, so a window here spans
    exactly the frames a Branch A window would.
    """
    g = {}
    for i, f in enumerate(files):
        m = parse(f)
        if m is None:
            raise SystemExit(f"unparseable filename: {f}")
        g.setdefault((m[0], m[1], m[2]), []).append((m[3], i))
    return {k: [i for _, i in sorted(v)] for k, v in g.items()}


def pool_prob(P, idx, w):
    """Window statistics over the 3 softmax outputs, mirroring Branch A.

    `neural PERCLOS` is the direct analogue of the winning feature: PERCLOS is
    the fraction of a window below an EAR threshold, this is the fraction above
    a learned drowsiness probability. Same quantity, different closure
    detector.
    """
    out = np.zeros((len(idx), 13), np.float32)
    d = P[idx, DROWSY]
    yaw = P[idx, THREE.index("yawning")]
    for j in range(len(idx)):
        lo = max(0, j - w + 1)
        seg = d[lo:j + 1]
        k = len(seg)
        col = [float((seg > t).mean()) for t in PROB_T]
        col += [float(seg.min()), float(seg.mean()),
                float(seg.std()) if k > 1 else 0.0]
        col.append(float(np.polyfit(np.arange(k), seg, 1)[0]) if k > 1 else 0.0)
        best = cur = 0
        for v in seg:
            cur = cur + 1 if v > RUN_T else 0
            best = max(best, cur)
        col.append(float(best) / max(w, 1))
        for c in range(3):
            col.append(float(P[idx, c][lo:j + 1].mean()))
        col.append(float(yaw[lo:j + 1].max()))
        col.append(float(k) / max(w, 1))
        out[j] = col
    return out


def pool_emb(E, idx, w, stats):
    """mean (and optionally std/min/max) of the embedding over the window."""
    n, dim = len(idx), E.shape[1]
    out = np.zeros((n, dim * len(stats)), np.float32)
    sub = E[idx]
    for j in range(n):
        seg = sub[max(0, j - w + 1):j + 1]
        parts = []
        for s in stats:
            if s == "mean":
                parts.append(seg.mean(0))
            elif s == "std":
                parts.append(seg.std(0) if len(seg) > 1
                             else np.zeros(dim, np.float32))
            elif s == "min":
                parts.append(seg.min(0))
            elif s == "max":
                parts.append(seg.max(0))
        out[j] = np.concatenate(parts)
    return out


def build_pooled(rep, E, P, groups, n, w, pca=None):
    """Assemble a pooled feature matrix for every row, clip by clip."""
    if rep == "prob":
        dim = 13
    elif rep == "emb":
        dim = E.shape[1]
    elif rep == "embstat":
        dim = E.shape[1] * 4
    elif rep == "embpca":
        dim = pca.n_components_ * 4
    else:
        raise SystemExit(f"unknown representation {rep}")
    Ep = pca.transform(E).astype(np.float32) if rep == "embpca" else None
    X = np.zeros((n, dim), np.float32)
    for idx in groups.values():
        if rep == "prob":
            X[idx] = pool_prob(P, idx, w)
        elif rep == "emb":
            X[idx] = pool_emb(E, idx, w, ["mean"])
        elif rep == "embstat":
            X[idx] = pool_emb(E, idx, w, ["mean", "std", "min", "max"])
        else:
            X[idx] = pool_emb(Ep, idx, w, ["mean", "std", "min", "max"])
    return X


def svm():
    return make_pipeline(StandardScaler(),
                         SVC(kernel="linear", C=1, class_weight="balanced",
                             random_state=0))


def score(tag, yt, yp, per_person=None):
    pr, rc, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=THREE, zero_division=0)
    row = {"tag": tag,
           "accuracy": float(accuracy_score(yt, yp)),
           "balanced": float(balanced_accuracy_score(yt, yp)),
           "macro_f1": float(f1.mean()),
           "drowsy_recall": float(rc[DROWSY]),
           "drowsy_precision": float(pr[DROWSY])}
    if per_person:
        row["per_person"] = {k: float(v) for k, v in per_person.items()}
    return row


def branch_a_features(files):
    """The winning Branch A subset E, aligned to the crop rows.

    Subset E (perclos_20, MAR, closed_run_w) won experiment_subsets.py at
    95.0%. The row check is not decorative: if the two archives ever diverge,
    silently scoring misaligned features would look like a result.
    """
    d = np.load(os.path.join(HERE, "features_temporal.npz"), allow_pickle=True)
    ti = {str(x): i for i, x in enumerate(d["temporal_names"])}
    keep = np.isin(d["y"], THREE)
    if not np.array_equal(d["files"][keep], files):
        raise SystemExit("features_temporal.npz rows do not match the crop rows")
    T, G = d["X_temporal"][keep], d["X_geo"][keep]
    return np.column_stack([T[:, ti["perclos_20"]], G[:, 4],
                            T[:, ti["closed_run_w"]]])


def branch_a_reference(files, y, persons, w):
    """Branch A on the SAME rows and folds, computed here rather than quoted."""
    X = branch_a_features(files)
    yt, yp, per = [], [], {}
    for p in sorted(set(persons.tolist())):
        te = persons == p
        pred = svm().fit(X[~te], y[~te]).predict(X[te])
        per[p] = accuracy_score(y[te], pred)
        yt.append(y[te]); yp.append(pred)
    return score("BranchA-EAR/MAR+1.2s", np.concatenate(yt),
                 np.concatenate(yp), per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--scheme", default="both",
                    choices=["nested", "flat", "both"])
    ap.add_argument("--probe", action="store_true",
                    help="train one CNN, report timing, exit")
    ap.add_argument("--no-cache", action="store_true",
                    help="retrain even if cached embeddings exist")
    ap.add_argument("--seed-mode", default="perfold", choices=["perfold", "once"],
                    help="perfold: reseed before every fold (all folds share an "
                         "init). once: seed once, as train_branch_b_cnn.py does, "
                         "so each fold gets a different init")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device

    d = np.load(os.path.join(HERE, "crops_aligned.npz"), allow_pickle=True)
    keep = np.isin(d["y"], THREE)
    X, y = d["faces"][keep], d["y"][keep]
    persons, files = d["persons"][keep], d["files"][keep]
    people = sorted(set(persons.tolist()))
    groups = groups_of(files)
    n = len(y)
    print(f"n={n}  persons={people}  clips={len(groups)}  "
          f"window={a.window} ({a.window/5:.1f}s)  device={dev}")

    if a.probe:
        te = persons == people[0]
        t0 = time.time()
        train_cnn(X[~te], y[~te], 1, a.batch, a.lr, a.workers, dev)
        dt = time.time() - t0
        print(f"\n1 epoch on {int((~te).sum())} images: {dt:.1f}s")
        print(f"nested run = 16 CNNs x {a.epochs} epochs "
              f"~ {dt*a.epochs*16/60:.0f} min (inner folds are smaller)")
        return

    t0 = time.time()
    # The 16 trainings are the entire cost of this experiment, and every
    # downstream question - fusion, confusion matrices, significance testing -
    # needs the SAME embeddings rather than freshly trained ones. So they are
    # cached to disk and reused. Without this, adding one analysis arm means
    # paying for 16 more trainings.
    # seed mode and seed MUST be in the cache key: reusing perfold embeddings
    # for a `once` run would silently answer the wrong question.
    if a.seed_mode == "once":
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
    fold_seed = a.seed if a.seed_mode == "perfold" else None
    cache = os.path.join(
        HERE, f"cnn_embeddings_e{a.epochs}_{a.seed_mode}{a.seed}.npz")
    store, cached = {}, False
    if os.path.exists(cache) and not a.no_cache:
        z = np.load(cache, allow_pickle=True)
        if int(z["n"]) == n and list(z["people"]) == people:
            store = {k: z[k] for k in z.files if k.startswith(("oE_", "oP_",
                                                              "iE_", "iP_"))}
            cached = True
            print(f"reusing cached embeddings from {os.path.basename(cache)} "
                  f"({len(store)} arrays) - no training needed")
        else:
            print("cache present but shape/people differ - retraining")

    # ---- 4 outer encoders: CNN_p never saw person p ----------------------
    outer_E, outer_P, frame_pred = {}, {}, {}
    for p in people:
        te = persons == p
        if cached:
            E, P = store[f"oE_{p}"], store[f"oP_{p}"]
        else:
            m = train_cnn(X[~te], y[~te], a.epochs, a.batch, a.lr,
                          a.workers, dev, seed=fold_seed)
            E, P = encode(m, X, dev, a.workers)
            store[f"oE_{p}"], store[f"oP_{p}"] = E, P
            del m
            print(f"  encoder trained excluding {p:<8} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        outer_E[p], outer_P[p] = E, P
        frame_pred[p] = np.array([THREE[i] for i in P[te].argmax(1)])

    # per-frame control - should reproduce the published 79.1%
    yt = np.concatenate([y[persons == p] for p in people])
    yp = np.concatenate([frame_pred[p] for p in people])
    per = {p: accuracy_score(y[persons == p], frame_pred[p]) for p in people}
    results = [score("CNN per-frame (control)", yt, yp, per)]
    print(f"\n  per-frame control: {100*results[0]['accuracy']:.1f}% "
          f"(published 79.1%)")

    # ---- inner encoders for the nested scheme ---------------------------
    inner_E, inner_P = {}, {}
    if a.scheme in ("nested", "both"):
        for p in people:
            for q in [x for x in people if x != p]:
                if cached:
                    E, P = store[f"iE_{p}_{q}"], store[f"iP_{p}_{q}"]
                else:
                    tr = ~np.isin(persons, [p, q])
                    m = train_cnn(X[tr], y[tr], a.epochs, a.batch, a.lr,
                                  a.workers, dev, seed=fold_seed)
                    E, P = encode(m, X, dev, a.workers)
                    store[f"iE_{p}_{q}"], store[f"iP_{p}_{q}"] = E, P
                    del m
                    print(f"  inner encoder fold={p:<8} excl {q:<8} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                inner_E[(p, q)], inner_P[(p, q)] = E, P

    if not cached:
        np.savez_compressed(cache, n=n, people=np.array(people), **store)
        print(f"\n  embeddings cached -> {os.path.basename(cache)} "
              f"({os.path.getsize(cache)/1e6:.0f} MB) - reruns are now free")

    # ---- evaluate every representation under both schemes ---------------
    # window=1 arms are controls: same classifier, same features, NO temporal
    # context. They separate "an SVM on top of CNN outputs" from "the 1.2s
    # window", so any gain cannot be credited to the wrong component.
    # `fusion` is the one arm that could actually BEAT Branch A: the CNN sees
    # appearance the geometry cannot encode (head pose, occlusion, expression),
    # while EAR/MAR carry a strong prior the CNN cannot learn from 595 images.
    # If they fail in different places the combination should win; if it does
    # not, they are making the same errors and that is worth knowing too.
    XA = branch_a_features(files)
    arms = [("prob", 1), ("prob", a.window), ("emb", 1), ("emb", a.window),
            ("embstat", a.window), ("embpca", a.window), ("fusion", a.window)]
    for scheme in (["nested", "flat"] if a.scheme == "both" else [a.scheme]):
        for rep, win in arms:
            yt, yp, per = [], [], {}
            for p in people:
                te = persons == p
                # assemble per-person encodings for this outer fold
                Ef = np.zeros_like(outer_E[p])
                Pf = np.zeros_like(outer_P[p])
                Ef[te], Pf[te] = outer_E[p][te], outer_P[p][te]
                for q in [x for x in people if x != p]:
                    m = persons == q
                    src = (inner_E[(p, q)], inner_P[(p, q)]) \
                        if scheme == "nested" else (outer_E[q], outer_P[q])
                    Ef[m], Pf[m] = src[0][m], src[1][m]
                pca = None
                if rep == "embpca":
                    pca = PCA(n_components=a.pca, random_state=0).fit(Ef[~te])
                if rep == "fusion":
                    Xp = np.column_stack(
                        [build_pooled("prob", Ef, Pf, groups, n, win), XA])
                else:
                    Xp = build_pooled(rep, Ef, Pf, groups, n, win, pca)
                pred = svm().fit(Xp[~te], y[~te]).predict(Xp[te])
                per[p] = accuracy_score(y[te], pred)
                yt.append(y[te]); yp.append(pred)
            span = "no window" if win == 1 else f"{win/5:.1f}s"
            r = score(f"CNN {rep} + {span} [{scheme}]",
                      np.concatenate(yt), np.concatenate(yp), per)
            r["scheme"], r["representation"], r["win"] = scheme, rep, win
            results.append(r)
            print(f"  {r['tag']:<34} {100*r['accuracy']:>6.1f}%", flush=True)

    results.append(branch_a_reference(files, y, persons, a.window))

    print("\n" + "=" * 82)
    print(f"{'model':<36}{'acc':>8}{'bal':>8}{'macroF1':>9}"
          f"{'drowsyR':>9}{'drowsyP':>9}")
    print("-" * 82)
    for r in results:
        print(f"{r['tag']:<36}{100*r['accuracy']:>7.1f}%{100*r['balanced']:>7.1f}%"
              f"{100*r['macro_f1']:>8.1f}%{100*r['drowsy_recall']:>8.1f}%"
              f"{100*r['drowsy_precision']:>8.1f}%")
    print("=" * 82)

    ref = [r for r in results if r["tag"].startswith("BranchA")][0]
    ctl = results[0]
    def arm(rep, win, scheme):
        m = [r for r in results if r.get("representation") == rep
             and r.get("win") == win and r.get("scheme") == scheme]
        return m[0] if m else None

    # Prefer the nested arms, but fall back to whatever scheme was actually
    # run. Hard-coding "nested" crashed a --scheme flat run AFTER printing the
    # table and BEFORE writing the JSON, losing a 100-minute result to a
    # summary line.
    pref = "nested" if a.scheme in ("nested", "both") else a.scheme
    temporal = [r for r in results
                if r.get("win") == a.window and r.get("scheme") == pref]
    best = max(temporal, key=lambda r: r["accuracy"]) if temporal else None
    print(f"\nper-frame CNN      {100*ctl['accuracy']:.1f}%")
    if best:
        print(f"best temporal CNN  {100*best['accuracy']:.1f}%  ({best['tag']})")
    print(f"Branch A           {100*ref['accuracy']:.1f}%")
    if best:
        print(f"\ngap to Branch A: {100*(ref['accuracy']-ctl['accuracy']):.1f} -> "
              f"{100*(ref['accuracy']-best['accuracy']):.1f} points")

    # what the WINDOW contributes, separated from what the SVM contributes
    print("\nwindow contribution (%s, w=1 -> w=%d):" % (pref, a.window))
    for rep in ("prob", "emb"):
        w1, wf = arm(rep, 1, pref), arm(rep, a.window, pref)
        if w1 and wf:
            print(f"  {rep:<8} {100*w1['accuracy']:.1f}% -> "
                  f"{100*wf['accuracy']:.1f}%  "
                  f"({100*(wf['accuracy']-w1['accuracy']):+.1f} points)")
    if a.scheme == "both":
        print("\nnon-nested optimism (flat - nested):")
        for rep, win in [(r, a.window) for r in REPS]:
            nn_, fl = arm(rep, win, "nested"), arm(rep, win, "flat")
            if nn_ and fl:
                print(f"  {rep:<8} {100*(fl['accuracy']-nn_['accuracy']):+.1f} points")

    out = os.path.join(HERE, "cnn_temporal.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"window": a.window, "epochs": a.epochs,
                   "minutes": (time.time() - t0) / 60,
                   "results": results}, f, indent=1)
    print(f"\nresults -> {out}   ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
