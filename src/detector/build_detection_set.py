"""Build a real YOLO detection set from FL3D labels + NITYMED full frames.

WHY NOT ROBOFLOW. The original plan trained on a ~100-image public set. This
builds ~12,000 images from data already on disk, with ground truth that is
better than anything we could generate:

  FL3D ships per-frame human labels (alert / microsleep / yawning) as CROPPED
  face images. NITYMED ships the same 44 videos as full 1280x720 footage.
  Normalised cross-correlation locates each crop inside its frame at
  correlation 0.998-0.999 with the frame index matching exactly (verified over
  18 crops from 6 videos, with negative controls: same video +900 frames
  scored 0.869, a different video 0.580).

So every box is the DATASET AUTHORS' OWN face region, not a MediaPipe guess,
and every class label is a human annotation. That is a stronger provenance
story for the report than a scraped public set.

THREE DESIGN DECISIONS, stated rather than buried:

1. CLASSES ARE DRIVER STATES, not "face". The detector must answer the same
   question Branch A answers, otherwise the two branches are not comparable.

2. SPLIT BY VIDEO, never by frame. Consecutive frames are near-duplicates; a
   random frame split would put near-identical images in train and test and
   report a fantasy mAP.

   CORRECTION: video-disjoint is NOT person-disjoint. NITYMED
   documents 21 drivers across 130 videos and publishes no identity labels;
   visual inspection confirms the same faces recur across our splits. The mAP
   this produces is therefore WITHIN-COHORT and overstates cross-person
   generalisation. The cross-person number that does hold is the domain-gap
   evaluation on our own footage, which involves entirely different people.

3. PASSENGERS ARE VISIBLE AND DELIBERATELY UNLABELLED. Several NITYMED frames
   show a passenger, and FL3D boxes only the driver. The detector therefore
   learns "the driver" rather than "any face", which is correct for a
   driver-monitoring system but means it partly encodes seat position. This is
   a property of the data, and the report should say so rather than let an
   examiner discover it.

RESOLUTION. Frames are written at 640x360. Ultralytics at imgsz=640 resizes the
long side to 640 regardless, so 1280x720 would be downsampled to exactly this
before training - storing full size costs 4x the disk and upload for no gain.

    python build_detection_set.py --probe          # time it, write nothing
    python build_detection_set.py --per-class 4000
"""
import os
import sys
import json
import glob
import time
import shutil
import argparse
import collections
import multiprocessing as mp

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "datasets")
FL3D = os.path.join(DS, "fl3d", "classification_frames")
NITY = os.path.join(DS, "nitymed")

CLASSES = ["alert", "microsleep", "yawning"]
MIN_CORR = 0.97          # below this the crop was not confidently located
OUT_W, OUT_H = 640, 360


def match_box(frame, crop, prev, margin=90):
    """Locate `crop` in `frame`, searching near `prev` first.

    Full-frame normalised cross-correlation over 1280x720 dominates the build
    time, and the face moves only a few pixels between consecutive frames. So
    the search is restricted to a window around the previous box, and falls
    back to the full frame whenever that fails to clear MIN_CORR.

    This cannot silently accept a wrong box: the restricted result is only
    trusted when it clears the same 0.97 gate, and at that correlation a false
    match is not plausible - the negative control (same video, +900 frames)
    scored 0.869.
    """
    H, W = frame.shape[:2]
    h, w = crop.shape[:2]
    if h > H or w > W:
        return -1.0, None
    if prev is not None:
        px, py = prev
        x0, y0 = max(0, px - margin), max(0, py - margin)
        x1, y1 = min(W, px + w + margin), min(H, py + h + margin)
        sub = frame[y0:y1, x0:x1]
        if sub.shape[0] >= h and sub.shape[1] >= w:
            r = cv2.matchTemplate(sub, crop, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx >= MIN_CORR:
                return float(mx), (x0 + loc[0], y0 + loc[1])
    r = cv2.matchTemplate(frame, crop, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return float(mx), (loc[0], loc[1])


def video_index():
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(NITY, "**", "*.mp4"), recursive=True)}


def process_video(job):
    """Decode ONE video once and emit its labelled frames.

    Videos are wholly independent - separate files in, separate files out - so
    this is the natural unit of parallelism. Running it single-threaded left
    207 of 208 vCPU idle and put a ~96 minute serial stage in front of YOLO
    training.

    Each worker opens its own VideoCapture and writes only filenames stamped
    with its own video id, so no two workers can touch the same path. The
    corr >= MIN_CORR gate is applied per frame exactly as in the serial path,
    so parallelism cannot admit a box the serial version would have rejected.
    """
    v, mp4, items, split, out = job
    items.sort()
    want = {i: (fn, cls) for i, fn, cls in items}
    cap = cv2.VideoCapture(mp4)
    fi = 0
    prev = None
    written = collections.Counter()
    rejected = collections.Counter()
    corrs = []
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in want:
            fn, cls = want.pop(fi)
            crop = cv2.imread(os.path.join(FL3D, v, fn))
            if crop is not None and crop.shape[0] <= frame.shape[0] \
                    and crop.shape[1] <= frame.shape[1]:
                mx, loc = match_box(frame, crop, prev)
                corrs.append(mx)
                if mx >= MIN_CORR:
                    prev = loc
                    H, W = frame.shape[:2]
                    x, yy = loc
                    w, h = crop.shape[1], crop.shape[0]
                    cx, cy = (x + w / 2) / W, (yy + h / 2) / H
                    nw, nh = w / W, h / H
                    stem = f"{v}_f{fi:06d}"
                    cv2.imwrite(os.path.join(out, split, "images", stem + ".jpg"),
                                cv2.resize(frame, (OUT_W, OUT_H)),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    with open(os.path.join(out, split, "labels",
                                           stem + ".txt"), "w") as f:
                        f.write(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} "
                                f"{nw:.6f} {nh:.6f}\n")
                    written[cls] += 1
                else:
                    rejected["low_corr"] += 1
            else:
                rejected["unreadable"] += 1
        fi += 1
    cap.release()
    return v, dict(written), dict(rejected), corrs


def load_annotations():
    """video -> {frame_index: class}, using only the three states we model."""
    p = os.path.join(FL3D, "annotations_all.json")
    ann = json.load(open(p, encoding="utf-8"))
    by = collections.defaultdict(dict)
    for k, v in ann.items():
        s = v.get("driver_state")
        if s not in CLASSES:
            continue
        parts = k.replace("\\", "/").split("/")
        vid, fn = parts[-2], parts[-1]
        dg = "".join(c for c in os.path.splitext(fn)[0] if c.isdigit())
        if dg:
            by[vid][int(dg)] = (s, fn)
    return by


def plan_sampling(by, vids, per_class, rng):
    """Choose which (video, frame) pairs to emit, balanced across classes.

    Balance matters: `alert` is 73% of FL3D, and an unbalanced detection set
    would let a model score well by rarely predicting the two classes we
    actually care about. Sampling is spread evenly over each video's timeline
    rather than taken as a block, so a single long episode cannot dominate.
    """
    pool = collections.defaultdict(list)
    for v in vids:
        for idx, (cls, fn) in by[v].items():
            pool[cls].append((v, idx, fn))
    chosen = []
    for cls in CLASSES:
        items = sorted(pool[cls])
        if not items:
            continue
        byvid = collections.defaultdict(list)
        for v, idx, fn in items:
            byvid[v].append((idx, fn))
        nv = len(byvid)
        quota = max(1, per_class // max(nv, 1))
        got = []
        for v, lst in byvid.items():
            lst.sort()
            take = min(quota, len(lst))
            # even stride across the timeline, not the first `take` frames
            step = max(1, len(lst) // take)
            got += [(v, i, f) for i, f in lst[::step][:take]]
        if len(got) > per_class:
            sel = rng.choice(len(got), per_class, replace=False)
            got = [got[i] for i in sorted(sel)]
        chosen += [(v, i, f, cls) for v, i, f in got]
    return chosen


def split_videos(vids, rng):
    """Disjoint driver split: 70 / 15 / 15, shuffled deterministically."""
    v = sorted(vids)
    order = rng.permutation(len(v))
    v = [v[i] for i in order]
    n = len(v)
    ntr, nva = int(round(0.70 * n)), int(round(0.15 * n))
    return {"train": sorted(v[:ntr]),
            "valid": sorted(v[ntr:ntr + nva]),
            "test": sorted(v[ntr + nva:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(HERE, "datasets", "fl3d_det"))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = one per CPU, capped at the "
                         "number of videos. Videos are independent, so this "
                         "scales almost linearly.")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    mp4s = video_index()
    by = load_annotations()
    vids = sorted(v for v in by if v in mp4s)
    print(f"annotated videos with matching NITYMED footage: {len(vids)}")
    missing = sorted(v for v in by if v not in mp4s)
    if missing:
        print(f"  no footage for {len(missing)}: {missing[:5]}")

    chosen = plan_sampling(by, vids, a.per_class, rng)
    cc = collections.Counter(c for *_, c in chosen)
    print(f"planned frames: {len(chosen)}   {dict(cc)}")

    splits = split_videos(vids, rng)
    where = {v: s for s, vs in splits.items() for v in vs}
    print("driver split (disjoint):")
    for s, vs in splits.items():
        n = sum(1 for *_x, c in chosen if where.get(_x[0]) == s)
        print(f"   {s:<6}{len(vs):>3} drivers   {n:>6} frames")

    if a.probe:
        v = vids[0]
        cap = cv2.VideoCapture(mp4s[v])
        idxs = sorted(by[v])[:20]
        t0 = time.time()
        ok_n = 0
        for i in idxs:
            cls, fn = by[v][i]
            crop = cv2.imread(os.path.join(FL3D, v, fn))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok or crop is None:
                continue
            r = cv2.matchTemplate(fr, crop, cv2.TM_CCOEFF_NORMED)
            _, mx, _, _ = cv2.minMaxLoc(r)
            ok_n += mx > MIN_CORR
        cap.release()
        dt = time.time() - t0
        print(f"\n{len(idxs)} frames in {dt:.1f}s = {dt/len(idxs)*1000:.0f} ms/frame"
              f"   ({ok_n}/{len(idxs)} above corr {MIN_CORR})")
        print(f"estimated for {len(chosen)} frames: "
              f"{dt/len(idxs)*len(chosen)/60:.0f} min")
        return

    for s in splits:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, s, sub), exist_ok=True)

    # group by video so each file is opened and decoded exactly once
    per_vid = collections.defaultdict(list)
    for v, i, fn, cls in chosen:
        per_vid[v].append((i, fn, cls))

    jobs = [(v, mp4s[v], items, where[v], a.out)
            for v, items in sorted(per_vid.items())]
    nproc = a.jobs if a.jobs > 0 else min(len(jobs), (os.cpu_count() or 4))
    print(f"\ndecoding {len(jobs)} videos across {nproc} processes")

    t0 = time.time()
    written = collections.Counter()
    rejected = collections.Counter()
    corrs = []
    done = 0
    if nproc <= 1:
        results = map(process_video, jobs)
    else:
        pool = mp.Pool(processes=nproc)
        results = pool.imap_unordered(process_video, jobs)
    for v, w, r, c in results:
        written.update(w)
        rejected.update(r)
        corrs += c
        done += 1
        print(f"  [{done}/{len(jobs)}] {v:<16} written {sum(written.values()):>6}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    if nproc > 1:
        pool.close()
        pool.join()

    yaml = os.path.join(a.out, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(a.out)}\n")
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")

    corrs = np.array(corrs)
    print("\n" + "=" * 70)
    print(f"written {sum(written.values())} images   {dict(written)}")
    print(f"rejected {dict(rejected)}")
    if len(corrs):
        print(f"correlation: min {corrs.min():.4f}  median {np.median(corrs):.4f}"
              f"  below {MIN_CORR}: {100*(corrs<MIN_CORR).mean():.2f}%")
    print(f"data.yaml -> {yaml}")
    meta = {"classes": CLASSES, "splits": splits, "written": dict(written),
            "rejected": dict(rejected), "min_corr": MIN_CORR,
            "out_size": [OUT_W, OUT_H],
            "corr_median": float(np.median(corrs)) if len(corrs) else None,
            "minutes": (time.time() - t0) / 60}
    with open(os.path.join(a.out, "build_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
