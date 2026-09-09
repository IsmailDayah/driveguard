"""Pre-demo readiness check: is the person, the seat and the light demo-ready?

`camera_test.py` answers "does the camera work". This answers "will the SYSTEM
behave for THIS person, in THIS chair, under THIS light" - which is a different
question, and the one that has bitten this project before.

Every check below exists because something measurable went wrong earlier:

  RESTING EAR vs the 0.20 cutoff   The shipped threshold of 0.26 called 86.9% of
                                   P2's wide-awake frames "eyes shut". With a
                                   sustained-closure trigger the demo would have
                                   alarmed continuously for half the team, and
                                   nobody had tested it on them (the threshold analysis). If the
                                   sitter's resting EAR is near 0.20, the same
                                   failure returns.

  EYE-REGION brightness + detail   The night failure cost 16.3 points and was
                                   traced NOT to whole-frame darkness (119.6 ->
                                   114.9, barely moved) but to the EYE REGION
                                   being 69% darker with 53% less fine detail
                                   (the illumination analysis). Whole-frame brightness hides
                                   it, so the eye strip is measured directly.

  INTER-OCULAR DISTANCE            Proxy for how far away the sitter is. The
                                   corpus was recorded at ~50 cm; sitting much
                                   further shrinks the eye region and the same
                                   detail collapse follows.

  FACE CENTRING + tracking rate    A face drifting out of frame is the one
                                   failure an examiner will notice instantly.

    python demo_readiness.py                 # 10 s check on camera 0
    python demo_readiness.py --seconds 15 --camera 1
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, "..", "features"))
sys.path.insert(0, TRAIN)
from extract_features import ear_of, mar_of, LEFT_EYE, RIGHT_EYE, MOUTH

MODEL = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")

# Reference values measured on our own corpus, for comparison.
DAY_EYE_BRIGHT, DAY_EYE_DETAIL = 89.6, 2609.0     # daytime reference
NIGHT_EYE_BRIGHT, NIGHT_EYE_DETAIL = 27.7, 1220.0  # the failing condition
EAR_CUTOFF = 0.20                                  # the threshold analysis, tuned on our data


def eye_strip_stats(gray, pts):
    """Brightness / contrast / fine detail in the EYE REGION only.

    Anchored on inter-ocular distance so it is scale invariant, matching the
    geometry used in experiment_fill.py.
    """
    lc, rc = pts[LEFT_EYE].mean(axis=0), pts[RIGHT_EYE].mean(axis=0)
    iod = float(np.hypot(*(lc - rc)))
    if iod < 8:
        return None
    cx, cy = (lc + rc) / 2.0
    h, w = gray.shape
    x0, x1 = int(max(0, cx - 1.3 * iod)), int(min(w, cx + 1.3 * iod))
    y0, y1 = int(max(0, cy - 0.45 * iod)), int(min(h, cy + 0.45 * iod))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    return {"iod": iod, "bright": float(roi.mean()),
            "contrast": float(roi.std()),
            "detail": float(cv2.Laplacian(roi, cv2.CV_64F).var()),
            "glare_pct": float(100.0 * (roi > 220).mean()),
            "cx": float(cx / w), "cy": float(cy / h)}


def verdict(ok, msg_ok, msg_bad):
    return (f"  [OK]   {msg_ok}" if ok else f"  [FIX]  {msg_bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    if not os.path.exists(MODEL):
        raise SystemExit(f"missing {MODEL}")
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.camera} - run find_camera.py")
    for _ in range(5):                     # discard warm-up frames
        cap.read()

    lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL)),
        running_mode=RunningMode.VIDEO, num_faces=1))

    print(f"Sit as you will for the demo and look at the camera. "
          f"Measuring {a.seconds:.0f} s ...\n")
    rows, frames, lost = [], 0, 0
    best_frame, best_pts = None, None
    t0 = time.monotonic()
    ts = 0
    while time.monotonic() - t0 < a.seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frames += 1
        ts += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = lm.detect_for_video(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))),
            ts)
        if not res.face_landmarks:
            lost += 1
            continue
        h, w = gray.shape
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        st = eye_strip_stats(gray, pts)
        if st is None:
            continue
        st["ear"] = (ear_of(pts, LEFT_EYE) + ear_of(pts, RIGHT_EYE)) / 2.0
        st["mar"] = mar_of(pts, MOUTH)
        st["frame_bright"] = float(gray.mean())
        rows.append(st)
        if best_frame is None:
            best_frame, best_pts = frame.copy(), pts
    dt = time.monotonic() - t0
    cap.release()
    lm.close()

    if not rows:
        raise SystemExit("No face measured. Is anyone in front of the camera?")

    g = lambda k: np.array([r[k] for r in rows])
    fps = frames / dt
    track = 100.0 * len(rows) / max(frames, 1)
    ear, iod = g("ear"), g("iod")
    bright, detail = g("bright"), g("detail")

    print("=" * 72)
    print("MEASURED")
    print("=" * 72)
    print(f"  frames {frames} in {dt:.1f}s = {fps:.1f} fps   "
          f"face tracked {track:.1f}%")
    print(f"  inter-ocular distance   {iod.mean():6.1f} px  "
          f"(sd {iod.std():.1f})")
    print(f"  resting EAR             {ear.mean():6.3f}  "
          f"(min {ear.min():.3f}, 5th pct {np.percentile(ear,5):.3f})")
    print(f"  MAR                     {g('mar').mean():6.3f}")
    print(f"  whole-frame brightness  {g('frame_bright').mean():6.1f}")
    print(f"  EYE-REGION brightness   {bright.mean():6.1f}   "
          f"(our daytime {DAY_EYE_BRIGHT}, failing night {NIGHT_EYE_BRIGHT})")
    print(f"  EYE-REGION detail       {detail.mean():6.0f}   "
          f"(our daytime {DAY_EYE_DETAIL:.0f}, failing night {NIGHT_EYE_DETAIL:.0f})")
    print(f"  glare in eye region     {g('glare_pct').mean():6.2f}%")
    print(f"  face centre             x {g('cx').mean():.2f}  y {g('cy').mean():.2f} "
          f"(0.5/0.5 is centred)")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    fails = 0

    ok = fps >= 20
    fails += not ok
    print(verdict(ok, f"{fps:.1f} fps, above the 20 fps floor",
                  f"{fps:.1f} fps is BELOW 20 - close other apps, use a direct USB port"))

    ok = track >= 90
    fails += not ok
    print(verdict(ok, f"face tracked {track:.1f}% of frames",
                  f"face tracked only {track:.1f}% - centre yourself and improve lighting"))

    # THE critical one: resting eyes must sit clearly above the closure cutoff
    margin = ear.mean() - EAR_CUTOFF
    ok = margin >= 0.06
    fails += not ok
    print(verdict(ok,
                  f"resting EAR {ear.mean():.3f} is {margin:+.3f} above the "
                  f"{EAR_CUTOFF} cutoff - comfortable headroom",
                  f"resting EAR {ear.mean():.3f} is only {margin:+.3f} from the "
                  f"{EAR_CUTOFF} cutoff. THE SYSTEM MAY ALARM WHILE YOU ARE AWAKE "
                  f"(this is exactly the night-time failure mode). Open your eyes wider, raise "
                  f"the camera to eye level, or brighten your face"))

    ok = 55 <= iod.mean() <= 120
    fails += not ok
    print(verdict(ok, f"IOD {iod.mean():.0f} px - a sensible demo distance",
                  f"IOD {iod.mean():.0f} px - "
                  + ("too far, move closer to ~50 cm" if iod.mean() < 55
                     else "too close, back off to ~50 cm")))

    ok = bright.mean() >= 55
    fails += not ok
    print(verdict(ok, f"eye region {bright.mean():.0f} - well lit",
                  f"eye region only {bright.mean():.0f} (failing night case was "
                  f"{NIGHT_EYE_BRIGHT}). Put light IN FRONT of your face, not behind"))

    ok = detail.mean() >= 1500
    fails += not ok
    print(verdict(ok, f"eye detail {detail.mean():.0f} - eyelid edge resolvable",
                  f"eye detail {detail.mean():.0f} is low (failing night case was "
                  f"{NIGHT_EYE_DETAIL:.0f}). This is the variable that actually "
                  f"controls accuracy - add a frontal light"))

    ok = abs(g("cx").mean() - 0.5) < 0.18 and abs(g("cy").mean() - 0.5) < 0.22
    fails += not ok
    print(verdict(ok, "face is reasonably centred",
                  "face is off-centre - recentre so you stay in frame while moving"))

    print("\n" + "=" * 72)
    if fails == 0:
        print("READY - every check passed. Run:  python driveguard_live.py --no-serial")
    else:
        print(f"{fails} check(s) need attention above. Fix and re-run.")
    print("=" * 72)

    if best_frame is not None:
        for i in (33, 133, 362, 263, 13, 14):
            x, y = best_pts[i]
            cv2.circle(best_frame, (int(x), int(y)), 3, (0, 230, 255), -1)
        lc = best_pts[LEFT_EYE].mean(axis=0); rc = best_pts[RIGHT_EYE].mean(axis=0)
        iodv = float(np.hypot(*(lc - rc))); cx, cy = (lc + rc) / 2.0
        cv2.rectangle(best_frame,
                      (int(cx - 1.3*iodv), int(cy - 0.45*iodv)),
                      (int(cx + 1.3*iodv), int(cy + 0.45*iodv)), (0, 200, 255), 2)
        p = os.path.join(HERE, "readiness_frame.png")
        cv2.imwrite(p, best_frame)
        print(f"\nframe saved for inspection -> {p}")


if __name__ == "__main__":
    main()
