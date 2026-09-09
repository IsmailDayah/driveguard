"""DriveGuard AI — camera acceptance test (run this first).

Checks the Fantech C30 (or any webcam):
  1. What resolution/fps it ACTUALLY delivers (not what the box says).
  2. Whether face tracking is stable at demo distance (~50 cm).

Uses MediaPipe's Tasks API (FaceLandmarker) — the legacy `mp.solutions.face_mesh`
API was fully removed upstream, so this is the current, supported way to get the
same 478-point face mesh (same landmark indices, e.g. the EAR/MAR points used
elsewhere in this project still apply unchanged).

Usage:
    python camera_test.py            # default camera 0
    python camera_test.py 1          # try camera index 1 if the wrong camera opens

Requires models/face_landmarker.task. Run fetch_models.py once to download it.

PASS criteria printed at the end. Press q to finish early.
"""
import os
import sys
import time

import cv2
import numpy as np

from dg_log import log_event

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )
    HAVE_MP = True
except ImportError:
    HAVE_MP = False

cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
REQ_W = int(sys.argv[2]) if len(sys.argv) > 2 else 640
REQ_H = int(sys.argv[3]) if len(sys.argv) > 3 else 480
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "models", "face_landmarker.task")

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
# Measured on the Fantech C30 used to build this:
# this webcam's DirectShow/YUY2 path collapses above 640x480 (720p raw
# capture measured at ~2 fps; MJPG requests are silently ignored by the
# driver — cap.get(CAP_PROP_FOURCC) still reports YUY2). 640x480 measured
# a stable ~18-22 fps end-to-end and is plenty of resolution for face/eye
# landmark tracking. Override via `python camera_test.py <idx> <w> <h>` if
# testing a different camera.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # harmless if ignored
cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    sys.exit(f"Camera {cam_index} did not open — try another index (python camera_test.py 1)")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera {cam_index} opened at {w}x{h}")

landmarker = None
if HAVE_MP and os.path.exists(MODEL_PATH):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = FaceLandmarker.create_from_options(options)
elif not HAVE_MP:
    print("(mediapipe not installed in this env — fps test only)")
else:
    print(f"(models/face_landmarker.task not found - run fetch_models.py; fps test only)")

frames = 0
face_frames = 0
TEST_SECONDS = 15

# Warm up BEFORE starting the clock. The first DirectShow read (device
# negotiation) and the first FaceLandmarker inference (graph build + XNNPACK
# delegate init) each pay a one-time cost of up to a second. Starting t0 above
# them amortised that startup into the throughput figure and understated fps.
for _ in range(10):
    ok, warm = cap.read()
if landmarker is not None and ok:
    warm_rgb = cv2.cvtColor(warm, cv2.COLOR_BGR2RGB)
    landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(warm_rgb)), 0)

t0 = time.time()
last_ts = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames += 1
    elapsed = time.time() - t0

    tracked = False
    if landmarker is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # VIDEO mode requires STRICTLY increasing timestamps — two frames landing
        # in the same millisecond would raise and kill the run mid-demo.
        timestamp_ms = max(int(elapsed * 1000) + 1, last_ts + 1)
        last_ts = timestamp_ms
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks:
            face_frames += 1
            tracked = True
            for lm in result.face_landmarks[0][::12]:   # sparse overlay
                cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])),
                           1, (0, 220, 120), -1)

    fps = frames / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"{w}x{h}  {fps:5.1f} fps  face: {'YES' if tracked else 'no'}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 220, 120) if tracked else (0, 0, 255), 2)
    cv2.putText(frame, f"sit at demo distance (~50cm) — {max(0, TEST_SECONDS - int(elapsed))}s left",
                (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("DriveGuard camera acceptance test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or elapsed >= TEST_SECONDS:
        break

# Freeze the measurement window HERE, before teardown. cap.release() and
# destroyAllWindows() take ~1-2s on DirectShow; recomputing elapsed after them
# divided that dead time into the frame count and deflated fps by ~10-15%
# (which is why a 15s test kept reporting "over 17s").
elapsed = time.time() - t0

cap.release()
cv2.destroyAllWindows()
if landmarker is not None:
    landmarker.close()

fps = frames / elapsed if elapsed else 0
track_pct = 100.0 * face_frames / frames if frames and landmarker is not None else None

print("\n===== RESULT =====")
# 2 decimals on purpose: at 1 decimal a 19.96 fps run printed "20.0 fps" next
# to "fps >= 20: FAIL", which reads like a bug in the check rather than a
# genuine near-miss.
print(f"delivered:      {w}x{h} @ {fps:.2f} fps over {elapsed:.1f}s ({frames} frames)")
if track_pct is not None:
    print(f"face tracked:   {track_pct:.0f}% of frames")
fps_ok = fps >= 20
track_ok = (track_pct is None) or track_pct >= 90
print(f"fps >= 20:      {'PASS' if fps_ok else 'FAIL'}")
if track_pct is not None:
    print(f"tracking >=90%: {'PASS' if track_ok else 'FAIL — improve lighting/distance and rerun'}")
verdict = "PASS" if (fps_ok and track_ok) else "FAIL"
print("VERDICT:", "CAMERA IS GOOD — proceed with the build"
      if verdict == "PASS" else "NOT YET — fix the FAIL line above and rerun")

log_event("camera_test", resolution=f"{w}x{h}", fps=round(fps, 2), frames=frames,
          seconds=round(elapsed, 1),
          tracked_pct=(round(track_pct, 0) if track_pct is not None else "n/a"),
          fps_pass=fps_ok, track_pass=track_ok, verdict=verdict)
