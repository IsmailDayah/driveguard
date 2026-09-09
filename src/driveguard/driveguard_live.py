"""DriveGuard AI - the live system. Camera -> Branch A -> risk stage -> ESP32.

This is the deployed application the reported system metrics describe.
It deliberately contains no detection logic of its own: all of that lives in
`driveguard_engine.RiskEngine`, which `measure_system_metrics.py` also drives.
The numbers reported for latency, false alarms and stage accuracy therefore
describe THIS system rather than a re-implementation of it.

    python driveguard_live.py                 # camera 0, ESP32 auto-detected
    python driveguard_live.py --no-serial     # software only, no board needed
    python driveguard_live.py --record demo.mp4
    python driveguard_live.py --camera 1 --width 1280 --height 720

Keys:  q quit   r reset the risk score   s save a screenshot

FAILS SOFT BY DESIGN. No ESP32, no camera permission, a dropped frame or a lost
face must never crash the system. Every one of those degrades to a printed warning
and the loop continues; laptop audio remains the fallback alarm.
"""
import os
import sys
import time
import argparse
import threading
import collections

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from driveguard_engine import RiskEngine, load_branch_a, STAGE_NAMES

MODEL = os.path.join(HERE, "..", "..", "models", "face_landmarker.task")

# Laptop audio is the FALLBACK alarm: if the
# ESP32 is unplugged or fails, the driver must still be warned. It is suppressed
# whenever the board is connected, so the demo does not double-alarm.
try:
    import winsound
    HAVE_AUDIO = os.name == "nt"
except ImportError:
    HAVE_AUDIO = False

# (frequency Hz, duration ms) per stage. Stage 0 is silent - green is not an
# alarm. Pitch and repetition rise with severity to mirror the firmware.
AUDIO = {1: [(880, 120)],
         2: [(1000, 120), (1000, 120)],
         3: [(1320, 180), (1320, 180), (1320, 180)]}


def sound_stage(stage, enabled):
    """Fallback beep on a DAEMON THREAD - the detection loop must never stall.

    winsound.Beep blocks for its full duration. Called inline, a stage-3 alarm
    froze the pipeline for ~540 ms and throughput fell from 83 to 28 fps -
    i.e. the system stopped watching the driver at exactly the moment it had
    decided the driver was in danger. Measured, not theorised.

    A daemon thread keeps the alarm audible without touching frame timing, and
    dies with the process so Ctrl+C cannot leave a beeping orphan.
    """
    if not (enabled and HAVE_AUDIO):
        return
    seq = AUDIO.get(stage)
    if not seq:
        return

    def _play():
        for f, d in seq:
            try:
                winsound.Beep(f, d)
            except Exception:
                return

    threading.Thread(target=_play, daemon=True).start()


# ---------------------------------------------------------------- Branch B
# Phone use is invisible to facial geometry, so it comes from the detector
# branch instead. A zero-shot COCO YOLOv8n scored precision 1.000 and recall
# 0.768 on our own footage, which is why no fine-tuning was needed for it.
#
# It costs 38 ms per frame on this CPU against Branch A's 7.5 ms, so running it
# on every frame would drop the pipeline below the camera's 25 fps. Phone use is
# a sustained behaviour, not a millisecond event, so it is sampled every
# PHONE_EVERY frames and the verdict is held between samples.
PHONE_CLASS = 67          # COCO 'cell phone'
PHONE_CONF = 0.30         # measured precision was 1.000 at 0.25; margin added
PHONE_EVERY = 4           # ~6 Hz at 25 fps -> ~9 ms average added per frame
PHONE_HOLD_S = 1.0        # keep the verdict this long so it cannot flicker


class PhoneDetector:
    """Zero-shot phone detection, sampled rather than run every frame."""

    def __init__(self, weights):
        self.model = None
        self.ok = False
        self.last_seen = -99.0
        self.conf = 0.0
        self.box = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.ok = True
        except Exception as e:
            print(f"phone detector unavailable ({type(e).__name__}) - "
                  f"drowsiness still runs")

    def update(self, frame, n, now):
        """Run on every PHONE_EVERY-th frame; return the held verdict."""
        if not self.ok:
            return False
        if n % PHONE_EVERY == 0:
            try:
                r = self.model.predict(frame, verbose=False, device="cpu",
                                       imgsz=640, classes=[PHONE_CLASS],
                                       conf=PHONE_CONF)[0]
                if len(r.boxes):
                    i = int(r.boxes.conf.argmax())
                    self.conf = float(r.boxes.conf[i])
                    self.box = [int(v) for v in r.boxes.xyxy[i].tolist()]
                    self.last_seen = now
            except Exception:
                pass
        return (now - self.last_seen) < PHONE_HOLD_S


# BGR, matching the traffic-light hardware so screen and LEDs agree.
STAGE_BGR = {0: (80, 200, 80), 1: (60, 190, 240),
             2: (40, 140, 250), 3: (60, 60, 240)}


def _panel(frame, x0, y0, x1, y1, alpha=0.55):
    """Translucent dark plate behind a text block.

    A per-glyph outline was tried first and rejected by inspection: a 3 px
    stroke around 0.58-scale text closes the apertures of e/a/o, so
    "connected (soft)" rendered as an unreadable smear. A plate darkens the
    background instead of thickening the glyphs, so the text stays thin and
    sharp while remaining legible over a sunlit window or a projector.
    """
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, alpha, roi, 1.0 - alpha, 0.0, roi)


def _txt(frame, text, org, scale, colour, thick=1):
    """Thin, sharp text. Legibility comes from _panel, not from a stroke."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, thick, cv2.LINE_AA)


def draw_hud(frame, out, fps, ms, serial_ok, prov):
    """On-screen state. The demo has to be readable from across a room."""
    h, w = frame.shape[:2]
    stage, name = out["stage"], out["stage_name"]
    col = STAGE_BGR[stage]

    cv2.rectangle(frame, (0, 0), (w, 64), (18, 26, 43), -1)
    cv2.rectangle(frame, (0, 0), (w, 64), col, 2)
    _txt(frame, f"STAGE {stage} - {name}", (14, 42), 1.05, col, 2)

    # risk bar, with the band edges drawn on so escalation is legible live
    bx, by, bw, bh = 14, 78, w - 28, 18
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 70, 90), 1)
    frac = min(1.0, out["risk"] / 12.0)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * frac), by + bh), col, -1)
    for edge in (1.5, 4.0, 7.5):
        x = bx + int(bw * edge / 12.0)
        cv2.line(frame, (x, by - 3), (x, by + bh + 3), (150, 160, 180), 1)
    _risk_label = f"risk {out['risk']:.2f}s"

    lines = [
        f"class : {out['cls'] or '-'}",
        f"EAR   : {out['ear']:.3f}" if out["ear"] is not None else "EAR   : -",
        f"MAR   : {out['mar']:.3f}" if out["mar"] is not None else "MAR   : -",
        f"fps   : {fps:4.1f}   {ms:5.1f} ms/frame",
        f"ESP32 : {'connected' if serial_ok else 'not connected (soft)'}",
    ]
    if out.get("phone"):
        lines.insert(1, "PHONE : detected (Branch B)")
    if not out["window_full"]:
        lines.append("warming up (1.2 s window)")
    if out["reason"] == "no-face":
        lines.append("NO FACE - risk held, not decayed")
    _rw = cv2.getTextSize(_risk_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    _panel(frame, 8, by + bh + 4, bx + _rw + 10, by + bh + 24)
    _txt(frame, _risk_label, (bx, by + bh + 18), 0.5, (235, 240, 250))
    # Width measured from the longest line rather than fixed at 330 px: the
    # fixed plate ran past the text and covered part of the driver's head.
    _tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0]
              for t in lines)
    _panel(frame, 8, 122, 14 + _tw + 10, 122 + 22 * len(lines) + 6)
    for i, t in enumerate(lines):
        _txt(frame, t, (14, 140 + 22 * i), 0.58, (235, 242, 255))
    _panel(frame, 8, h - 26, w - 8, h - 4, alpha=0.5)
    _txt(frame, f"Branch A {prov['features']}  n={prov['n_train']}",
         (14, h - 12), 0.45, (200, 215, 235))
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default="",
                    help="replay a recorded clip instead of the camera. "
                         "Exercises the ENTIRE live path - landmarks, engine, "
                         "HUD, serial - so the app can be validated before the "
                         "demo without needing a webcam plugged in.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-serial", action="store_true")
    ap.add_argument("--no-phone", action="store_true",
                    help="disable the Branch B phone detector")
    ap.add_argument("--record", default="", help="write annotated video here")
    ap.add_argument("--no-display", action="store_true",
                    help="headless; still logs and drives the ESP32")
    a = ap.parse_args()

    clf, prov = load_branch_a()
    engine = RiskEngine(clf)

    phone_det = None
    if not a.no_phone:
        w = "yolov8n.pt"   # ultralytics fetches this on first use
        phone_det = PhoneDetector(w)
        print("Branch B phone detector:",
              "ready" if phone_det.ok else "unavailable")

    # ---- hardware, optional ------------------------------------------------
    alert, serial_ok = None, False
    if not a.no_serial:
        try:
            from serial_bridge import SerialAlert
            alert = SerialAlert()
            serial_ok = getattr(alert, "ser", None) is not None
            print("ESP32: connected" if serial_ok else
                  "ESP32: not found - continuing without hardware")
        except Exception as e:
            print(f"ESP32: unavailable ({type(e).__name__}) - continuing")

    if not os.path.exists(MODEL):
        raise SystemExit(f"missing {MODEL}")
    lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL)),
        running_mode=RunningMode.VIDEO, num_faces=1))

    if a.video:
        if not os.path.exists(a.video):
            raise SystemExit(f"no such video: {a.video}")
        cap = cv2.VideoCapture(a.video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {a.video}")
        print(f"replaying {a.video} through the live path")
    else:
        cap = cv2.VideoCapture(a.camera,
                               cv2.CAP_DSHOW if os.name == "nt" else 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera {a.camera}")
    gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"{'video' if a.video else 'camera ' + str(a.camera)}: "
          f"{gw}x{gh} @ {src_fps:.0f} fps")

    writer = None
    if a.record:
        writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 20.0, (gw, gh))

    lat = collections.deque(maxlen=120)
    last_ts = -1
    stage_log, t_start, frames = [], time.monotonic(), 0
    last_stage = 0
    print("running - q quit, r reset risk, s screenshot")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if a.video:
                    break            # end of file, not a dropped frame
                print("frame grab failed - retrying")
                continue
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # MediaPipe VIDEO mode requires STRICTLY increasing timestamps and
            # raises if two frames share one. Wall-clock milliseconds collide
            # whenever two frames arrive inside the same millisecond - certain
            # during a file replay, and possible on a fast camera. Derived from
            # the frame counter instead, then forced monotonic.
            ts = int(frames * 1000.0 / src_fps) if a.video                 else int((time.monotonic() - t_start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb)), ts)
            pts = None
            if res.face_landmarks:
                pts = np.array([[p.x * gw, p.y * gh]
                                for p in res.face_landmarks[0]])
            now = (frames / src_fps) if a.video else None
            _t = now if now is not None else time.monotonic() - t_start
            phone = (phone_det.update(frame, frames, _t)
                     if phone_det is not None else False)
            out = engine.step(pts, now=now, phone=phone)
            out["phone"] = phone
            ms = (time.perf_counter() - t0) * 1000.0
            lat.append(ms)
            frames += 1

            if out["stage"] != last_stage:
                # Log on the SAME clock the engine runs on. Replaying a file
                # far faster than real time made these two disagree: the engine
                # escalated at video-time 1.5 s while this line printed the
                # wall-clock 0.48 s, so the summary described a moment that
                # does not exist in the footage.
                t_log = (frames - 1) / src_fps if a.video                     else time.monotonic() - t_start
                stage_log.append((round(t_log, 2), last_stage, out["stage"]))
                print(f"  [{stage_log[-1][0]:7.2f}s] stage {last_stage} -> "
                      f"{out['stage']} ({STAGE_NAMES[out['stage']]})  "
                      f"risk {out['risk']:.2f}")
                # Escalation only: waking up must not replay the alarm, and the
                # beep is suppressed while the ESP32 is driving the buzzer.
                if out["stage"] > last_stage:
                    sound_stage(out["stage"], enabled=not serial_ok)
                last_stage = out["stage"]
            if alert is not None:
                try:
                    alert.send_stage(out["stage"])
                except Exception:
                    pass

            if pts is not None:
                for i in (33, 133, 362, 263, 13, 14):      # eye + mouth anchors
                    x, y = pts[i]
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 230, 255), -1)
            if phone and phone_det is not None and phone_det.box:
                x1, y1, x2, y2 = phone_det.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                # label BELOW the box: above it, the text landed on top of
                # the info panel whenever the phone was held high in frame.
                _txt(frame, f"phone {phone_det.conf:.2f}",
                     (x1, min(frame.shape[0] - 6, y2 + 18)), 0.5,
                     (0, 200, 255))
            fps = 1000.0 / max(np.mean(lat), 1e-6)
            frame = draw_hud(frame, out, fps, ms, serial_ok, prov)
            if writer is not None:
                writer.write(frame)
            if not a.no_display:
                cv2.imshow("DriveGuard AI", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("r"):
                    engine.reset(); last_stage = 0
                    print("  risk reset")
                if k == ord("s"):
                    p = os.path.join(HERE, f"shot_{int(time.time())}.png")
                    cv2.imwrite(p, frame); print("  saved", p)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        el = time.monotonic() - t_start
        cap.release()
        if writer is not None:
            writer.release()
        if not a.no_display:
            cv2.destroyAllWindows()
        lm.close()
        if alert is not None:
            try:
                alert.all_off(); alert.close()
            except Exception:
                pass
        print(f"\n{frames} frames in {el:.1f}s = {frames/max(el,1e-9):.1f} fps")
        if lat:
            q = np.percentile(lat, [50, 95, 99])
            print(f"per-frame latency  median {q[0]:.1f} ms   "
                  f"p95 {q[1]:.1f} ms   p99 {q[2]:.1f} ms")
        print(f"faces lost: {engine.faces_lost}/{engine.frames_seen} frames")
        print(f"stage changes: {len(stage_log)}")
        for t, fr, to in stage_log:
            print(f"  {t:7.2f}s  {fr} -> {to}")


if __name__ == "__main__":
    main()
