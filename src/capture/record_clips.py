"""DriveGuard AI — labelled clip recorder (image data collection).

Captures short labelled video clips of each team member for the four classes
the project needs, and organises them into the layout the training pipeline
expects:

    data/raw/<class>/<person>_<class>_<n>.mp4
    data/frames/<class>/<person>_<class>_<n>_f0007.jpg     (extracted)

Classes
    normal   — eyes open, facing forward, alert
    drowsy   — eyes closed / very heavy lids (simulate a micro-sleep)
    yawning  — mouth wide open, mid-yawn
    phone    — holding a phone up near the face / looking down at it

Usage
    python record_clips.py                     # interactive, asks who you are
    python record_clips.py --person P1     # skip the name prompt
    python record_clips.py --status            # coverage report only
    python record_clips.py --extract           # turn existing clips into frames
    python record_clips.py --seconds 8         # clip length (default 6)
    python record_clips.py --person P1 --only normal --count 2
                                               # add 2 EXTRA clips of one class

Controls while recording
    SPACE  start / stop the current clip
    n      skip to the next class
    q      quit
"""
import os
import sys
import time
import glob

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "..", "data", "raw")
FRAMES = os.path.join(HERE, "..", "..", "data", "frames")

CLASSES = [
    ("normal",  "Eyes open, look at camera. Blink naturally."),
    ("drowsy",  "Close your eyes and KEEP them closed."),
    ("yawning", "HOLD mouth wide open. Do NOT cover it with your hand."),
    ("phone",   "Phone FULLY in frame, held up near your face."),
]

# Printed once before recording. The first two rules exist because a real clip
# had to be thrown away: it was shot in a different room, so the background
# correlated with the 'phone' label and a classifier could have learned the
# room instead of the behaviour.
SETUP_RULES = """
SETUP RULES - read before recording
  1. DO NOT MOVE between classes. Same chair, same room, same camera position
     for all 8 clips, otherwise the background leaks into the labels.
  2. Keep the phone FULLY inside the frame - not clipped at the bottom edge.
  3. Light on your FACE, not behind you. Avoid a bright window behind your head.
  4. Sit ~50 cm from the camera, face roughly centred.
  5. Press q to stop - do not unplug the webcam mid-recording.
  6. YAWNING: hold the mouth wide open for most of the 6 seconds and keep
     your HAND AWAY from your face. A polite hand-over-mouth yawn hides the
     mouth from the camera and breaks face tracking, so the clip cannot be
     used - real clips were rejected for exactly this.
"""
TEAM = ["P1", "P2", "P3", "P4", "P5"]

# Matches the detector's capture settings so the training data has the same
# resolution and framing as the live system (see driveguard_live.py).
CAP_W, CAP_H, CAP_FPS = 640, 480, 25
TARGET_CLIPS_PER_CLASS = 2      # per person, per class


def ensure_dirs():
    for c, _ in CLASSES:
        os.makedirs(os.path.join(RAW, c), exist_ok=True)
        os.makedirs(os.path.join(FRAMES, c), exist_ok=True)


MAX_READ_FAILS = 5          # consecutive dead frames before giving up


def _configure(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    return cap


def try_open_camera():
    """Open the webcam, or return None (does not exit)."""
    cap = _configure(cv2.VideoCapture(0, cv2.CAP_DSHOW))
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    return cap


def open_camera():
    cap = try_open_camera()
    if cap is None:
        sys.exit("Camera 0 did not open. Close any app using the webcam and retry.")
    return cap


def existing_count(person, cls):
    return len(glob.glob(os.path.join(RAW, cls, f"{person}_{cls}_*.mp4")))


def next_index(person, cls):
    return existing_count(person, cls) + 1


def status():
    ensure_dirs()
    print("\nDataset coverage  (target: "
          f"{TARGET_CLIPS_PER_CLASS} clips per person per class)\n")
    header = f"{'person':<10}" + "".join(f"{c:>10}" for c, _ in CLASSES) + f"{'done':>8}"
    print(header)
    print("-" * len(header))
    total = 0
    for person in TEAM:
        row = f"{person:<10}"
        person_ok = True
        for cls, _ in CLASSES:
            n = existing_count(person, cls)
            total += n
            row += f"{n:>10}"
            if n < TARGET_CLIPS_PER_CLASS:
                person_ok = False
        row += f"{'YES' if person_ok else 'no':>8}"
        print(row)
    print("-" * len(header))
    print(f"\ntotal clips: {total}   "
          f"(target {len(TEAM) * len(CLASSES) * TARGET_CLIPS_PER_CLASS})")
    frames = sum(len(glob.glob(os.path.join(FRAMES, c, '*.jpg'))) for c, _ in CLASSES)
    print(f"extracted frames: {frames}")
    print("\nAnyone with 'no' still needs to record. Run:  python record_clips.py")


def record(person, seconds, only_cls=None, count=None):
    ensure_dirs()
    cap = open_camera()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    print(SETUP_RULES)
    print(f"Recording for: {person}")
    print("SPACE = start/stop clip   n = next class   q = quit\n")

    todo = CLASSES if only_cls is None else [c for c in CLASSES if c[0] == only_cls]
    if only_cls is not None and not todo:
        print(f"unknown class '{only_cls}' - choose from "
              f"{[c for c, _ in CLASSES]}")
        cap.release()
        return

    for cls, hint in todo:
        have = existing_count(person, cls)
        # --only means "record this many MORE", so raise the bar above what
        # already exists instead of skipping a finished class.
        target = (TARGET_CLIPS_PER_CLASS if only_cls is None
                  else have + (count if count else TARGET_CLIPS_PER_CLASS))
        # Auto-skip classes that are already finished. Previously the recorder
        # still opened the live window for them and sat waiting for 'n', which
        # looked like it had restarted from the beginning.
        if have >= target:
            print(f"--- {cls.upper()} ({have}/{target}) - already done, skipping ---")
            continue
        print(f"--- {cls.upper()} ({have}/{target} recorded) ---")
        print(f"    {hint}")
        while True:
            writer = None
            recording = False
            started = 0.0
            frames_written = 0
            advance = False
            quit_all = False

            fail_count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    # Unplugging the webcam (or another app stealing it) makes
                    # every read fail. Previously this broke out and the outer
                    # loop retried instantly - a tight spin on a dead camera,
                    # which froze the window and left garbled text on screen.
                    fail_count += 1
                    print(f"    [camera] frame read failed "
                          f"({fail_count}/{MAX_READ_FAILS}) - trying to reopen")
                    if writer is not None:
                        writer.release()
                        writer = None
                        recording = False
                        print("    [camera] partial clip closed and kept")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(1.2)
                    cap = try_open_camera()
                    if cap is None or fail_count >= MAX_READ_FAILS:
                        print("    [camera] could not recover - exiting cleanly.\n"
                              "    Your saved clips are safe. Re-run the same command "
                              "to continue where you left off.")
                        cv2.destroyAllWindows()
                        status()
                        return
                    continue
                fail_count = 0
                disp = frame.copy()
                have_now = existing_count(person, cls)

                if recording:
                    writer.write(frame)
                    frames_written += 1
                    left = seconds - (time.time() - started)
                    cv2.rectangle(disp, (0, 0), (CAP_W, CAP_H), (0, 0, 255), 6)
                    cv2.putText(disp, f"REC {left:4.1f}s", (14, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if left <= 0:
                        writer.release()
                        writer = None
                        recording = False
                        print(f"    saved clip ({frames_written} frames)")
                        break
                else:
                    cv2.putText(disp, f"{person} / {cls}  ({have_now}/{target})",
                                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2)
                    cv2.putText(disp, "SPACE=record  n=next  q=quit", (14, CAP_H - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(disp, hint[:52], (14, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("DriveGuard clip recorder", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and not recording:
                    idx = next_index(person, cls)
                    path = os.path.join(RAW, cls, f"{person}_{cls}_{idx}.mp4")
                    writer = cv2.VideoWriter(path, fourcc, CAP_FPS, (CAP_W, CAP_H))
                    recording = True
                    started = time.time()
                    frames_written = 0
                    print(f"    recording -> {os.path.basename(path)}")
                elif key == ord(" ") and recording:
                    writer.release(); writer = None; recording = False
                    print(f"    stopped early ({frames_written} frames)")
                    break
                elif key == ord("n"):
                    advance = True
                    break
                elif key == ord("q"):
                    quit_all = True
                    break

            if writer is not None:
                writer.release()
            if quit_all:
                cap.release(); cv2.destroyAllWindows(); status(); return
            if advance:
                break
            if existing_count(person, cls) >= target:
                print(f"    {cls} complete\n")
                break

    cap.release()
    cv2.destroyAllWindows()
    status()


def extract(every_n=5):
    """Turn recorded clips into individual JPGs for classifier training."""
    ensure_dirs()
    total = 0
    for cls, _ in CLASSES:
        for clip in sorted(glob.glob(os.path.join(RAW, cls, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(clip))[0]
            cap = cv2.VideoCapture(clip)
            i = kept = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % every_n == 0:
                    out = os.path.join(FRAMES, cls, f"{stem}_f{i:05d}.jpg")
                    cv2.imwrite(out, frame)
                    kept += 1
                i += 1
            cap.release()
            total += kept
            print(f"  {os.path.basename(clip)} -> {kept} frames")
    print(f"\nextracted {total} frames into {FRAMES}")


def main():
    args = sys.argv[1:]
    if "--status" in args:
        status(); return
    if "--extract" in args:
        extract(); return

    seconds = 6
    if "--seconds" in args:
        seconds = float(args[args.index("--seconds") + 1])

    only_cls = None
    if "--only" in args:
        only_cls = args[args.index("--only") + 1].strip().lower()
    count = None
    if "--count" in args:
        count = int(args[args.index("--count") + 1])

    if "--person" in args:
        person = args[args.index("--person") + 1].strip().lower()
    else:
        print("Who is recording?  " + " / ".join(TEAM))
        person = input("name: ").strip().lower()
    if person not in TEAM:
        print(f"'{person}' is not in the team list {TEAM} — using it anyway.")
    record(person, seconds, only_cls, count)


if __name__ == "__main__":
    main()
