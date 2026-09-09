"""Which camera index is the webcam? Answer it in 10 seconds, before anything else.

This machine's built-in camera is DISABLED in Device Manager
(ConfigManagerErrorCode 22) and its IR camera is not exposed to DirectShow, so
a plugged-in USB webcam should land on index 0. "Should" is not "does", and the
demo is not the moment to find out - so this probes every index and reports
exactly what each one delivers.

It also reports the ACTUAL resolution and frame rate, which is not necessarily
what was requested: the Fantech C30 is a 25 fps YUY2 sensor and is bandwidth
limited, measured on 30 Jul at 25.0 fps @ 640x480 but only 6.3 fps @ 1280x720.

    python find_camera.py                # probe indices 0-4 at 640x480
    python find_camera.py --max 8        # look further
    python find_camera.py --res 1280 720 # check what 720p really costs
"""
import argparse
import time

import cv2


def probe(index, w, h, warmup=5, timed=15):
    """Open one index and measure what it really delivers.

    The first frames after opening are unreliable - exposure and gain are still
    settling and the driver may hand back stale buffers - so `warmup` frames are
    read and discarded before timing starts. Timing the warm-up is how a camera
    gets wrongly reported as slow.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(warmup):
        cap.read()
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return {"opened": True, "frames": False}
    t0 = time.perf_counter()
    got = 0
    for _ in range(timed):
        r, _f = cap.read()
        if r:
            got += 1
    dt = time.perf_counter() - t0
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return {"opened": True, "frames": True, "res": res,
            "fps": got / dt if dt > 0 else 0.0, "read": got, "asked": timed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=4)
    ap.add_argument("--res", nargs=2, type=int, default=[640, 480])
    a = ap.parse_args()
    w, h = a.res
    print(f"probing camera indices 0-{a.max} at {w}x{h}\n")
    working = []
    for i in range(a.max + 1):
        r = probe(i, w, h)
        if r is None:
            print(f"  index {i}: not present")
        elif not r["frames"]:
            print(f"  index {i}: opens but delivers NO FRAMES "
                  f"(in use by another app, or a disabled device)")
        else:
            print(f"  index {i}: {r['res'][0]}x{r['res'][1]}  "
                  f"{r['fps']:5.1f} fps  ({r['read']}/{r['asked']} frames read)")
            working.append((i, r))

    print()
    if not working:
        print("NO WORKING CAMERA FOUND.")
        print("  - is the webcam plugged into a USB-A port directly (not a hub)?")
        print("  - is another app holding it (Teams, Zoom, the Camera app)?")
        print("  - Windows camera privacy was checked as Allow, so that is not it")
        raise SystemExit(1)

    best = max(working, key=lambda t: t[1]["fps"])
    print(f"USE:  python driveguard_live.py --camera {best[0]} --no-serial")
    if best[1]["fps"] < 20:
        print(f"WARNING: {best[1]['fps']:.1f} fps is below the 20 fps the "
              f"acceptance test requires.")
        print("  Try a lower resolution, or a different USB port.")
    else:
        print(f"      {best[1]['fps']:.1f} fps at "
              f"{best[1]['res'][0]}x{best[1]['res'][1]} - above the 20 fps floor.")


if __name__ == "__main__":
    main()
