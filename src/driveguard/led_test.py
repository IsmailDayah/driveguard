"""DriveGuard AI — LED TEST (all-in-one alert-hardware check).

One command does everything:
  1. finds the ESP32 (ignores Windows Bluetooth COM ports)
  2. checks whether the DriveGuard firmware is already on the board
  3. if it is NOT, flashes it automatically via PlatformIO
  4. runs the 4-stage alert test so you can watch the LEDs / buzzer / vibration
  5. logs the outcome to logs/session.log

Usage:
    python led_test.py                 # auto-cycle each stage (default)
    python led_test.py --manual        # type stages yourself
    python led_test.py --reflash       # force a re-flash even if firmware is live
    python led_test.py --port COM5     # skip auto-detection
"""
import os
import subprocess
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial is missing — run:  pip install pyserial")

from serial_bridge import find_esp32_port, BAUD
from dg_log import log_event

HERE = os.path.dirname(os.path.abspath(__file__))
FW_DIR = os.path.join(HERE, "esp32_firmware")
FW_MARKER = "DriveGuard"          # printed by the firmware on boot
BOOT_WAIT = 2.5                   # ESP32 resets when the port opens

STAGES = [
    ("0", "NORMAL",   "green LED steady"),
    ("1", "DROWSY",   "amber LED + one short chirp + vibration pulse"),
    ("2", "FATIGUE",  "amber blinking + double-beep every 2s"),
    ("3", "CRITICAL", "red LED fast blink + continuous siren + vibration"),
]


def find_pio():
    """Locate the PlatformIO CLI."""
    candidates = [
        os.path.join(os.environ.get("USERPROFILE", ""), ".platformio", "penv", "Scripts", "pio.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), ".platformio", "penv", "bin", "pio"),
        "pio",
    ]
    for c in candidates:
        if c == "pio" or os.path.exists(c):
            return c
    return None


def _drain(ser, seconds):
    """Accumulate everything the board sends for `seconds`.

    A single ser.read(in_waiting or 1) is timing-dependent and often misses the
    boot banner, so collect in a loop instead.
    """
    buf = b""
    t0 = time.time()
    while time.time() - t0 < seconds:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
    return buf.decode(errors="replace")


def probe_firmware(port):
    """True if the DriveGuard firmware is running on the board.

    Checks the boot banner first. If that is missed, falls back to forcing a
    stage CHANGE and looking for the 'stage=N' echo — note the firmware only
    prints that echo when the stage actually changes, so probing with '0'
    against a freshly-booted board (which is already on stage 0) proves
    nothing. We probe with '1' and then restore '0'.
    """
    try:
        with serial.Serial(port, BAUD, timeout=1) as ser:
            if FW_MARKER in _drain(ser, BOOT_WAIT):
                return True
            ser.write(b"1")
            echo = _drain(ser, 0.8)
            ser.write(b"0")                      # restore the safe state
            time.sleep(0.2)
            return "stage=" in echo
    except Exception as e:
        print(f"  [probe] could not talk to {port}: {e}")
        return False


def flash_firmware(port):
    pio = find_pio()
    if pio is None:
        print("  [flash] PlatformIO CLI not found — open esp32_firmware/ in VS Code "
              "and click Upload instead.")
        return False
    print(f"  [flash] uploading firmware to {port} … (this takes ~15s)")
    proc = subprocess.run(
        [pio, "run", "--target", "upload", "-d", FW_DIR, "--upload-port", port],
        capture_output=True, text=True,
    )
    tail = (proc.stdout or "")[-700:] + (proc.stderr or "")[-300:]
    if proc.returncode == 0 and "SUCCESS" in tail:
        print("  [flash] SUCCESS")
        return True
    print("  [flash] FAILED — output tail:")
    print("    " + tail.replace("\n", "\n    ")[-1200:])
    return False


def run_auto_test(port, dwell=3.0):
    """Cycle through every stage so the tester can watch the hardware."""
    print(f"\nWatch the alert unit — each stage holds for {dwell:.0f}s:\n")
    acks = []
    with serial.Serial(port, BAUD, timeout=1) as ser:
        _drain(ser, BOOT_WAIT)                 # clear the boot banner
        # Prime to a non-zero stage first: the firmware only echoes on a CHANGE,
        # so testing stage 0 against a board already on 0 would look like a
        # failure when nothing is actually wrong.
        ser.write(b"3")
        _drain(ser, 0.5)
        for code, name, expect in STAGES:
            ser.write(code.encode())
            echo = _drain(ser, 0.5).strip()
            ok = f"stage={code}" in echo
            acks.append(ok)
            print(f"  stage {code} — {name:<9} → {expect}   [{'ack' if ok else 'NO ACK'}]")
            time.sleep(max(0.0, dwell - 0.5))
        ser.write(b"0")                        # leave it safe on green
    if not all(acks):
        print("\n  ⚠ some stages were not acknowledged over serial — the board may "
              "have been reset mid-test.")
    answer = input("\nDid all four stages behave as described? [y/n] ").strip().lower()
    return answer.startswith("y") and all(acks)


def run_manual_test(port):
    print("\nType a stage number then Enter.  q = finish")
    confirmed = []
    with serial.Serial(port, BAUD, timeout=1) as ser:
        time.sleep(BOOT_WAIT)
        ser.read(ser.in_waiting or 1)
        while True:
            cmd = input("stage> ").strip().lower()
            if cmd == "q":
                break
            if cmd not in ("0", "1", "2", "3"):
                print("  type 0, 1, 2, 3 or q")
                continue
            ser.write(cmd.encode())
            time.sleep(0.4)
            expect = dict((s[0], s[2]) for s in STAGES)[cmd]
            print(f"  sent {cmd} → expect: {expect}")
            if input("  correct? [y/n] ").strip().lower().startswith("y"):
                confirmed.append(cmd)
        ser.write(b"0")
    return sorted(set(confirmed)) == ["0", "1", "2", "3"]


def main():
    args = sys.argv[1:]
    manual = "--manual" in args
    reflash = "--reflash" in args
    port = None
    if "--port" in args:
        port = args[args.index("--port") + 1]

    print("═══ DriveGuard LED TEST ═══")

    port = port or find_esp32_port()
    if port is None:
        print("No ESP32 found. Check that:")
        print("  • the board is plugged in")
        print("  • the micro-USB cable is a DATA cable, not charge-only")
        print("  • the CH340/CP210x driver is installed")
        log_event("led_test", result="NO_PORT", verdict="ESP32 not detected")
        sys.exit(1)
    print(f"ESP32 found on {port}")

    print("Checking firmware …")
    live = False if reflash else probe_firmware(port)
    if live:
        print("  firmware already installed and responding ✓")
        flashed = False
    else:
        print("  firmware not detected — flashing now" if not reflash
              else "  --reflash requested")
        if not flash_firmware(port):
            log_event("led_test", result="FLASH_FAILED", port=port,
                      verdict="could not upload firmware")
            sys.exit(1)
        flashed = True
        time.sleep(1.5)
        if not probe_firmware(port):
            print("  flashed, but the board still is not responding.")
            log_event("led_test", result="NO_RESPONSE_AFTER_FLASH", port=port,
                      verdict="upload reported success but no serial reply")
            sys.exit(1)
        print("  firmware flashed and responding ✓")

    ok = run_manual_test(port) if manual else run_auto_test(port)

    log_event("led_test", result="PASS" if ok else "FAIL", port=port,
              firmware="flashed_now" if flashed else "already_present",
              mode="manual" if manual else "auto",
              verdict="all 4 alert stages confirmed working" if ok
                      else "one or more stages did not behave — check that stage's wiring")
    print("\n" + ("✅ LED TEST PASS — alert hardware fully working."
                  if ok else
                  "❌ LED TEST FAIL — see the log; recheck wiring for the bad stage."))


if __name__ == "__main__":
    main()
