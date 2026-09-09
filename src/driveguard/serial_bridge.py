"""DriveGuard AI — serial bridge between the Python detector and the ESP32 alert unit.

Usage inside the detection code (3 lines):

    from serial_bridge import SerialAlert
    alert = SerialAlert()                 # once, before the main loop
    alert.send_stage(CURRENT_STAGE)       # every loop iteration (0..3)

Run this file directly for a hardware self-test:

    python serial_bridge.py               # then type 0/1/2/3 to switch stages, q to quit
"""
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing — run:  pip install pyserial")

BAUD = 115200
USB_CHIP_HINTS = ("CP210", "CH340", "CH910", "USB SERIAL", "SILICON LABS")


def find_esp32_port():
    """Return the COM port that looks like an ESP32 USB-serial chip.

    Deliberately does NOT fall back to "any port". Windows exposes Bluetooth
    RFCOMM ports ("Standard Serial over Bluetooth link") that open successfully
    but are not the board — a blind fallback picks one, reports "connected",
    and then silently does nothing, which is far more confusing than admitting
    the board was not found.
    """
    for p in list_ports.comports():
        desc = f"{p.description} {p.manufacturer or ''}".upper()
        if "BLUETOOTH" in desc:
            continue
        if any(h in desc for h in USB_CHIP_HINTS):
            return p.device
    return None


class SerialAlert:
    """Sends risk-stage characters to the ESP32; fails soft if unplugged."""

    def __init__(self, port=None, quiet=False):
        self.last_stage = None
        self.ser = None
        self.port_used = None
        port = port or find_esp32_port()
        self.port_used = port
        if port is None:
            if not quiet:
                print("[SerialAlert] no COM port found — running without hardware")
            return
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.1)
            time.sleep(2.0)                     # ESP32 resets when port opens
            if not quiet:
                print(f"[SerialAlert] connected on {port}")
        except Exception as e:
            if not quiet:
                print(f"[SerialAlert] could not open {port}: {e} — running without hardware")
            self.ser = None

    def send_stage(self, stage):
        stage = max(0, min(3, int(stage)))
        if stage == self.last_stage or self.ser is None:
            self.last_stage = stage
            return
        try:
            self.ser.write(str(stage).encode())
            self.last_stage = stage
        except Exception as e:
            print(f"[SerialAlert] write failed ({e}) — disabling hardware output")
            self.ser = None

    def all_off(self):
        """Put the alert unit into standby (everything dark).

        Note stage 0 is *green on*, not off — so shutting down with send_stage(0)
        would leave the green LED lit until the board was physically unplugged.
        """
        if self.ser is None:
            return
        try:
            self.ser.write(b"x")
            self.ser.flush()
            time.sleep(0.15)          # let the firmware act before the port closes
            self.last_stage = None
        except Exception:
            pass

    def close(self):
        if self.ser is not None:
            try:
                self.all_off()
                self.ser.close()
            except Exception:
                pass
            self.ser = None


if __name__ == "__main__":
    from dg_log import log_event

    print("Detected serial ports:")
    for p in list_ports.comports():
        print(f"  {p.device}  —  {p.description}")

    alert = SerialAlert()
    if alert.ser is None:
        log_event("serial_bridge", result="NO_PORT",
                  note="ESP32 not detected — check data cable / driver")
        sys.exit("Plug in the ESP32 and try again.")

    print("\nHardware self-test — type a stage number then Enter.")
    print("  0=normal  1=drowsy  2=fatigue  3=critical")
    print("  q=quit (records which stages you confirmed working)")
    confirmed = []
    while True:
        cmd = input("stage> ").strip().lower()
        if cmd == "q":
            break
        if cmd in ("0", "1", "2", "3"):
            alert.send_stage(int(cmd))
            ok = input(f"  sent stage {cmd} — did the hardware react correctly? [y/n] ").strip().lower()
            if ok.startswith("y"):
                confirmed.append(cmd)
                print("  ✓ recorded as working")
            else:
                print("  ✗ recorded as NOT working")
        else:
            print("  type 0, 1, 2, 3 or q")
    alert.close()
    all_ok = sorted(set(confirmed)) == ["0", "1", "2", "3"]
    log_event("serial_bridge", result="PASS" if all_ok else "PARTIAL",
              port=alert.port_used or "n/a",
              stages_working=",".join(sorted(set(confirmed))) or "none",
              verdict="all 4 stages confirmed" if all_ok
                      else "not all stages confirmed — check wiring for the failing ones")
    print("done — LEDs back to green." if all_ok
          else "done — some stages unconfirmed, see log.")
