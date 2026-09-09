"""Shared logging for DriveGuard test/build scripts.

Every script that produces a result (camera_test.py, serial_bridge.py, later
the detector itself) calls log_event() once at the end. Results land in
logs/session.log (human-readable) and logs/session.jsonl (one JSON object per
line) so results can be read back directly from disk instead of being pasted
into chat.
"""
import os
import json
import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
TEXT_LOG = os.path.join(LOG_DIR, "session.log")
JSON_LOG = os.path.join(LOG_DIR, "session.jsonl")


def log_event(source, **fields):
    """Append one result line to both logs. Returns the text line written."""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {source}: " + " | ".join(f"{k}={v}" for k, v in fields.items())
    with open(TEXT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    with open(JSON_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "source": source, **fields}) + "\n")
    return line
