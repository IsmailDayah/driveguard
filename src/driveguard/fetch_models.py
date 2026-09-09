"""Download the MediaPipe face landmark model.

Every branch of this project measures eye and mouth aspect ratios from a
478-point face mesh, so nothing runs without this file. It is Google's
published MediaPipe bundle rather than anything trained here, so it is fetched
on first use instead of being vendored into the repository.

    python src/driveguard/fetch_models.py
"""
import hashlib
import os
import sys
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
SIZE = 3758596

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.abspath(os.path.join(HERE, "..", "..", "models",
                                    "face_landmarker.task"))


def main():
    if os.path.exists(DEST):
        got = hashlib.sha256(open(DEST, "rb").read()).hexdigest()
        if got == SHA256:
            print("already present and verified:", DEST)
            return 0
        print("checksum mismatch, re-downloading")
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print("downloading %.1f MB ..." % (SIZE / 1048576))
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    got = hashlib.sha256(blob).hexdigest()
    if got != SHA256:
        print("REFUSING TO WRITE - sha256 mismatch")
        print("  expected", SHA256)
        print("  got     ", got)
        return 1
    with open(DEST, "wb") as f:
        f.write(blob)
    print("written and verified:", DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
