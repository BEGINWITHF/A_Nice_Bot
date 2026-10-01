"""
Fetch The Vision Weights
Brings in the detector so the bot can name what it sees (OPEN-10)

This is optional: main.py asks for the same file on the first frame it looks
at.  Run it when you would rather do the download now than during someone's
first look at the world, or when you want to know that the network is fine.
"""

from core import vision_model


def main():
    try:
        path = vision_model.fetch_model()
    except RuntimeError as exc:
        print("could not fetch the vision model: %s" % exc)
        print("the bot still sees - IT falls back to the old heuristics")
        return 1

    print("vision model ready: %s" % path)
    print("  %d bytes, sha256 %s"
          % (vision_model.MODEL_BYTES, vision_model.MODEL_SHA256))
    print("  NanoDet-m-plus-1.5x_416, %d classes, Apache-2.0"
          % len(vision_model.CLASSES))
    print("  from %s" % vision_model.MODEL_URL)

    # Prove it runs, not merely that it downloaded.
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        print("  not loaded: %s" % exc)
        return 0

    detector = vision_model.Detector(path)
    detector.detect(np.zeros((320, 320, 3), dtype="uint8"))
    if not detector.available:
        print("  downloaded but will not run: %s" % detector.error)
        return 1
    print("  loads and runs on %s" % detector.engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
