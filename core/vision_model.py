"""
Vision model - what gives the eyes a real vocabulary.

`_it_processing` used to guess at labels from edges and dominant colour
("complex_object", "green_object") and then feed those guesses back into
`known_objects`, which fed them back into the next guess.  The author's
answer on 2026-10-01 to "what next for OPEN-10" was C: first patch the
quality of perception, then wire learning up (OPEN-10).  Follow-up answers:
"一个一个补，先视觉" (do one sense at a time, vision first) and, after being
told the trade-off, "只要开源是对的" - which fixed the licence at AGPL-3.0
(OPEN-20) and therefore left the detector free to be Apache-2.0.

Model: NanoDet-m-plus-1.5x_416 from opencv/opencv_zoo, Apache-2.0 (see the
LICENSE file inside that model directory).  It ships as ONNX specifically so
that ``cv2.dnn`` can run it, so this adds no new pip dependency: cv2 and
numpy are already in requirements.txt.

Everything here is imported lazily.  ``core`` modules are stdlib-only at
module scope, and a machine with no camera and no opencv must still be able
to import ``main.py``.
"""

import hashlib
import os
import time
import urllib.error
import urllib.request

# opencv_zoo ships this under Apache-2.0; the URL is immutable-ish because we
# pin both the byte count and the SHA-256 below, so an upstream change or a
# truncated/LFS-pointer download is rejected rather than executed.
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
)
MODEL_BYTES = 3800954
MODEL_SHA256 = "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"

# NanoDet-Plus anchor-free head, 416x416 input, distribution focal loss with
# reg_max=7 -> 8 bins per side -> 4 sides -> 32 values per location.
INPUT_SIZE = (416, 416)
REG_MAX = 7
STRIDES = (8, 16, 32, 64)
CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Value: 0.5.  Source: opencv_zoo's own default is 0.35, but on a 2026-10-01
# bench run (OpenCV's pedestrian clip) 0.35 also labelled a traffic cone
# "person" at 0.43 and a patch of grass at 0.38, while the people it caught
# ran 0.53-0.84.  Raised to 0.5 to buy precision over recall, because these
# labels are what `pure_learning.learn_word` will end up learning from, and a
# word learned from a false label is worse than a word never learned.
# ASSUMPTION: no literature settles a detector confidence floor; the author
# may retune it here in one place.
MIN_CONFIDENCE = 0.5
NMS_IOU = 0.6

# How long to sit out before trying to obtain the weights again after a failed
# attempt.  A lost network must cost one timeout, not one timeout per frame.
RETRY_SECONDS = 300.0


def default_model_path():
    """Where the weights live.  Not under data/: this is not a memory."""
    return os.path.join("models", "vision", "nanodet.onnx")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model(path=None, timeout=60):
    """
    Download the weights if they are not already good, and return their path.

    Raises RuntimeError when the file cannot be obtained or does not match the
    pinned hash; callers are expected to fall back to heuristic perception
    rather than to crash.  The write goes to a temporary file first so an
    interrupted download can never leave a half model where a whole one is
    expected.
    """
    path = path or default_model_path()
    if os.path.exists(path):
        if os.path.getsize(path) == MODEL_BYTES and _sha256(path) == MODEL_SHA256:
            return path
        # Anything that is not exactly the pinned bytes - a truncated download,
        # an LFS pointer, a hand-edited file - is worse than nothing, because
        # it would be loaded into the process.  Throw it away and refetch.
        os.remove(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".part"
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as response:
            with open(temp, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        if os.path.exists(temp):
            os.remove(temp)
        raise RuntimeError("cannot download the vision model: %s" % exc)

    if os.path.getsize(temp) != MODEL_BYTES or _sha256(temp) != MODEL_SHA256:
        os.remove(temp)
        raise RuntimeError("downloaded vision model failed its hash check")
    os.replace(temp, path)
    return path


def _anchors(stride):
    import numpy as np

    feat_w = INPUT_SIZE[1] // stride
    feat_h = INPUT_SIZE[0] // stride
    xs = np.arange(feat_w, dtype=np.float32) * stride
    ys = np.arange(feat_h, dtype=np.float32) * stride
    xv, yv = np.meshgrid(xs, ys)
    # Same centre convention opencv_zoo uses, so boxes line up with theirs.
    return np.column_stack((xv.flatten() + 0.5 * (stride - 1),
                            yv.flatten() + 0.5 * (stride - 1)))


class Detector:
    """
    Runs the ONNX detector on one BGR frame and returns named boxes.

    Construction never touches the network or the model; loading happens on
    the first ``detect`` and every failure mode - no cv2, no numpy, no
    download, unreadable file - is recorded in ``error`` and reported as an
    empty result instead of raising.  `see_frame` has to keep working with
    nothing but the old heuristics when this is unavailable.
    """

    def __init__(self, model_path=None, threshold=MIN_CONFIDENCE):
        self.model_path = model_path or default_model_path()
        self.threshold = threshold
        self.net = None
        self.error = None
        self.engine = None
        # No attempt before this monotonic time.  Without it a machine with no
        # network would block the sense loop on a download timeout every single
        # frame instead of once every RETRY_SECONDS.
        self._retry_after = 0.0

    @property
    def available(self):
        return self.error is None and self.net is not None

    def _fail(self, message):
        self.error = message
        self.net = None
        return []

    def _load(self):
        now = time.monotonic()
        if now < self._retry_after:
            return
        self._retry_after = now + RETRY_SECONDS

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            return self._fail("missing dependency: %s" % exc)

        try:
            path = fetch_model(self.model_path)
        except RuntimeError as exc:
            return self._fail(str(exc))

        # ENGINE_NEW is the OpenCV 5 graph engine (measured at 14ms a frame on
        # this machine against 80ms for the classic one).  OpenCV 4 has neither
        # the enum nor the parameter, so fall back to plain readNet there.
        net, engine = None, None
        for name in ("ENGINE_NEW", "ENGINE_CLASSIC"):
            if not hasattr(cv2.dnn, name):
                continue
            candidate = getattr(cv2.dnn, name)
            try:
                net = cv2.dnn.readNetFromONNX(path, candidate)
            except (cv2.error, TypeError):
                continue
            engine = name
            break
        if net is None:
            try:
                net = cv2.dnn.readNet(path)
            except cv2.error as exc:
                return self._fail("cannot read the vision model: %s" % exc)

        self.net, self.engine = net, engine
        # Warm the graph up here rather than paying for it on the first frame
        # the bot ever looks at.
        try:
            net.setInput(cv2.dnn.blobFromImage(np.zeros(
                (INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32)))
            net.forward(net.getUnconnectedOutLayersNames())
        except cv2.error as exc:
            return self._fail("vision model will not run: %s" % exc)
        return []

    def detect(self, frame, threshold=None):
        """Return [{"label", "confidence", "box": [x1, y1, x2, y2]}], frame coords."""
        if threshold is None:
            threshold = self.threshold
        if self.net is None:
            self._load()
        if self.net is None:
            return []

        import cv2

        try:
            height, width = frame.shape[:2]
            boxed, scale = _letterbox(frame)
            net = self.net
            net.setInput(cv2.dnn.blobFromImage(_normalize(boxed)))
            outs = net.forward(net.getUnconnectedOutLayersNames())
        except cv2.error as exc:
            return self._fail("vision model stopped running: %s" % exc)

        try:
            return _post_process(outs, (height, width), scale, threshold)
        except ValueError as exc:
            return self._fail(str(exc))


def _post_process(outs, shape, scale, threshold):
    """
    Raw network outputs -> detections in the frame's own coordinates.

    Kept out of ``detect`` on purpose: this is the half that can be wrong in an
    interesting way, and a test should be able to check it with no camera and
    no 3.8MB download.
    """
    import cv2
    import numpy as np

    # The classic and the new engine emit outputs in different orders
    # (interleaved cls/reg vs grouped cls then reg), so pair them by what
    # they are rather than by where they sit.  opencv_zoo's own nanodet.py
    # indexes with preds[::2] and crashes under OpenCV 5 for exactly this
    # reason.
    cls_by_len, reg_by_len = {}, {}
    for out in outs:
        flat = out.reshape(-1, out.shape[-1])
        if flat.shape[1] == len(CLASSES):
            cls_by_len[flat.shape[0]] = flat
        elif flat.shape[1] == (REG_MAX + 1) * 4:
            reg_by_len[flat.shape[0]] = flat
    if not cls_by_len or set(cls_by_len) != set(reg_by_len):
        raise ValueError("unexpected output shape from the vision model")

    boxes, scores, labels = [], [], []
    for length in sorted(cls_by_len, key=int):
        stride = round(INPUT_SIZE[1] / (length ** 0.5))
        if stride not in STRIDES:
            continue
        scores_at = cls_by_len[length]
        kept = scores_at.max(axis=1) > threshold
        if not kept.any():
            continue
        boxes.append(_decode_box(reg_by_len[length], stride)[kept])
        scores.append(scores_at.max(axis=1)[kept])
        labels.append(scores_at.argmax(axis=1)[kept])

    if not boxes:
        return []
    boxes = np.concatenate(boxes).astype(np.float32)
    scores = np.concatenate(scores).astype(np.float32)
    labels = np.concatenate(labels)

    keep = cv2.dnn.NMSBoxes(
        [(float(x), float(y), float(w), float(h)) for x, y, w, h in boxes],
        scores.tolist(), threshold, NMS_IOU)
    results = []
    for index in list(keep)[:32]:
        x1, y1, w, h = boxes[index]
        results.append({
            "label": CLASSES[int(labels[index])],
            "confidence": round(float(scores[index]), 3),
            "box": _unletterbox([x1, y1, x1 + w, y1 + h], shape, scale),
        })
    return results


def _letterbox(frame):
    """Scale to fit and pad to exactly INPUT_SIZE, preserving aspect ratio."""
    import cv2

    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE[0] / height, INPUT_SIZE[1] / width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    boxed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (INPUT_SIZE[0] - new_h) // 2
    left = (INPUT_SIZE[1] - new_w) // 2
    boxed = cv2.copyMakeBorder(
        boxed, top, INPUT_SIZE[0] - new_h - top,
        left, INPUT_SIZE[1] - new_w - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return boxed, (top, left, new_h, new_w)


def _normalize(boxed):
    """BGR -> RGB, then the mean/std the model was trained with."""
    import cv2
    import numpy as np

    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (rgb - mean) / std


def _decode_box(reg, stride, kept=None):
    """Distribution focal loss -> four distances, in input-image pixels."""
    import numpy as np

    project = np.arange(REG_MAX + 1, dtype=np.float32)
    raw = reg.reshape(-1, REG_MAX + 1)
    # Subtract the row max before exp: NanoDet's head emits logits, and an
    # unshifted exp overflows to inf on a confident location.
    raw = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(raw)
    distance = (exp / exp.sum(axis=1, keepdims=True)) @ project
    distance = distance.reshape(-1, 4) * stride
    points = _anchors_for(stride)
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.column_stack((x1, y1, x2 - x1, y2 - y1)).astype(np.float32)


def _anchors_for(stride):
    import numpy as np

    cached = getattr(_anchors_for, "_cache", None)
    if cached is None:
        cached = _anchors_for._cache = {}
    if stride not in cached:
        cached[stride] = _anchors(stride)
    return cached[stride]


def _unletterbox(box, shape, scale):
    """Map a box from the 416x416 working image back to the real frame."""
    top, left, new_h, new_w = scale
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(shape[1]), (x1 - left) * shape[1] / new_w)),
        max(0.0, min(float(shape[0]), (y1 - top) * shape[0] / new_h)),
        max(0.0, min(float(shape[1]), (x2 - left) * shape[1] / new_w)),
        max(0.0, min(float(shape[0]), (y2 - top) * shape[0] / new_h)),
    ]
