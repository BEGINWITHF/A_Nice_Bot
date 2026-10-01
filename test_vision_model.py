"""
Tests for the detector behind the new vision (OPEN-10).

Design clauses:
    OPEN-10  patch perception quality first, one sense at a time, vision
             first (author, 2026-10-01)
    OPEN-20  the repository is AGPL-3.0, which is what lets the Apache-2.0
             NanoDet weights be brought in at all
    DATA-5   what a memory may hold - here, a name and not a parameter
    IO-5     a frame is sampled and dropped, never written down

Only one of these tests needs the 3.8MB of weights, and it skips when they
are not present, so a fresh clone runs the suite offline.

Run:
    python -m pytest test_vision_model.py -v
"""

import hashlib
import io
import os
import urllib.error

import pytest

try:
    import numpy as np
except ImportError:                     # pragma: no cover - bare interpreter
    np = None
try:
    import cv2
except ImportError:                     # pragma: no cover - bare interpreter
    cv2 = None

needs_numpy = pytest.mark.skipif(np is None, reason="numpy is not installed")
needs_cv2 = pytest.mark.skipif(cv2 is None, reason="opencv is not installed")

from core import vision_model as vm

# Captured here, before any fixture runs, so a test can put the real fetch
# back after conftest has blocked it.
REAL_FETCH = vm.fetch_model

# A 768x576 frame letterboxes to (top=52, left=0, 312x416); kept here so the
# expectations below are worked out by hand rather than by the code under test.
FRAME_SHAPE = (576, 768)
FRAME_SCALE = (52, 0, 312, 416)


# ---------------------------------------------------------------------------
# What the model promises
# ---------------------------------------------------------------------------

def test_the_label_list_is_the_eighty_coco_classes():
    assert len(vm.CLASSES) == 80
    assert len(set(vm.CLASSES)) == 80
    assert vm.CLASSES[0] == "person"
    assert vm.CLASSES[-1] == "toothbrush"


def test_every_threshold_and_limit_is_a_plain_number():
    assert 0.0 < vm.MIN_CONFIDENCE < 1.0
    assert 0.0 < vm.NMS_IOU < 1.0
    assert vm.RETRY_SECONDS > 0


# ---------------------------------------------------------------------------
# Obtaining the weights (conftest keeps the network switched off)
# ---------------------------------------------------------------------------

def _offline(*args, **kwargs):
    raise urllib.error.URLError("the network is switched off for this test")


def test_weights_that_do_not_match_the_pin_are_discarded_not_trusted(
        monkeypatch, tmp_path):
    target = tmp_path / "nanodet.onnx"
    target.write_bytes(b"a truncated download, or an LFS pointer, or garbage")
    monkeypatch.setattr(vm.urllib.request, "urlopen", _offline)

    with pytest.raises(RuntimeError, match="cannot download"):
        REAL_FETCH(str(target))

    # It had to go: leaving it would mean loading unverified bytes into the
    # process the next time the model is needed.
    assert not target.exists()
    assert not os.path.exists(str(target) + ".part")


def test_a_good_download_is_hashed_before_it_is_kept(monkeypatch, tmp_path):
    payload = b"stand-in for 3.8MB of ONNX"
    target = tmp_path / "nanodet.onnx"
    monkeypatch.setattr(vm, "MODEL_BYTES", len(payload))
    monkeypatch.setattr(vm, "MODEL_SHA256",
                        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(vm.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(payload))

    assert REAL_FETCH(str(target)) == str(target)
    assert target.read_bytes() == payload
    assert not os.path.exists(str(target) + ".part")


def test_a_download_of_the_wrong_bytes_never_reaches_the_model_path(
        monkeypatch, tmp_path):
    payload = b"too short to be the model"
    target = tmp_path / "nanodet.onnx"
    monkeypatch.setattr(vm, "MODEL_BYTES", len(payload) + 1)
    monkeypatch.setattr(vm, "MODEL_SHA256", "0" * 64)
    monkeypatch.setattr(vm.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="hash check"):
        REAL_FETCH(str(target))

    assert not target.exists()
    assert not os.path.exists(str(target) + ".part")


def test_the_downloader_is_reachable_without_a_camera_or_a_model():
    assert callable(REAL_FETCH)
    assert vm.MODEL_URL.endswith(".onnx")
    assert vm.MODEL_BYTES > 0


# ---------------------------------------------------------------------------
# Failing gracefully, which is what keeps the bot's eyes open
# ---------------------------------------------------------------------------

@needs_numpy
def test_a_detector_that_cannot_get_weights_says_why_instead_of_raising(
        tmp_path):
    detector = vm.Detector(str(tmp_path / "nowhere" / "nanodet.onnx"))

    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []
    assert detector.error is not None
    assert not detector.available


@needs_numpy
def test_a_second_frame_does_not_retry_a_failed_fetch(tmp_path):
    """One timeout per RETRY_SECONDS, not one per frame."""
    detector = vm.Detector(str(tmp_path / "nowhere" / "nanodet.onnx"))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    detector.detect(frame)
    assert detector._retry_after > 0
    before = detector._retry_after
    detector.detect(frame)
    assert detector._retry_after == before  # sat out, did not go again


@needs_numpy
def test_a_detector_without_weights_never_leaves_the_frame_unheard(
        tmp_path, monkeypatch):
    """SeeingSystem must still produce an observation when there is no model."""
    from core.seeing import SeeingSystem

    seeing = SeeingSystem(str(tmp_path))
    observation = seeing.see_frame(np.zeros((64, 64, 3), dtype=np.uint8))

    assert isinstance(observation["analysis"]["it_objects"], list)
    assert seeing.detector.error is not None


# ---------------------------------------------------------------------------
# Decoding, which is the half that can be wrong in an interesting way
# ---------------------------------------------------------------------------

def _outputs(planted=None, grouped=False):
    """
    The six tensors a 416x416 forward returns: three scales, each with a
    classification head and a regression head.

    ``grouped`` picks the order the OpenCV 5 engine emits in (all the
    classifications first) over the classic engine's interleaved order.  The
    decoder must not care which one it is handed.
    """
    scales = []
    for length in (169, 676, 2704):
        cls = np.full((1, length, len(vm.CLASSES)), 0.01, dtype=np.float32)
        reg = np.zeros((1, length, (vm.REG_MAX + 1) * 4), dtype=np.float32)
        if planted is not None and planted[0] == length:
            _, index, label, score = planted
            cls[0, index, label] = score
        scales.append((cls, reg))

    if grouped:
        return [c for c, _ in scales] + [r for _, r in scales]
    return [item for pair in scales for item in pair]


# The middle of the stride-8 map: 52x52, so cell (26, 26) sits at the centre.
PLANTED = (2704, 26 * 52 + 26, 0, 0.9)   # scale, index, label ("person"), score


@needs_numpy
@needs_cv2
def test_the_decoded_box_lands_where_the_cell_actually_is():
    results = vm._post_process(_outputs(PLANTED), FRAME_SHAPE, FRAME_SCALE,
                               vm.MIN_CONFIDENCE)

    assert len(results) == 1
    hit = results[0]
    assert hit["label"] == "person"
    assert hit["confidence"] == pytest.approx(0.9, abs=1e-6)

    # An all-zero regression head decodes to 3.5 bins per side, i.e. 7 cells
    # across: 7 * stride 8 = 56 input pixels, scaled back into the frame.
    # The box comes back in the working image and is mapped out by the
    # letterbox's own new_w/new_h - not by INPUT_SIZE, because a frame padded
    # to a square does not keep its proportions in that direction.
    cell_centre = 26 * 8 + 0.5 * (8 - 1)
    scale_x = FRAME_SHAPE[1] / FRAME_SCALE[3]
    scale_y = FRAME_SHAPE[0] / FRAME_SCALE[2]
    expected_half = 7 * 8 / 2

    assert hit["box"][0] == pytest.approx(
        (cell_centre - expected_half) * scale_x, abs=0.5)
    assert hit["box"][1] == pytest.approx(
        (cell_centre - expected_half - FRAME_SCALE[0]) * scale_y, abs=0.5)
    assert hit["box"][2] - hit["box"][0] == pytest.approx(56 * scale_x, abs=0.5)
    assert hit["box"][3] - hit["box"][1] == pytest.approx(56 * scale_y, abs=0.5)


@needs_numpy
@needs_cv2
def test_the_decode_does_not_care_which_order_the_engine_reports_in():
    interleaved = vm._post_process(_outputs(PLANTED), FRAME_SHAPE,
                                   FRAME_SCALE, vm.MIN_CONFIDENCE)
    grouped = vm._post_process(_outputs(PLANTED, grouped=True), FRAME_SHAPE,
                               FRAME_SCALE, vm.MIN_CONFIDENCE)

    # This is the bug opencv_zoo's own nanodet.py still has: it pairs outputs
    # by position (preds[::2]) and dies under OpenCV 5.
    assert interleaved == grouped


@needs_numpy
@needs_cv2
def test_a_weak_score_is_not_reported_as_something_it_is_not():
    weak = (2704, 26 * 52 + 26, 0, 0.2)
    assert vm._post_process(_outputs(weak), FRAME_SHAPE, FRAME_SCALE,
                            vm.MIN_CONFIDENCE) == []


@needs_numpy
@needs_cv2
def test_a_garbled_network_output_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="unexpected output shape"):
        vm._post_process([np.zeros((1, 169, 100), dtype=np.float32)],
                         FRAME_SHAPE, FRAME_SCALE, vm.MIN_CONFIDENCE)


@needs_numpy
@needs_cv2
def test_letterboxing_survives_the_trip_back_out():
    frame = np.arange(FRAME_SHAPE[0] * FRAME_SHAPE[1] * 3,
                      dtype=np.uint8).reshape(FRAME_SHAPE[0], FRAME_SHAPE[1], 3)
    boxed, scale = vm._letterbox(frame)

    assert boxed.shape[:2] == (vm.INPUT_SIZE[0], vm.INPUT_SIZE[1])
    assert scale == FRAME_SCALE

    back = vm._unletterbox([0, 0, vm.INPUT_SIZE[1], vm.INPUT_SIZE[0]],
                           FRAME_SHAPE, scale)
    assert back[0] == pytest.approx(0, abs=1)
    assert back[1] == pytest.approx(0, abs=1)
    assert back[2] == pytest.approx(FRAME_SHAPE[1], abs=1)
    assert back[3] == pytest.approx(FRAME_SHAPE[0], abs=1)


# ---------------------------------------------------------------------------
# The real weights, when someone has them
# ---------------------------------------------------------------------------

@needs_numpy
@needs_cv2
@pytest.mark.skipif(not os.path.exists(vm.default_model_path()),
                    reason="the weights are not downloaded - "
                           "run python fetch_vision_model.py")
def test_the_pinned_weights_load_and_run_one_frame(monkeypatch):
    monkeypatch.setattr(vm, "fetch_model", REAL_FETCH)
    detector = vm.Detector()

    hits = detector.detect(np.full((480, 640, 3), 128, dtype=np.uint8))

    assert detector.error is None, detector.error
    assert detector.available
    assert hits == []          # a flat grey frame holds nothing to name
