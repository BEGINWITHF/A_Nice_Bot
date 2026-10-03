"""
Focus: R1 / R2 / R3, wired rather than merely present.

Diary/articles/2026-10-02-focus-and-fidelity.md section 5.  The author's
rule was "when the bot is focused enough on one thing it can remember a
very noisy picture", and three consequences were settled from the literature:

  R1  focus puts the whole of the resource on the one thing, so what it
      measures is the finest measurement of the frame (Bays 2009)
  R2  focus lowers the contrast threshold, so a weak region gets admitted
      instead of dying below the ordinary level (Reynolds & Heeger 2009)
  R3  focus acts on this act of encoding and never on a record already
      written (Myers 2014, marked # ASSUMPTION - the one counter-evidence
      measured a cue given *after* encoding and called its own design
      insensitive)

Before this, `focus_level` and `set_focus()` existed and nothing read them -
the same shape of gap as `OPEN-10`'s "the main loop never trains".
"""

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

needs_numpy = pytest.mark.skipif(np is None, reason="numpy is not installed")
needs_cv2 = pytest.mark.skipif(cv2 is None, reason="opencv is not installed")

from core.seeing import (
    SeeingSystem,
    _describe_region,
    _propose_regions,
    _saliency_map,
)


def _gabors(gray):
    """The four orientation channels, sliced per region inside the describer."""
    return [
        cv2.filter2D(
            gray, cv2.CV_8UC3,
            cv2.getGaborKernel((21, 21), 4.0, np.deg2rad(angle),
                               10.0, 0.5, 0, ktype=cv2.CV_32F),
        )
        for angle in (0, 45, 90, 135)
    ]


def _frame_with_peak(height, width, peak):
    """A saliency map whose only thing to look at is at `peak` = (x, y)."""
    saliency = np.zeros((height, width), dtype=np.uint8)
    saliency[peak[1], peak[0]] = 200
    return saliency


def _weak_region_frame():
    """
    A frame where attention is the only thing that can let a region in.

    The background is a fast sine: the 21x21 blur cannot follow it, so the
    frame carries plenty of salience, while no step anywhere reaches the
    Canny thresholds - so there is no contour source at all.  The blob is
    saturated (it is the seed by colour) but its own edge sits between the
    ordinary bar and the one attention lowers it to.

    Found by sweeping constructions rather than by picking numbers: a lone
    blob on flat grey can never demonstrate R2, because on flat grey the
    ordinary bar is near zero and the blob clears it either way.
    """
    height, width = 240, 320
    y, x = np.mgrid[0:height, 0:width]
    base = (128 + 120 * np.sin(2 * np.pi * x / 16.0)).clip(0, 255).astype(np.uint8)
    frame = np.dstack([base, base, base])
    frame[90:150, 130:190] = (33, 154, 33)
    return frame


# ---------------------------------------------------------------------------
# Focus itself: attention that stays put is attention that is held
# ---------------------------------------------------------------------------

def test_focus_rises_while_attention_holds_and_falls_when_it_moves(tmp_path):
    seeing = SeeingSystem(str(tmp_path))
    saliency = _frame_with_peak(240, 320, (100, 80))
    start = datetime(2026, 10, 3, 8, 0, 0)

    # Nothing to hold yet: the first frame only records where attention is.
    seeing._visual_attention(saliency, start)
    assert seeing.focus_level == 0.0
    assert seeing.attention_map["center"] == (100, 80)

    # Two seconds later, still on the same place: focus rises.
    seeing._visual_attention(saliency, start + timedelta(seconds=2))
    held = seeing.focus_level
    assert 0.5 < held < 1.0

    # Then it moves off, and focus releases.
    seeing._visual_attention(_frame_with_peak(240, 320, (300, 200)),
                             start + timedelta(seconds=4))
    assert seeing.focus_level < held


def test_focus_never_leaves_its_own_range(tmp_path):
    seeing = SeeingSystem(str(tmp_path))
    saliency = _frame_with_peak(240, 320, (10, 10))
    start = datetime(2026, 10, 3, 8, 0, 0)

    for step in range(40):
        seeing._visual_attention(saliency, start + timedelta(seconds=step))
        assert 0.0 <= seeing.focus_level <= 1.0


def test_focus_is_recomputed_rather_than_kept(tmp_path):
    """R3's first half: focus belongs to one act of encoding, not to state."""
    seeing = SeeingSystem(str(tmp_path))
    seeing._visual_attention(_frame_with_peak(64, 64, (8, 8)),
                             datetime(2026, 10, 3, 8, 0, 0))
    seeing._save_state()

    with open(os.path.join(str(tmp_path), "seeing_state.json"),
              encoding="utf-8") as handle:
        saved = json.load(handle)
    assert "focus_level" not in saved
    assert SeeingSystem(str(tmp_path)).focus_level == 0.0


def test_focus_cannot_be_set_by_hand(tmp_path):
    """OPEN-9: a setter with no caller and no counterpart in a person goes."""
    seeing = SeeingSystem(str(tmp_path))
    assert not hasattr(seeing, "set_focus")


# ---------------------------------------------------------------------------
# R2 - attention's lowered contrast threshold is what lets a weak region in
# ---------------------------------------------------------------------------

@needs_cv2
def test_r2_a_weak_region_waits_for_attention(tmp_path):
    frame = _weak_region_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saliency = _saliency_map(gray)

    assert _propose_regions(gray, hsv, saliency, 0.0) == []

    found = _propose_regions(gray, hsv, saliency, 1.0)
    assert len(found) == 1
    assert found[0][0] == (130, 90, 60, 60, 3600)

    # Half way is still not enough: R2 is a threshold moving, not a switch.
    assert _propose_regions(gray, hsv, saliency, 0.25) == []


@needs_cv2
def test_r2_only_ever_adds_regions_and_never_takes_one_away(tmp_path):
    """Lowering a threshold is monotone; anything else would be a bug."""
    rng = np.random.default_rng(7)
    for frame in (_weak_region_frame(),
                  np.full((240, 320, 3), 64, dtype=np.uint8),
                  rng.integers(0, 255, (240, 320, 3)).astype(np.uint8)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saliency = _saliency_map(gray)

        dull = [entry[0] for entry in _propose_regions(gray, hsv, saliency, 0.0)]
        sharp = [entry[0] for entry in _propose_regions(gray, hsv, saliency, 1.0)]
        assert set(dull) <= set(sharp)
        assert len(sharp) >= len(dull)


# ---------------------------------------------------------------------------
# R1 - the measurement the focus is on leans on the part carrying the signal
# ---------------------------------------------------------------------------

def _one_blob_frame():
    """A single saturated blob, so there is exactly one region to describe."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[30:90, 40:120] = (0, 0, 200)
    return frame


@needs_cv2
def test_r1_an_unfocused_region_is_measured_exactly_as_before(tmp_path):
    """precision 0 must reproduce the old numbers, or nothing is comparable."""
    frame = _one_blob_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saliency = _saliency_map(gray)
    gabors = _gabors(gray)

    box, mask = _propose_regions(gray, hsv, saliency, 0.0)[0]
    plain = _describe_region(gray, hsv, saliency, mask, box, gabors)
    explicit = _describe_region(gray, hsv, saliency, mask, box, gabors, 0.0)

    assert plain is not None
    assert plain[1] == explicit[1]
    assert len(plain[1]) == 19


@needs_cv2
def test_r1_focusing_changes_what_the_attended_region_measures(tmp_path):
    frame = _one_blob_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saliency = _saliency_map(gray)
    gabors = _gabors(gray)

    box, mask = _propose_regions(gray, hsv, saliency, 0.0)[0]
    plain = _describe_region(gray, hsv, saliency, mask, box, gabors, 0.0)
    focused = _describe_region(gray, hsv, saliency, mask, box, gabors, 1.0)

    assert focused is not None
    # Still nineteen numbers: focus changes how finely a region is measured,
    # never the shape of what `what` is (OPEN-21 A).
    assert len(focused[1]) == 19
    assert focused[1] != plain[1]
    # No weight sum focus can empty: every value stays a usable measurement.
    assert all(np.isfinite(value) and value >= 0.0 for value in focused[1])


# ---------------------------------------------------------------------------
# R3 - focus acts on this encoding and on nothing already stored
# ---------------------------------------------------------------------------

def test_r3_focus_never_reaches_back_into_a_record_already_written(tmp_path):
    seed = [0.2] * 19
    instance = [0.4] * 19

    distracted = SeeingSystem(str(tmp_path / "distracted"))
    intent = SeeingSystem(str(tmp_path / "intent"))
    assert distracted._induct(seed) == intent._induct(seed)

    distracted.focus_level = 0.0
    intent.focus_level = 1.0
    assert distracted._induct(instance) == intent._induct(instance)

    # The same update either way: attention changes the measurement that is
    # about to be written, never the amount by which a stored one moves.
    assert (distracted.known_objects["0"]["simulation"]
            == intent.known_objects["0"]["simulation"])
    assert (distracted.known_objects["0"]["memory_strength"]
            == intent.known_objects["0"]["memory_strength"])


# ---------------------------------------------------------------------------
# Attention runs before the measuring, not after it
# ---------------------------------------------------------------------------

@needs_cv2
def test_attention_is_computed_before_the_frame_is_described(tmp_path):
    seeing = SeeingSystem(str(tmp_path))

    analysis = seeing.see_frame(_one_blob_frame(), source="camera:test")["analysis"]

    # It used to run after IT had already described everything, so there was
    # nothing left for it to guide.
    assert seeing.attention_map is not None
    assert analysis["attention"] is seeing.attention_map
    assert 0.0 <= seeing.focus_level <= 1.0
    # One map, one computation: attention and proposal read the same numbers
    # rather than two copies that merely resemble each other.
    assert isinstance(analysis["what"], list)
    assert all(len(descriptor) == 19 for descriptor in analysis["what"])
