"""
The input half of STEP-2: perception reaching learning (OPEN-10).

OPEN-10 found that the main loop only slept - `learn_pattern` had no caller
anywhere in real running, so nothing grew. What to feed it was parked there
until perception was rebuilt; it was rebuilt on 2026-10-02/03, and this is
the wiring.

Two rulings are nailed down here.

An observation is learned from only when it was memorised. The pipeline's
salience gate already decides what is worth writing down, so a frame too
dull to remember is too dull to learn from: `learn_from` takes whatever
`MemoryPipeline.record` returned and does nothing at all with None. There
is no second threshold anywhere in the learning path.

What is learned is `last_induction`, never `what`. `what` is the flat list
of descriptors - that is content - while an induction key is a content-free
index into the ledger (OPEN-21 C, Quest L10: an index holds no specific
information and is agnostic to its contents). So a learned pattern reads
"this thing, then that thing" with a name for neither, which is the whole
of what DATA-7 asked for.

The sequence itself - the order - is the priority order an open-attention
frame was listed in (OPEN-21 A). Human statistical learning runs on
adjacent items and their transitional probabilities (Saffran, Aslin &
Newport 1996; Fiser & Aslin 2002; Turk-Browne, Jungé & Scholl 2005), and
`learn_pattern` learns adjacent pairs, so the shape matches.
"""

import os
import sys
from datetime import datetime

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

needs_cv2 = pytest.mark.skipif(cv2 is None, reason="opencv is not installed")

from core.host_ingest import HostIngest
from core.memory_pipeline import MemoryPipeline
from core.pure_learning import PureLearningSystem
from core.sensory import SensorySystem


def _brain(tmp_path):
    return PureLearningSystem(str(tmp_path / "pure"))


def _entry():
    """Something that stands in for a record the pipeline accepted."""
    return {"key": "k", "kind": "seeing", "payload": {}, "salience": 0.9}


# ---------------------------------------------------------------------------
# The gate: memorised means learned from, dropped means nothing at all
# ---------------------------------------------------------------------------

def test_a_memorised_observation_is_learned_from(tmp_path):
    brain = _brain(tmp_path)

    brain.learn_from(_entry(), ["3", "7", "1"])

    assert [(p["from"], p["to"]) for p in brain.patterns] == [("3", "7"), ("7", "1")]


def test_a_dropped_observation_teaches_nothing(tmp_path):
    """The pipeline says None; that is the whole of the decision."""
    brain = _brain(tmp_path)

    brain.learn_from(None, ["3", "7", "1"])

    assert brain.patterns == []


# ---------------------------------------------------------------------------
# The sequence: adjacent pairs, made of keys
# ---------------------------------------------------------------------------

def test_only_adjacent_pairs_are_learned(tmp_path):
    brain = _brain(tmp_path)

    brain.learn_from(_entry(), ["a", "b", "c", "d"])

    assert [(p["from"], p["to"]) for p in brain.patterns] == [
        ("a", "b"), ("b", "c"), ("c", "d")]


def test_the_items_are_ledger_keys_and_never_descriptors(tmp_path):
    """
    A key is a content-free index - a short decimal string, nothing in it.

    This is the guard against the mistake OPEN-21 C exists to prevent:
    putting content, or a name for content, into what the bot remembers.
    """
    brain = _brain(tmp_path)
    keys = ["0", "3", "12"]

    brain.learn_from(_entry(), keys)

    for pattern in brain.patterns:
        for side in ("from", "to"):
            item = pattern[side]
            assert isinstance(item, str)
            assert item.isdigit()
            assert item in keys


def test_meeting_a_pair_again_rehearses_instead_of_duplicating(tmp_path):
    """A repeat is a rehearsal - it moves the clock, it does not add a row."""
    brain = _brain(tmp_path)

    brain.learn_from(_entry(), ["3", "7"])
    before = brain.patterns[0]["learned_at"]
    brain.learn_from(_entry(), ["3", "7"])

    assert len(brain.patterns) == 1
    assert brain.patterns[0]["learned_at"] >= before


# ---------------------------------------------------------------------------
# Both eyes share one rule
# ---------------------------------------------------------------------------

def _frame():
    """Something with an object in it, so induction has a region to file."""
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[20:70, 20:70] = (0, 255, 0)
    return frame


@needs_cv2
def test_a_device_frame_reaches_the_learner_the_same_way_a_local_one_does(tmp_path):
    pipeline = MemoryPipeline(data_dir=str(tmp_path / "mem"),
                              policy={"min_salience": 0.0})
    senses = SensorySystem(str(tmp_path / "senses"))
    learned = []

    ingest = HostIngest(
        senses, pipeline,
        learn=lambda entry, keys: learned.append((entry is not None,
                                                  list(keys))),
    )

    ok, buffer = cv2.imencode(".jpg", _frame())
    assert ok
    ingest("vision", buffer.tobytes(), {"device": "cam"})

    assert learned, "a frame the pipeline memorised must reach the learner"
    recorded, keys = learned[0]
    assert recorded is True
    assert keys == senses.seeing.last_induction


@needs_cv2
def test_a_host_ingest_with_no_learner_left_in_place_still_works(tmp_path):
    """`learn` defaults to None, so the server does not have to have one."""
    pipeline = MemoryPipeline(data_dir=str(tmp_path / "mem"),
                              policy={"min_salience": 0.0})
    senses = SensorySystem(str(tmp_path / "senses"))
    ingest = HostIngest(senses, pipeline)

    ok, buffer = cv2.imencode(".jpg", _frame())
    assert ok
    answer = ingest("vision", buffer.tobytes(), {"device": "cam"})

    assert answer["channel"] == "vision"
    assert answer["recorded"] is True
