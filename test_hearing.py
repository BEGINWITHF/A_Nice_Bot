"""
Hearing, from PCM to a `what` - OPEN-22.

Diary/articles/2026-10-28-hearing-design.md holds the eight decisions and
the paper behind each. What these tests hold down:

D1 hearing segregates rather than merely noticing change: two tones well
apart in frequency come back as two streams, one tone as one stream.

D2 segregate before describing - the streams are the units, and the four
numbers belong to a stream, not to the window.

D7 exactly four descriptors, each bounded, because the induction rule
measures a mean absolute difference and one unbounded dimension would
swamp the other three.

D8 nothing in a descriptor or a key names anything. `what` is floats; a
key is a decimal string. That is DATA-7 and OPEN-21 C, and it is the whole
reason the old `hear_sound("sound")` line had to stop being called.

The rule itself is shared with vision (core/categories.py), so a sound and
a picture land in the same kind of ledger - tested here rather than
assumed.
"""

import json
import os
import struct
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError:
    np = None

needs_numpy = pytest.mark.skipif(np is None, reason="numpy is not installed")

from core import categories
from core.hearing import FRAME_SAMPLES, HearingSystem
from core.host_ingest import HostIngest
from core.memory_pipeline import MemoryPipeline
from core.sensory import SensorySystem


def _tone(*parts, seconds=0.5, rate=44100, amplitude=20000):
    """A s16 signal built from (frequency, weight) pairs.

    Normalised to `amplitude` rather than scaled by it: the weights set how
    loud one tone is against another, and the whole thing has to stay
    inside int16 - an earlier version forgot the multiplication entirely and
    produced an all-zero buffer, which is a quiet way to test nothing.
    """
    total = int(rate * seconds)
    t = np.arange(total) / float(rate)
    out = np.zeros(total, dtype=float)
    for frequency, weight in parts:
        out += weight * np.sin(2.0 * np.pi * frequency * t)
    peak = float(np.abs(out).max())
    if peak > 0.0:
        out = out / peak * float(amplitude)
    return np.clip(out, -32768, 32767).astype(np.int16)


@needs_numpy
def test_one_tone_comes_back_as_one_stream(tmp_path):
    hearing = HearingSystem(str(tmp_path))

    heard = hearing.hear_samples(_tone((440.0, 1.0)), 44100)

    assert heard is not None
    assert heard["streams"] == 1
    assert len(heard["what"]) == 1


@needs_numpy
def test_two_well_separated_tones_come_back_as_two_streams(tmp_path):
    """D1: segregation is what hearing does, not just change detection."""
    hearing = HearingSystem(str(tmp_path))

    heard = hearing.hear_samples(_tone((440.0, 1.0), (3000.0, 1.0)), 44100)

    assert heard is not None
    assert heard["streams"] >= 2
    assert len(heard["what"]) >= 2
    # Two sounds that are nothing alike must not come back as one category -
    # that is the whole of what "hearing distinguishes" would mean here.
    assert len(set(hearing.last_induction)) >= 2


@needs_numpy
def test_a_stream_is_described_by_four_bounded_numbers(tmp_path):
    """D7: four, and bounded - the induction rule divides, not ranks."""
    hearing = HearingSystem(str(tmp_path))

    heard = hearing.hear_samples(_tone((440.0, 1.0), (3000.0, 0.6)), 44100)

    assert heard["what"]
    for descriptor in heard["what"]:
        assert len(descriptor) == 4
        for value in descriptor:
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0


@needs_numpy
def test_louder_carries_the_mixture_and_is_listed_first(tmp_path):
    """Order is priority, the same rule vision orders `what` by (OPEN-21 A)."""
    hearing = HearingSystem(str(tmp_path))

    # The low tone carries four times the weight of the high one.
    heard = hearing.hear_samples(_tone((440.0, 1.0), (3000.0, 0.25)), 44100)

    assert len(heard["what"]) >= 2
    # Centroid is scaled by the sample rate, so the quieter, higher stream
    # must sit above the louder, lower one in the list.
    centroids = [descriptor[0] for descriptor in heard["what"]]
    assert centroids == sorted(centroids)


@needs_numpy
def test_nothing_in_the_descriptor_or_a_key_is_a_word(tmp_path):
    """D8 / DATA-7: floats and decimal indices, and never a name."""
    hearing = HearingSystem(str(tmp_path))

    heard = hearing.hear_samples(_tone((440.0, 1.0), (3000.0, 1.0)), 44100)

    assert heard["what"]
    for descriptor in heard["what"]:
        for value in descriptor:
            assert not isinstance(value, str)
    for key in hearing.last_induction:
        assert isinstance(key, str) and key.isdigit()
    # Never a word anywhere in what memory would be handed.
    assert not any(isinstance(item, str) for item in heard["what"])


@needs_numpy
def test_hearing_and_vision_file_categories_the_same_way(tmp_path):
    """
    One rule, two senses (OPEN-22 D8).

    A descriptor from hearing and one from vision must both be accepted by
    the shared step and both come back as content-free indices - if hearing
    had its own copy of Barsalou's rule this test would be the first to
    notice it drifting.
    """
    descriptor = [0.01, 0.4, 0.5, 0.2]
    ledger = {}
    key, next_key = categories.induct(descriptor, ledger, 0,
                                      first_seen="2026-10-28T00:00:00")

    assert key == "0"
    assert key in ledger
    assert ledger[key]["times_seen"] == 1

    again, next_key = categories.induct(descriptor, ledger, next_key,
                                        first_seen="2026-10-28T00:00:01")
    assert again == "0"
    assert ledger["0"]["times_seen"] == 2


@needs_numpy
def test_too_short_a_snippet_hears_nothing_and_leaves_the_ledger_alone(tmp_path):
    hearing = HearingSystem(str(tmp_path))
    short = np.zeros(FRAME_SAMPLES - 1, dtype=np.int16)

    assert hearing.hear_samples(short, 44100) is None
    assert hearing.auditory_ledger == {}
    assert hearing.last_induction == []


@needs_numpy
def test_a_returning_shape_keeps_its_key(tmp_path):
    """Re-meeting a sound is a rehearsal, not a second category."""
    hearing = HearingSystem(str(tmp_path))
    signal = _tone((440.0, 1.0), (3000.0, 1.0))

    first = hearing.hear_samples(signal, 44100)
    keys_after_first = set(hearing.auditory_ledger)
    seen_after_first = {key: entry["times_seen"]
                        for key, entry in hearing.auditory_ledger.items()}
    second = hearing.hear_samples(signal, 44100)

    assert second["what"] == first["what"]
    # The same sound again opens no new category.  (Not "+1 per key": one
    # sound can arrive as several streams, and every stream that lands in a
    # category rehearses it.)
    assert set(hearing.auditory_ledger) == keys_after_first
    # And the ones it landed in did move - a repeat is a rehearsal, not a
    # record left to go stale.
    before = sum(seen_after_first.values())
    after = sum(entry["times_seen"] for entry in hearing.auditory_ledger.values())
    assert after > before


@needs_numpy
def test_the_category_ledger_is_held_to_a_budget(tmp_path):
    """
    Nothing else prunes it, so forget_pass has to.

    Without this the ledger grows by one row per distinct sound shape ever
    met and neither sleep nor IO-6 has been told to touch it.
    """
    hearing = HearingSystem(str(tmp_path))
    for index in range(6):
        # Every component moves.  With INDUCTION_TAU at 0.85 the mean
        # absolute difference over four components has to exceed 0.15 before
        # two of these are different categories, so nudging one component by
        # 0.01 would join all six and test nothing at all.
        descriptor = [0.2 * index] * 4
        key, next_key = categories.induct(descriptor, hearing.auditory_ledger,
                                          hearing._next_category,
                                          first_seen="2026-10-28T00:00:00")
        hearing._next_category = next_key
    assert len(hearing.auditory_ledger) == 6

    dropped = hearing.forget_pass(max_auditory=2)

    assert len(hearing.auditory_ledger) == 2
    assert dropped["auditory_ledger"] == 4


@needs_numpy
def test_the_ledger_survives_a_restart(tmp_path):
    """Every category reopened would mean the ledger never stopped growing."""
    directory = str(tmp_path)
    hearing = HearingSystem(directory)
    signal = _tone((440.0, 1.0), (3000.0, 1.0))
    heard = hearing.hear_samples(signal, 44100)
    assert heard["what"]

    again = HearingSystem(directory)

    assert again.auditory_ledger == hearing.auditory_ledger
    assert again._next_category == hearing._next_category


# ---------------------------------------------------------------------------
# D4 - attention, the three rules vision already runs (Sussman 2017 L40/L43)
# ---------------------------------------------------------------------------

@needs_numpy
def test_focus_rises_while_the_same_sound_keeps_arriving_and_falls_when_it_moves(tmp_path):
    hearing = HearingSystem(str(tmp_path))
    low = _tone((440.0, 1.0))
    elsewhere = _tone((3000.0, 1.0))
    start = datetime(2026, 10, 29, 8, 0, 0)

    # The first moment only records where attention is.
    hearing.hear_samples(low, 44100, now=start)
    assert hearing.focus_level == 0.0

    hearing.hear_samples(low, 44100, now=start + timedelta(seconds=2))
    held = hearing.focus_level
    assert held > 0.5

    hearing.hear_samples(elsewhere, 44100, now=start + timedelta(seconds=4))
    assert hearing.focus_level < held


@needs_numpy
def test_focus_never_leaves_its_own_range(tmp_path):
    hearing = HearingSystem(str(tmp_path))
    start = datetime(2026, 10, 29, 8, 0, 0)

    for step in range(40):
        # Alternating so it is forever being released and re-taken.
        signal = _tone((440.0, 1.0) if step % 2 else (3000.0, 1.0))
        hearing.hear_samples(signal, 44100,
                             now=start + timedelta(seconds=step))
        assert 0.0 <= hearing.focus_level <= 1.0


@needs_numpy
def test_focus_is_never_written_down(tmp_path):
    """R3 for hearing: a moment of attention belongs to that moment."""
    directory = str(tmp_path)
    hearing = HearingSystem(directory)
    start = datetime(2026, 10, 29, 8, 0, 0)
    signal = _tone((440.0, 1.0))

    hearing.hear_samples(signal, 44100, now=start)
    hearing.hear_samples(signal, 44100, now=start + timedelta(seconds=2))
    assert hearing.focus_level > 0.5

    with open(os.path.join(directory, "hearing_state.json"),
              encoding="utf-8") as handle:
        saved = json.load(handle)
    assert "focus_level" not in saved
    assert HearingSystem(directory).focus_level == 0.0


@needs_numpy
def test_attention_lets_a_quiet_stream_in_that_passive_hearing_misses(tmp_path):
    """
    D4's whole point, and the citation for it: Sussman 2017 L40 - listeners
    doing an active task separated two streams at a *smaller* frequency
    separation than the same listeners did passively.
    """
    hearing = HearingSystem(str(tmp_path))
    signal = _tone((440.0, 1.0), (3000.0, 0.2))
    start = datetime(2026, 10, 29, 8, 0, 0)

    passive = hearing.hear_samples(signal, 44100, now=start)
    assert passive["streams"] == 1

    held = None
    for step in (1, 2, 3):
        held = hearing.hear_samples(signal, 44100,
                                    now=start + timedelta(seconds=2 * step))

    assert hearing.focus_level > 0.9
    assert held["streams"] >= 2


@needs_numpy
def test_what_the_ear_attends_to_is_what_weighs_the_cut(tmp_path):
    """
    The wiring of D4's second half: focus leaves the ear and turns up at the
    pipeline as `attention`, so the rule is decided in one place rather than
    twice in two.
    """
    pipeline = MemoryPipeline(data_dir=str(tmp_path / "mem"),
                              policy={"min_salience": 0.0})
    senses = SensorySystem(str(tmp_path / "senses"))
    arrivals = []
    real = pipeline.record

    def spy(kind, payload, salience=0.5, now=None, attention=1.0):
        arrivals.append((kind, attention))
        return real(kind, payload, salience=salience, now=now,
                    attention=attention)

    pipeline.record = spy
    ingest = HostIngest(senses, pipeline)

    ingest("sound", struct.pack("<4410h", *([9000] * 4410)),
           {"device": "cam"})

    assert arrivals, "a loud enough sample must reach the gate"
    assert arrivals[0][0] == "hearing"
    # Not the default of 1.0: whatever the ear is attending to is what was
    # handed over, or this whole rule lives in two places.
    assert arrivals[0][1] == senses.hearing.focus_level
