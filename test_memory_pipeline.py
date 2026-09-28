"""
Tests for the memory pipeline: WHEN to memorize and WHEN to forget.

Design clauses: STEP-1 (self improving sleeping system), PIPE-2 (memory
pipeline controls when to memorize / forget), COG-3 (sleep sorts memories).

Run:
    python -m pytest test_memory_pipeline.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory_pipeline import (
    MemoryPipeline,
    _fingerprint,
    audio_salience,
    visual_salience,
)
from core.hearing import HearingSystem
from core.seeing import SeeingSystem
from core.human_like import HumanLikeSystem
from core.pure_learning import PureLearningSystem
from core.sensory import SensorySystem


@pytest.fixture(autouse=True)
def no_camera(monkeypatch):
    """These tests must never touch hardware."""
    monkeypatch.setattr(SeeingSystem, "_check_camera", lambda self: None)


def new_pipeline(tmp_path, **policy):
    return MemoryPipeline(data_dir=str(tmp_path / "mem"), policy=policy or None)


# ---------------------------------------------------------------------------
# When to memorize
# ---------------------------------------------------------------------------

def test_unimportant_events_are_never_memorized(tmp_path):
    """IO/PIPE-2: not everything that happens deserves to be remembered."""
    p = new_pipeline(tmp_path, min_salience=0.25)

    assert p.record("hearing", {"volume_band": 1}, salience=0.10) is None
    assert p.record("hearing", {"volume_band": 2}, salience=0.24) is None

    assert len(p.short_term) == 0
    assert p.total_dropped == 2
    assert p.total_recorded == 0


def test_important_events_are_memorized(tmp_path):
    p = new_pipeline(tmp_path)

    entry = p.record("seeing", {"objects": ["cup"]}, salience=0.7)

    assert entry is not None
    assert len(p.short_term) == 1
    assert p.total_recorded == 1
    assert 0.0 <= entry["strength"] <= 1.0


def test_repeats_reinforce_instead_of_duplicating(tmp_path):
    """Meeting the same thing twice must not create two memories."""
    p = new_pipeline(tmp_path)

    first = p.record("seeing", {"objects": ["cup"]}, salience=0.5)
    second = p.record("seeing", {"objects": ["cup"]}, salience=0.9)

    assert len(p.short_term) == 1
    assert second is first
    assert second["hits"] == 2
    assert second["strength"] > 0.5
    assert second["salience"] == 0.9


def test_fingerprint_is_stable_and_kind_scoped():
    a = _fingerprint("seeing", {"objects": ["cup"]})
    b = _fingerprint("seeing", {"objects": ["cup"]})
    c = _fingerprint("hearing", {"objects": ["cup"]})

    assert a == b
    assert a != c


def test_full_buffer_evicts_the_weakest(tmp_path):
    """A working memory with a budget forgets on the spot."""
    p = new_pipeline(tmp_path, short_term_capacity=3)

    for i in range(5):
        p.record("seeing", {"n": i}, salience=0.5)

    assert len(p.short_term) == 3
    assert p.total_dropped == 2
    # The oldest ones are the ones that went
    assert {"n": 0} not in [e["payload"] for e in p.short_term]
    assert {"n": 4} in [e["payload"] for e in p.short_term]


# ---------------------------------------------------------------------------
# When to sleep
# ---------------------------------------------------------------------------

def test_sleep_is_due_after_enough_events(tmp_path):
    p = new_pipeline(tmp_path, sleep_after_events=5, sleep_after_seconds=10 ** 9)

    for i in range(4):
        p.record("seeing", {"n": i}, salience=0.5)
    assert p.due() is False

    p.record("seeing", {"n": 4}, salience=0.5)
    assert p.due() is True


def test_sleep_is_due_after_being_awake_too_long(tmp_path):
    p = new_pipeline(tmp_path, sleep_after_seconds=10, sleep_after_events=10 ** 9)

    assert p.due(now=p.last_sleep_ts + 9) is False
    assert p.due(now=p.last_sleep_ts + 11) is True


def test_sleep_resets_the_rhythm_and_returns_immediately(tmp_path):
    """PIPE-4: sleep is one pass and never blocks the other pipelines."""
    p = new_pipeline(tmp_path, sleep_after_events=2)
    for i in range(3):
        p.record("seeing", {"n": i}, salience=0.5)

    report = p.sleep(now=1000.0, reason="test")

    assert p.events_since_sleep == 0
    assert p.last_sleep_ts == 1000.0
    assert p.due(now=1000.0) is False
    assert report["reason"] == "test"


# ---------------------------------------------------------------------------
# What sleep does
# ---------------------------------------------------------------------------

def test_sleep_forgets_what_decayed_below_threshold(tmp_path):
    p = new_pipeline(
        tmp_path,
        decay_per_sleep=0.3,
        forget_threshold=0.2,
        promote_threshold=0.9,
        replay_salience=1.1,   # nothing gets replayed
    )
    p.record("hearing", {"volume_band": 3}, salience=0.5)

    report = p.sleep(now=1000.0)

    assert report["dropped_below_threshold"] == 1
    assert p.short_term == []


def test_sleep_promotes_salient_memories_to_long_term(tmp_path):
    promoted = []

    p = new_pipeline(
        tmp_path,
        decay_per_sleep=1.0,
        replay_salience=0.6,
        replay_boost=0.5,
        promote_threshold=0.7,
    )
    p.register("hearing", promote=lambda e: promoted.append(e["key"]))
    p.record("hearing", {"volume_band": 9}, salience=0.9)

    report = p.sleep(now=1000.0)

    assert report["replayed"] == 1
    assert report["promoted"] == ["hearing"]
    assert len(promoted) == 1


def test_broken_store_cannot_ruin_the_whole_sleep(tmp_path):
    def boom():
        raise RuntimeError("store is broken")

    p = new_pipeline(tmp_path)
    p.register("broken", consolidate=boom)
    p.record("seeing", {"n": 1}, salience=0.5)

    report = p.sleep(now=1000.0)

    assert report["stores"]["broken"]["error"].startswith("RuntimeError")
    assert p.last_sleep_ts == 1000.0
    assert p.events_since_sleep == 0


def test_dedup_index_stays_within_budget(tmp_path):
    p = new_pipeline(tmp_path, seen_key_max=10, seen_key_retention=10 ** 9)

    for i in range(50):
        p.record("seeing", {"n": i}, salience=0.5)
    p.sleep(now=1000.0)

    assert len(p.seen_keys) <= 10


def test_ledger_survives_a_restart(tmp_path):
    p = new_pipeline(tmp_path)
    for i in range(3):
        p.record("seeing", {"n": i}, salience=0.5)
    p.sleep(now=1000.0)

    again = new_pipeline(tmp_path)

    assert len(again.sleep_reports) == 1
    assert again.total_recorded == 3
    assert again.last_report()["reason"] == "scheduled"


# ---------------------------------------------------------------------------
# Subsystem forgetting - these must really delete
# ---------------------------------------------------------------------------

def test_hearing_forget_pass_really_deletes(tmp_path):
    h = HearingSystem(str(tmp_path))
    h.auditory_memory = {
        "weak": {"strength": 0.01, "times_heard": 1},
        "strong": {"strength": 0.9, "times_heard": 20},
    }
    h.known_sounds = {
        "rare": {"times_heard": 1, "strength": 0.01},
        "common": {"times_heard": 5, "strength": 0.5},
    }
    h.words_recognized = [str(i) for i in range(1000)]

    report = h.forget_pass(max_words=100)

    assert "weak" not in h.auditory_memory
    assert "strong" in h.auditory_memory
    assert "rare" not in h.known_sounds
    assert "common" in h.known_sounds
    assert len(h.words_recognized) == 100
    assert report["known_sounds"] == 1

    # and it is written to disk, otherwise sleep was theatre
    reloaded = HearingSystem(str(tmp_path))
    assert "strong" in reloaded.auditory_memory


def test_seeing_forget_pass_really_deletes(tmp_path):
    s = SeeingSystem(str(tmp_path))
    s.known_objects = {
        "forgettable": {"times_seen": 1, "memory_strength": 0.01},
        "memorable": {"times_seen": 30, "memory_strength": 0.9},
    }
    s.things_seen = [{"n": i} for i in range(1000)]

    report = s.forget_pass(max_things=50)

    assert "forgettable" not in s.known_objects
    assert "memorable" in s.known_objects
    assert len(s.things_seen) == 50
    assert report["things_seen"] == 950

    reloaded = SeeingSystem(str(tmp_path))
    assert "memorable" in reloaded.known_objects


def test_human_like_promotes_working_memory_and_persists(tmp_path):
    h = HumanLikeSystem(str(tmp_path))
    h.store_interaction_memory("a loud sound", "startle")
    h.store_interaction_memory("a quiet room", "ignore")
    h.interaction_memories[0]["strength"] = 0.9   # worth keeping
    h.interaction_memories[1]["strength"] = 0.5

    report = h.forget_pass()

    assert report["promoted"] == 1
    assert len(h.long_term_memories) == 1
    assert len(h.interaction_memories) == 1

    # COG-2/COG-3: what sleep sorted must still be there after a restart
    reloaded = HumanLikeSystem(str(tmp_path))
    assert len(reloaded.long_term_memories) == 1
    assert len(reloaded.interaction_memories) == 1


def test_human_like_long_term_merges_repeats(tmp_path):
    h = HumanLikeSystem(str(tmp_path))
    h.store_interaction_memory("the same event", "x")
    h.interaction_memories[0]["strength"] = 0.9
    h.forget_pass()

    h.store_interaction_memory("the same event", "x")
    h.interaction_memories[0]["strength"] = 0.9
    h.forget_pass()

    assert len(h.long_term_memories) == 1
    assert h.long_term_memories[0]["repeat_count"] == 2


def test_pure_learning_forget_pass_prunes_patterns_and_concepts(tmp_path):
    ai = PureLearningSystem(str(tmp_path))
    ai.patterns = (
        [{"from": f"a{i}", "to": f"b{i}", "strength": 0.9} for i in range(10)]
        + [{"from": f"c{i}", "to": f"d{i}", "strength": 0.01} for i in range(40)]
    )
    for i in range(600):
        ai.concepts[f"word{i}"] = {"context": "ctx", "learned_at": "2026-01-01"}
        ai.word_frequency[f"word{i}"] = i

    report = ai.forget_pass(max_patterns=5, max_concepts=10)

    assert len(ai.patterns) == 5
    # the survivors are the strong ones
    assert all(p["strength"] > 0.5 for p in ai.patterns)
    assert len(ai.concepts) == 10
    # highest frequency concepts win
    assert "word599" in ai.concepts
    assert report["patterns"] == 45
    assert report["concepts"] == 590


def test_pure_learning_vocabulary_survives_pruning(tmp_path):
    """Deleting words would scramble the network's one-hot indices."""
    ai = PureLearningSystem(str(tmp_path))
    for w in ["alpha", "beta", "gamma"]:
        ai.learn_word(w)
    ai.patterns = []

    ai.forget_pass(max_patterns=0, max_concepts=0)

    assert ai.word_to_index["alpha"] == 4
    assert ai.vocabulary_size == 7


def test_sensory_history_is_capped(tmp_path):
    s = SensorySystem(str(tmp_path))
    s.sensory_history = [{"n": i} for i in range(1000)]
    s.sensory_load = 0.9

    s.forget_pass(max_history=100)

    assert len(s.sensory_history) == 100
    assert s.sensory_load == 0.0


# ---------------------------------------------------------------------------
# CAP-7: the host computes salience from raw observations
# ---------------------------------------------------------------------------

def test_a_frame_with_nothing_new_is_not_memorised(tmp_path):
    """The whole reason the gate exists: boring frames never reach memory."""
    p = new_pipeline(tmp_path)

    salience = visual_salience([])
    assert salience < p.policy["min_salience"]
    assert p.record("seeing", {"objects": []}, salience=salience) is None


def test_a_fresh_object_is_more_salient_than_a_familiar_one():
    known = {"cup": {"times_seen": 1}, "plate": {"times_seen": 50}}

    assert visual_salience(["cup"], known) > visual_salience(["plate"], known)
    assert visual_salience(["cup", "plate", "bowl"], known) > \
        visual_salience(["cup"], known)


def test_visual_salience_is_bounded():
    assert visual_salience([]) == 0.2
    assert 0.0 <= visual_salience(["x"]) <= 1.0
    assert visual_salience([f"o{i}" for i in range(100)]) == 1.0


def test_quiet_sound_is_not_memorised(tmp_path):
    p = new_pipeline(tmp_path)

    assert audio_salience(0.02) < p.policy["min_salience"]
    assert p.record("hearing", {"volume_band": 0},
                    salience=audio_salience(0.02)) is None


def test_loud_sound_is_memorised_and_louder_is_stronger(tmp_path):
    p = new_pipeline(tmp_path)

    quiet, loud = audio_salience(0.05), audio_salience(0.6)
    assert quiet < loud
    assert p.record("hearing", {"volume_band": 1}, salience=quiet) is None
    assert p.record("hearing", {"volume_band": 12}, salience=loud) is not None
    assert audio_salience(9.0) == 1.0
    assert audio_salience(-1.0) == 0.0
