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
from core.parameters import accessibility
from core.episodes import LongTermEpisodes
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

    assert len(p.consciousness) == 0
    assert len(p.recent) == 0
    assert p.total_dropped == 2
    assert p.total_recorded == 0


def test_important_events_are_memorized(tmp_path):
    p = new_pipeline(tmp_path)

    entry = p.record("seeing", {"objects": ["cup"]}, salience=0.7)

    assert entry is not None
    assert len(p.consciousness) == 1
    assert p.total_recorded == 1
    # nothing is stored that could go stale: the clocks are the memory
    assert entry["when_first"] <= entry["when_last_rehearsed"]
    assert accessibility(0.0) == 1.0


def test_repeats_reinforce_instead_of_duplicating(tmp_path):
    """Meeting the same thing twice must not create two memories."""
    p = new_pipeline(tmp_path)

    first = p.record("seeing", {"objects": ["cup"]}, salience=0.5, now=100.0)
    second = p.record("seeing", {"objects": ["cup"]}, salience=0.9, now=120.0)

    assert len(p.consciousness) == 1
    assert second is first
    assert second["hits"] == 2
    assert second["salience"] == 0.9
    # a repeat is a rehearsal: the clock moves forward, it does not decay
    assert second["when_last_rehearsed"] == 120.0


def test_fingerprint_is_stable_and_kind_scoped():
    a = _fingerprint("seeing", {"objects": ["cup"]})
    b = _fingerprint("seeing", {"objects": ["cup"]})
    c = _fingerprint("hearing", {"objects": ["cup"]})

    assert a == b
    assert a != c


def test_focus_overflow_sinks_instead_of_deleting(tmp_path):
    """OPEN-15 D: a full focus moves the stalest entry down, never drops it."""
    p = new_pipeline(tmp_path, conscious_capacity=3, window_s=30.0)

    # one entry per event, and events are 31 s apart so each is a new one
    for i in range(5):
        p.record("seeing", {"objects": [f"o{i}"]},
                 salience=0.5, now=1000.0 + i * 31.0)

    assert len(p.consciousness) == 3
    assert len(p.recent) == 2          # sunk, not deleted
    assert p.total_dropped == 0        # nothing was thrown away
    # the least recently rehearsed are the ones that moved down
    assert p.consciousness[-1]["payload"] == {"objects": ["o4"]}
    assert [e["payload"] for e in p.recent] == [
        {"objects": ["o0"]}, {"objects": ["o1"]}]


def test_soft_capacity_makes_crowding_age_faster(tmp_path):
    """OPEN-15 D: crowding changes the decay, not the membership."""
    assert accessibility(3600.0, load=0.0) > \
        accessibility(3600.0, load=3.0)
    # ...but a memory nobody has ever crowded still survives its 112 days
    assert accessibility(10 * 86400.0, load=0.0) >= 0.08


def test_accessibility_follows_the_published_fit(tmp_path):
    """OPEN-13/DATA-6: the curve is a citation, not a magic number."""
    # Murre & Dros (2015) Table 5, Ebbinghaus column, normalised to 1 at t=0
    assert accessibility(0.0) == 1.0
    assert accessibility(600) == pytest.approx(0.905, abs=0.002)     # 10 min
    assert accessibility(3600) == pytest.approx(0.627, abs=0.005)    # 1 h
    assert accessibility(86400) == pytest.approx(0.448, abs=0.005)   # 1 day

    # with the documented floor of 0.08 and nobody rehearsing: ~112 days
    assert accessibility(111 * 86400) > 0.08 > accessibility(114 * 86400)


# ---------------------------------------------------------------------------
# OPEN-16 (decided: G) - window gives the default cut, shock may cut early
# ---------------------------------------------------------------------------

def test_window_expires_and_the_next_observation_starts_a_new_event(tmp_path):
    p = new_pipeline(tmp_path, window_s=30.0)

    a = p.record("seeing", {"objects": ["cup"]}, salience=0.5, now=1000.0)
    same = p.record("seeing", {"objects": ["cup"]}, salience=0.5, now=1010.0)
    later = p.record("seeing", {"objects": ["cup"]}, salience=0.5, now=1040.0)

    assert same is a              # inside the window: one event
    assert later is not a         # window expired: new event
    assert len(p.consciousness) == 2


def test_content_shock_cuts_early(tmp_path):
    """G's exception: the situation replaced itself completely."""
    p = new_pipeline(tmp_path, window_s=30.0, shock_threshold=0.6)

    a = p.record("seeing", {"objects": ["cup", "plate"]},
                 salience=0.5, now=1000.0)
    b = p.record("seeing", {"objects": ["lamp", "chair"]},
                 salience=0.5, now=1005.0)

    assert b is not a and a["key"] != b["key"]


def test_a_shock_cannot_cut_twice_in_the_same_window(tmp_path):
    """Otherwise G degenerates back into E (a new entry per frame)."""
    p = new_pipeline(tmp_path, window_s=30.0, shock_threshold=0.6)

    first = p.record("seeing", {"objects": ["cup"]}, salience=0.5, now=1000.0)
    second = p.record("seeing", {"objects": ["zzz"]}, salience=0.5, now=1002.0)
    third = p.record("seeing", {"objects": ["qqq"]}, salience=0.5, now=1004.0)

    assert second["key"] != first["key"]    # the one allowed early cut
    assert third["key"] == second["key"]    # second shock in the slot: refused


def test_scalar_readings_are_not_a_situation_change(tmp_path):
    """volume_band is a degree, not a new event (Zacks 2010 dimensions)."""
    p = new_pipeline(tmp_path, window_s=30.0, shock_threshold=0.6)

    a = p.record("hearing", {"volume_band": 3}, salience=0.5, now=1000.0)
    b = p.record("hearing", {"volume_band": 17}, salience=0.5, now=1002.0)

    assert b is a


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
    """OPEN-14: nobody rehearsed it, the curve took it away."""
    p = new_pipeline(tmp_path, replay_salience=1.1)   # nothing is replayed
    p.record("hearing", {"volume_band": 3}, salience=0.5, now=1000.0)

    # 200 days later the MCM curve is far under the floor
    report = p.sleep(now=1000.0 + 200 * 86400)

    assert report["dropped_below_threshold"] == 1
    assert p.recent == []
    assert p.consciousness == []


def test_sleep_keeps_what_was_rehearsed(tmp_path):
    """The other half of OPEN-14: replay refreshes the clock, it survives."""
    p = new_pipeline(tmp_path, replay_salience=0.6)
    p.record("hearing", {"volume_band": 3}, salience=0.9, now=1000.0)

    report = p.sleep(now=1000.0 + 200 * 86400)

    assert report["replayed"] == 1
    assert report["dropped_below_threshold"] == 0


def test_sleep_transfers_replayed_and_old_enough_entries(tmp_path):
    """OPEN-15: sleep decides, and it is selective (replay + >= 1 day)."""
    p = new_pipeline(tmp_path, replay_salience=0.6, promote_min_age=86400.0)
    p.record("seeing", {"objects": ["cup"]}, salience=0.9, now=1000.0)

    report = p.sleep(now=1000.0 + 2 * 86400)

    assert report["promoted"] == ["seeing"]
    assert p.recent == []
    assert p.episodes.count() == 1


def test_wake_never_reaches_long_term(tmp_path):
    """OPEN-15 A (narrow): the wake period only writes the short layers."""
    p = new_pipeline(tmp_path)
    for i in range(6):
        p.record("seeing", {"objects": [f"o{i}"]},
                 salience=0.9, now=1000.0 + i * 40.0)

    assert p.episodes.count() == 0
    assert len(p.consciousness) + len(p.recent) == 6


def test_long_term_stores_only_the_five_episode_fields(tmp_path):
    """OPEN-17 / DATA-5: no times_seen, no strength, no salience."""
    p = new_pipeline(tmp_path)
    p.record("seeing", {"objects": ["cup"], "where": "desk"},
             salience=0.9, now=1000.0)
    p.episodes.promote(p.consciousness[0], now=5000.0)

    ep = p.episodes.find("seeing", {"objects": ["cup"], "where": "desk"})
    assert ep is not None
    assert set(ep) == {"when_first", "when_last_rehearsed",
                       "where", "what", "kind"}
    assert ep["where"] == "desk"
    assert ep["when_first"] == 1000.0


def test_long_term_refreshes_instead_of_duplicating(tmp_path):
    """Meeting it again buys it time rather than making a second copy."""
    store = LongTermEpisodes(str(tmp_path / "mem"), budget=500)
    entry = {"kind": "seeing", "payload": {"objects": ["cup"]},
             "when_first": 100.0}

    assert store.promote(entry, now=200.0) is True
    assert store.promote(entry, now=900.0) is True
    assert store.count() == 1
    assert store.find("seeing", {"objects": ["cup"]})["when_last_rehearsed"] == 900.0


def test_long_term_forgets_by_the_curve_and_respects_budget(tmp_path):
    store = LongTermEpisodes(str(tmp_path / "mem"), budget=3)
    for i in range(5):
        store.promote({"kind": "seeing", "payload": {"n": i},
                       "when_first": 100.0}, now=200.0 + i)

    # fresh: nothing is under the curve floor, but the budget is a ceiling
    assert store.forget_pass(now=300.0) == 2
    assert store.count() == 3
    # the least recently rehearsed were the ones dropped
    assert store.find("seeing", {"n": 0}) is None
    assert store.find("seeing", {"n": 4}) is not None


def test_long_term_forgets_only_when_sleep_runs(tmp_path):
    store = LongTermEpisodes(str(tmp_path / "mem"), budget=500)
    store.promote({"kind": "seeing", "payload": {"objects": ["cup"]},
                   "when_first": 0.0}, now=1000.0)

    # nothing happens by merely waiting: the pass is what forgets (OPEN-14)
    assert store.count() == 1
    assert store.forget_pass(now=1000.0 + 400 * 86400) == 1
    assert store.count() == 0


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


def test_a_rehearsed_entry_comes_back_into_focus(tmp_path):
    """Rehearsal during wake brings an entry up from `recent` (A, narrow)."""
    p = new_pipeline(tmp_path, conscious_capacity=1, window_s=30.0)

    seen = p.record("seeing", {"objects": ["cup"]}, salience=0.5, now=1000.0)
    p.record("hearing", {"volume_band": 3}, salience=0.5, now=1000.5)
    assert seen in p.recent             # the focus held only the sound

    again = p.record("seeing", {"objects": ["cup"]}, salience=0.5, now=1001.0)
    assert again is seen                # same event: reinforced, not copied
    assert seen in p.consciousness      # brought back up
    assert seen not in p.recent
    assert seen["when_last_rehearsed"] == 1001.0
    # ...and the sound was the one that sank to make room
    assert p.consciousness[0] is seen
    assert len(p.recent) == 1 and p.recent[0]["kind"] == "hearing"


def test_ledger_survives_a_restart(tmp_path):
    p = new_pipeline(tmp_path, window_s=30.0)
    for i in range(3):                      # 31 s apart -> 3 separate events
        p.record("seeing", {"n": i}, salience=0.5, now=1000.0 + i * 31.0)
    p.sleep(now=1100.0)

    again = new_pipeline(tmp_path)

    assert len(again.sleep_reports) == 1
    assert again.total_recorded == 3
    assert again.last_report()["reason"] == "scheduled"
    # the long term store is its own file and comes back too
    assert again.episodes.count() == 0


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
