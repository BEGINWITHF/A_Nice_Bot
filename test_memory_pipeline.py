"""
Tests for the memory pipeline: WHEN to memorize and WHEN to forget.

Design clauses: STEP-1 (self improving sleeping system), PIPE-2 (memory
pipeline controls when to memorize / forget), COG-3 (sleep sorts memories).

Run:
    python -m pytest test_memory_pipeline.py -v
"""

import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory_pipeline import (
    MemoryPipeline,
    _fingerprint,
    audio_salience,
    visual_salience,
)
from core.parameters import accessibility
from core import parameters as P
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

# ---------------------------------------------------------------------------
# When to sleep - OPEN-12: the two-process model, not a schedule
# ---------------------------------------------------------------------------

def test_pressure_accumulates_slowly_and_recovers_fast(tmp_path):
    """The asymmetry the whole idea rests on: chi_w = 18.2 h, chi_s = 4.2 h."""
    import math
    # one time constant: 63.2% of the way there, in both directions
    woke = P.advance_pressure(P.H_SLEEP, P.TAU_WAKE_S, asleep=False)
    assert woke == pytest.approx(1 - (1 - P.H_SLEEP) * math.exp(-1))
    slept = P.advance_pressure(0.67, P.TAU_SLEEP_S, asleep=True)
    assert slept == pytest.approx(0.67 * math.exp(-1))
    # recovery is ~4.3x faster than accumulation -> one night restores most
    assert P.TAU_WAKE_S / P.TAU_SLEEP_S == pytest.approx(4.3, abs=0.05)
    # a backwards clock changes nothing (and does not blow up)
    assert P.advance_pressure(0.5, -100, asleep=False) == 0.5


def test_thresholds_follow_the_circadian_rhythm(tmp_path):
    """H+(t) = H0+ + a*C(t), H-(t) = H0- + a*C(t) - Skeldon 2025 eq. 7/8."""
    trough = P.thresholds(P.CIRCADIAN_MIN_PHASE_S)
    peak = P.thresholds(P.CIRCADIAN_MIN_PHASE_S + 12 * 3600)
    assert P.circadian_signal(P.CIRCADIAN_MIN_PHASE_S) == -1.0
    assert P.circadian_signal(P.CIRCADIAN_MIN_PHASE_S + 12 * 3600) == 1.0
    assert trough[0] == pytest.approx(P.H_WAKE - P.CIRCADIAN_AMPLITUDE)
    assert peak[0] == pytest.approx(P.H_WAKE + P.CIRCADIAN_AMPLITUDE)
    # the gap never closes: that is what makes "startled awake" possible
    assert all(up > lo for up, lo in (trough, peak))


def test_it_falls_asleep_when_pressure_passes_the_upper_threshold(tmp_path):
    """No time-of-day schedule anywhere - only pressure against a threshold."""
    p = new_pipeline(tmp_path)
    now = 1000.0
    p.asleep = False
    p.pressure_ts = now

    p.pressure = P.H_SLEEP            # 0.17: under every upper threshold
    assert p.due(now) is False

    p.pressure = 1.0                  # 1.0: over every upper threshold
    assert p.due(now) is True


def test_it_wakes_when_pressure_reaches_the_lower_threshold(tmp_path):
    p = new_pipeline(tmp_path)
    now = 1000.0
    p.asleep = True
    p.pressure_ts = now
    p.last_sleep_ts = now             # no sleep cycle has elapsed yet

    p.pressure = 1.0                  # still far above every lower threshold
    assert p.due(now) is False

    p.pressure = 0.0                  # under every lower threshold
    assert p.due(now) is True


def test_a_sleep_period_sorts_once_per_cycle(tmp_path):
    """OPEN-18a: the author chose one pass per 90-minute sleep cycle."""
    p = new_pipeline(tmp_path, sleep_cycle=90 * 60)
    now = 1000.0
    p.asleep = True
    p.pressure = 0.5
    p.pressure_ts = now
    p.last_sleep_ts = now

    assert p.due(now + 89 * 60) is False    # not yet a full cycle
    assert p.due(now + 91 * 60) is True     # next cycle is due


def test_sleep_pass_reports_which_transition_it_took(tmp_path):
    """Onset, then a mid-sleep cycle, then waking - three passes, one night."""
    p = new_pipeline(tmp_path, sleep_cycle=90 * 60)
    now = 1000.0
    p.asleep = False
    p.pressure = 1.0
    p.pressure_ts = now

    onset = p.sleep(now=now, reason="scheduled")
    assert onset["transition"] == "fell_asleep"
    assert onset["state"] == "asleep"
    assert p.asleep is True

    mid = p.sleep(now=now + 91 * 60, reason="scheduled")
    assert mid["transition"] == "cycle"
    assert p.asleep is True              # a cycle pass is still asleep

    p.pressure = 0.0                     # pressure spent itself overnight
    wake = p.sleep(now=now + 180 * 60, reason="scheduled")
    assert wake["transition"] == "woke"
    assert wake["state"] == "awake"
    assert p.asleep is False


def _simulate_rhythm(days=14, dt=60.0, start_pressure=None, asleep=False, t0=0.0):
    """Integrate Process S against Process C for `days`, returning the
    steady-state transitions.  Uses only the shipped constants."""
    pressure = P.H_SLEEP if start_pressure is None else start_pressure
    asleep, t = asleep, t0
    switches, awake_s, sleep_s = [], 0.0, 0.0
    for _ in range(int(days * 24 * 3600 / dt)):
        t += dt
        upper, lower = P.thresholds(t)
        pressure = P.advance_pressure(pressure, dt, asleep)
        if not asleep and pressure >= upper:
            asleep = True
            switches.append(("SLEEP", t))
        elif asleep and pressure <= lower:
            asleep = False
            switches.append(("WAKE", t))
        if asleep:
            sleep_s += dt
        else:
            awake_s += dt
    steady = [s for s in switches if s[1] > 3 * 24 * 3600]   # drop the first days
    onsets = [t2 for kind, t2 in steady if kind == "SLEEP"]
    offsets = [t2 for kind, t2 in steady if kind == "WAKE"]
    pairs = []
    for i in range(len(steady) - 1):
        kind, t1 = steady[i]
        nxt, t2 = steady[i + 1]
        if (kind, nxt) == ("SLEEP", "WAKE"):
            pairs.append(("sleep", (t2 - t1) / 3600))
        elif (kind, nxt) == ("WAKE", "SLEEP"):
            pairs.append(("wake", (t2 - t1) / 3600))
    mean = {k: sum(d for kk, d in pairs if kk == k) /
             max(1, sum(1 for kk, _ in pairs if kk == k)) for k in ("sleep", "wake")}
    return onsets, offsets, mean


def test_the_rhythm_entrains_to_one_24_hour_day(tmp_path):
    """
    OPEN-12's payoff: 16 h awake / 8 h asleep, one cycle a day.

    Nothing in this test sets a schedule - it integrates Process S against
    Process C using only the shipped constants.  parameter-values section 2
    predicts T_wake = 16.8 h and T_sleep = 5.8 h from the free-running model;
    with a = 0.12 the circadian forcing entrains that 22.6 h oscillator to
    exactly 24 h (Skeldon 2025, "Entrainment of the sleep-wake oscillator").
    If a swap of TAU_*/H_*/CIRCADIAN_* breaks the day, this says so.
    """
    onsets, offsets, mean = _simulate_rhythm()

    assert len(onsets) == len(offsets)
    assert 9 <= len(onsets) <= 13              # monophasic: one a day

    assert mean["sleep"] == pytest.approx(7.9, abs=0.6)
    assert mean["wake"] == pytest.approx(16.1, abs=0.6)
    assert mean["sleep"] + mean["wake"] == pytest.approx(24.0, abs=0.2)
    # phase-locked, not drifting: every night starts within the same 10 min
    # (compared modulo one day, since they are 24 h apart by construction)
    onset_clock = [t2 % 86400 for t2 in onsets]
    offset_clock = [t2 % 86400 for t2 in offsets]
    assert max(onset_clock) - min(onset_clock) < 600.0
    assert max(offset_clock) - min(offset_clock) < 600.0
    # documented consequence of CIRCADIAN_MIN_PHASE = 06:00: it sleeps
    # 01:27-09:24.  The numbers are not asserted, the *stability* above is -
    # changing the chronotype moves them, it does not break the rhythm.


def test_the_chronotype_is_the_only_thing_that_moves_the_schedule(tmp_path,
                                                                   monkeypatch):
    """CIRCADIAN_MIN_PHASE_S is the swap point (OPEN-13): moving the trough
    3 h earlier moves the whole day 3 h earlier, unchanged in length."""
    onsets1, offsets1, mean1 = _simulate_rhythm()
    monkeypatch.setattr(P, "CIRCADIAN_MIN_PHASE_S", 3 * 3600)
    onsets2, offsets2, mean2 = _simulate_rhythm()

    shift = (onsets2[0] - onsets1[0]) % 86400
    assert shift == pytest.approx(86400 - 3 * 3600, abs=600)   # 3 h earlier
    shift_off = (offsets2[0] - offsets1[0]) % 86400
    assert shift_off == pytest.approx(86400 - 3 * 3600, abs=600)
    # the length of the day is a property of TAU_*/H_*, not of the chronotype
    assert mean2["sleep"] == pytest.approx(mean1["sleep"], abs=0.2)
    assert mean2["wake"] == pytest.approx(mean1["wake"], abs=0.2)


def test_a_sleep_pass_is_one_pass_and_never_blocks(tmp_path):
    """PIPE-4: sleep() runs once and returns; the senses keep running."""
    p = new_pipeline(tmp_path)
    p.record("seeing", {"n": 0}, salience=0.5, now=1000.0)

    report = p.sleep(now=1000.0, reason="test")

    assert p.events_since_sleep == 0
    assert p.last_sleep_ts == 1000.0
    assert report["reason"] == "test"
    assert p.due(now=1000.0) is False


# ---------------------------------------------------------------------------
# OPEN-18b: asleep, the senses still run but the world is not memorised
# ---------------------------------------------------------------------------

def test_while_asleep_routine_observations_are_not_memorised(tmp_path):
    """PIPE-4 keeps the senses on; OPEN-18b keeps the memories off
    ("常规事情感知不到")."""
    p = new_pipeline(tmp_path)
    p.asleep = True
    p.pressure = 0.5
    p.pressure_ts = 21600.0

    assert p.record("hearing", {"volume_band": 3},
                    salience=0.4, now=21610.0) is None
    assert p.record("seeing", {"objects": ["cup"]},
                    salience=0.7, now=21611.0) is None   # passes the gate, not this one

    assert p.total_recorded == 0
    assert p.total_dropped_asleep == 2
    assert p.asleep is True            # ordinary noise does not wake it


def test_a_severe_observation_startles_the_bot_awake(tmp_path):
    """OPEN-18b: "受到剧烈情况惊醒" - and it is remembered, because it is
    the reason the bot is awake."""
    # 06:00 is the circadian trough: the upper threshold sits at its lowest
    # (0.55), so a pressure of 0.6 would send it straight back to sleep.
    p = new_pipeline(tmp_path)          # shipped AROUSAL_S, not a test value
    p.asleep = True
    p.pressure = 0.6
    p.pressure_ts = 21600.0
    p.last_sleep_ts = 21500.0

    entry = p.record("seeing", {"objects": ["intruder"]},
                     salience=0.95, now=21600.0)

    assert entry is not None
    assert p.asleep is False
    assert p.startle_count == 1
    assert p.aroused_until == 21600.0 + P.AROUSAL_S      # 10 min, author's call

    # Skeldon's "wake effort": pressure still says sleep, the arousal holds it
    assert p.due(21630.0) is False
    # ...and once that expires the model decides again
    assert p.due(21600.0 + P.AROUSAL_S + 1.0) is True


def test_startle_wake_effort_can_expire(tmp_path):
    p = new_pipeline(tmp_path, arousal_s=0.0)
    p.asleep = True
    p.pressure = 1.0
    p.pressure_ts = 21600.0

    p.record("seeing", {"objects": ["fire"]}, salience=0.95, now=21600.0)
    assert p.asleep is False
    # no wake effort -> the two-process model puts it back to sleep at once
    assert p.due(21601.0) is True


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


def test_human_like_keeps_state_but_holds_no_memories(tmp_path):
    """OPEN-19 (author 2026-09-30: delete the memory subsystem as a whole)."""
    h = HumanLikeSystem(str(tmp_path))

    # the four members of a memory subsystem are gone, not just empty
    assert not hasattr(h, "interaction_memories")
    assert not hasattr(h, "long_term_memories")
    assert not hasattr(h, "forget_pass")
    assert not hasattr(h, "sleep_like_consolidation")

    # ...while state a brain does have still works and still persists
    h.process_input("hello there", "world")
    reloaded = HumanLikeSystem(str(tmp_path))
    assert reloaded.personality == h.personality
    assert reloaded.emotions["happy"].intensity == h.emotions["happy"].intensity
    assert reloaded.social_context["conversation_count"] == 1


def test_human_state_file_carries_no_parameter_fields(tmp_path):
    """DATA-5: after a save, no `strength` / `importance` may be on disk."""
    h = HumanLikeSystem(str(tmp_path))
    h.process_input("hi", "world")

    path = os.path.join(str(tmp_path), "human_state.json")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    assert "interaction_memories" not in text
    assert "long_term_memories" not in text
    assert "emotional_impact" not in text
    assert '"replay_count"' not in text


def test_pure_learning_forgets_by_the_curve_not_by_a_score(tmp_path):
    """DATA-5 + OPEN-19: no `strength` anywhere; MCM(now - learned_at) decides."""
    ai = PureLearningSystem(str(tmp_path))
    now = datetime.now()
    fresh = now.isoformat()
    long_ago = datetime.fromtimestamp(now.timestamp() - 400 * 86400).isoformat()

    ai.patterns = (
        [{"from": f"a{i}", "to": f"b{i}", "learned_at": fresh} for i in range(10)]
        + [{"from": f"c{i}", "to": f"d{i}", "learned_at": long_ago} for i in range(40)]
    )
    for i in range(600):
        ai.concepts[f"word{i}"] = {"context": "ctx", "learned_at": fresh}
        ai.word_frequency[f"word{i}"] = i
    ai.concepts["ancient"] = {"context": "ctx", "learned_at": long_ago}

    report = ai.forget_pass(max_patterns=5, max_concepts=10)

    #40 patterns never rehearsed in 400 days are past the floor (112 days)
    assert all("strength" not in p for p in ai.patterns)
    assert len(ai.patterns) == 5
    assert all(p["learned_at"] == fresh for p in ai.patterns)

    # the curve deletes the ancient concept before any ranking gets to speak
    assert "ancient" not in ai.concepts
    assert len(ai.concepts) == 10
    assert "word599" in ai.concepts          # frequency still ranks survivors

    assert report["patterns"] == 45
    assert report["concepts"] == 591


def test_rehearsing_a_pattern_only_moves_its_clock(tmp_path):
    """Repetition is a rehearsal, not a score the entry carries around."""
    ai = PureLearningSystem(str(tmp_path))

    ai.learn_pattern(["the", "cat", "sat"])
    assert set(ai.patterns[0]) == {"from", "to", "learned_at"}

    before = ai.patterns[0]["learned_at"]
    ai.learn_pattern(["the", "cat"])
    assert len(ai.patterns) == 2
    assert set(ai.patterns[1]) == {"from", "to", "learned_at"}

    # meeting the same pair again must not duplicate or score it
    ai.learn_pattern(["the", "cat"])
    assert len(ai.patterns) == 2
    assert ai.patterns[1]["learned_at"] >= before


def test_legacy_patterns_lose_their_strength_on_load(tmp_path):
    """A file written before OPEN-19 must migrate, not keep the field."""
    ai = PureLearningSystem(str(tmp_path))
    state_file = os.path.join(str(tmp_path), "learning_state.json")
    with open(state_file, "w", encoding="utf-8") as fh:
        json.dump({
            "word_to_index": {"<UNK>": 0}, "index_to_word": {"0": "<UNK>"},
            "word_frequency": {}, "vocabulary_size": 4,
            "patterns": [{"from": "a", "to": "b", "strength": 7,
                          "learned_at": datetime.now().isoformat()}],
            "concepts": {},
        }, fh)

    reloaded = PureLearningSystem(str(tmp_path))

    assert set(reloaded.patterns[0]) == {"from", "to", "learned_at"}


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
