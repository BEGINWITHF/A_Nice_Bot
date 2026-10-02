"""
Memory Pipeline - decides WHEN to memorize and WHEN to forget.

Design source: Diary/articles/2026-09-29-design-passage.md
  "another pipeline for controlling the memory system, this can control when to
   memorize and when to forget: this might be a non AI part as even human beings
   can not control when to memorize and what to forget"
  "For sleep ... a perfect opportunity for it to manage and sort all the
   informations. (Of course, keeping the 'vertebral' pipeline running is
   extremely important ...)"

This module is deliberately a **deterministic, non-AI policy**: thresholds,
budgets and a rhythm. No neural network takes part in it.
It does not understand anything - it only allocates attention.

Three layers (decided in OPEN-15, author's answer "就这样做了"):

  consciousness   the focus, CONSCIOUS_CAPACITY entries.  When it overflows
                  the stalest entry *sinks* into `recent` - nothing is
                  deleted here (D: soft capacity, no hard eviction).
  recent          the cache below the focus, soft capacity: crowding it only
                  makes everything in it age faster (parameters.accessibility
                  load term), it never kicks an entry out on its own.
  episodes        long term, core/episodes.py - reached only during sleep and
                  only selectively (OPEN-17).

The rhythm (OPEN-12): there is no schedule anywhere.  Process S (sleep
pressure) rises with tau_w = 18.2 h while awake and falls with tau_s = 4.2 h
while asleep; it crosses a pair of thresholds that Process C (circadian,
24 h, amplitude 0.12) moves up and down.  Crossing upward is falling asleep,
crossing downward is waking - and the two together settle into exactly one
16 h awake / 8 h asleep day (verified against the shipped constants in the
test suite, not asserted by hand).

While asleep the senses keep running (PIPE-4) but routine observations are
not memorised; a severe one startles the bot awake instead (OPEN-18).

Two invariants kept from the original:
  1. The sensing/behaviour loops are never blocked by sleep. sleep() runs a
     single pass and returns immediately - a sleeping person still reacts.
     One sleep period runs several of them: at onset, once per 90-minute
     sleep cycle, and at waking (OPEN-18a).
  2. Forgetting really deletes. Below the threshold the entry is removed,
     instead of being multiplied by 0.9 forever.

Everything numeric lives in core/parameters.py (OPEN-13), so a whole source
can be swapped in one place.
"""

import hashlib
import json
import math
import os
import time
from datetime import datetime

from core import parameters as P
from core.episodes import LongTermEpisodes

# ---------------------------------------------------------------------------
# Policy - thin aliases over core/parameters.py, kept so that tests and
# reports can still override single values (OPEN-13: one place to swap a
# source; this dict is that place's view, not a second copy of the numbers).
# ---------------------------------------------------------------------------
DEFAULT_POLICY = {
    # --- when to memorize ---
    "min_salience": 0.25,               # below this: never enters the focus
    "conscious_capacity": P.CONSCIOUS_CAPACITY,   # focus size (Cowan, OPEN-14)
    "recent_capacity": P.RECENT_CAPACITY,         # soft, # ASSUMPTION (65)

    # --- when to sleep: OPEN-12 two-process model, no schedule anywhere ---
    # (the old sleep_after_seconds = 900 / sleep_after_events = 200 are gone:
    #  neither has a human counterpart - parameter-values section 2)
    "sleep_cycle": P.SLEEP_CYCLE_S,       # one sort per sleep cycle (OPEN-18a)
    "startle_salience": P.STARTLE_SALIENCE,  # wakes it: OPEN-18b
    "arousal_s": P.AROUSAL_S,             # how long it stays alertly awake

    # --- what sleep does ---
    "replay_salience": P.REPLAY_SALIENCE,   # salient enough to be replayed
    "forget_threshold": P.FORGET_THRESHOLD, # below this: really deleted
    "promote_min_age": P.PROMOTE_MIN_AGE_S, # Cepeda 2006: >= 1 day before
                                            # sleep may transfer it
    "max_sleep_reports": P.MAX_SLEEP_REPORTS,

    # --- event segmentation (OPEN-16, decided: G) ---
    "window_s": P.WINDOW_S,             # default cut on timeout
    # OPEN-21 point B: an early cut needs a prediction error running above
    # the event's own baseline, not a distance from where the event began.
    "event_model_tau_s": P.EVENT_MODEL_TAU_S,
    "shock_ratio": P.SHOCK_RATIO,
    "shock_floor": P.SHOCK_FLOOR,
}


# ---------------------------------------------------------------------------
# Observation -> salience.
# Design clause CAP-7: devices only report raw observations, the host decides
# what is worth keeping. These two functions are that decision, kept here so
# the policy lives with the rest of the pipeline and can be tested without a
# camera or a microphone.
# ---------------------------------------------------------------------------

def visual_salience(objects, known_objects=None):
    """
    Salience of one visual observation, 0..1.

    `objects` are the ledger keys the observation touched - internal
    indices, never names (OPEN-21 C) - and `known_objects` is the ledger
    itself.  A key sitting at one sighting is something the bot has met
    exactly once, which is the whole of what "new" means here: the gate is
    not comparing content with anything, it is asking how barely known the
    corner of the ledger this landed in still is.

    A frame that touches nothing scores below
    DEFAULT_POLICY["min_salience"] and is therefore never memorised.
    """
    known_objects = known_objects or {}
    objects = objects or []
    fresh = sum(
        1 for name in objects
        if known_objects.get(name, {}).get("times_seen", 0) == 1
    )
    return max(0.0, min(1.0, 0.2 + 0.2 * fresh + 0.05 * len(objects)))


def audio_salience(volume):
    """Salience of one audio observation, 0..1: room tone is not a memory."""
    return max(0.0, min(1.0, float(volume) * 3.0))


def _fingerprint(kind, payload):
    """Stable fingerprint of one observation's content."""
    if isinstance(payload, str):
        body = payload.strip().lower()
    else:
        body = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(f"{kind}|{body}".encode("utf-8", "replace")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# OPEN-16 (decided: G) - event segmentation
# ---------------------------------------------------------------------------
# G = "the window gives the default cut, a content shock may cut early".
#
# The old rule was E: the fingerprint of the content *was* the event key, so
# every change of the object list opened a new entry - the bottleneck's
# amplifier (see Diary/articles/2026-09-29-consolidation-bottleneck.md).
#
# Why G and not F (pure window): Gomez et al. (2025) showed duration alone
# predicts everyday event endings at 4-5% accuracy, and Guler et al. (2026)
# showed working memory is reset *by* boundaries rather than generating them
# on a timer - so a pure clock has no principled cut point.  Why not E: Shim
# et al. (2022) showed boundaries appear even for fully predictable changes,
# so "content changed" is not the rule either.  Evidence: Diary/articles/
# 2026-09-29-consolidation-evidence.md sections 4, 4.1, 4.2.
#
# OPEN-21 point B (2026-10-02) changed *how* the exception is measured, not
# that there is one.  The early cut is now a prediction error running above
# the running event model's own baseline - see _payload_vector,
# _prediction_error and _integrate below, and core/parameters.py for
# EVENT_MODEL_TAU_S, SHOCK_RATIO and SHOCK_FLOOR.  What was removed is the
# absolute 0.6 and the comparison against the frame the event started on:
# neither has a counterpart in the way people segment events (Kurby & Zacks
# 2008:162-164), and the frozen first frame was a bug in our model of them
# rather than a cost we had agreed to pay.

def _is_number(value):
    """A real quantity, not a boolean pretending to be one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _payload_vector(payload):
    """
    The numeric situation of one observation: the input an event model is
    asked to predict.

    Set-valued fields only, and only the numeric ones.  A scalar reading is
    deliberately left out - volume_band is a degree, not a new event, and
    Zacks et al. (2010) coded the situational dimensions as character,
    spatial, goal and object changes rather than as a loudness number.  A
    word is left out too: there is nothing to compare in it, and DATA-7 has
    no natural language in memory anyway.

    A list of regions is pooled into one vector of the same length on every
    frame, because the event model summarises what the situation *is* - it
    does not keep every frame's contents beside it.
    """
    if not isinstance(payload, dict):
        return []

    vector = []
    for key in sorted(payload):
        value = payload[key]
        rows = None
        if isinstance(value, (list, tuple)) and value:
            if all(_is_number(v) for v in value):
                rows = [[float(v) for v in value]]
            elif all(isinstance(v, (list, tuple)) and v
                     and all(_is_number(x) for x in v) for v in value):
                if len({len(v) for v in value}) == 1:
                    rows = [[float(x) for x in v] for v in value]
        if rows:
            vector.extend(
                sum(column) / float(len(rows)) for column in zip(*rows)
            )
    return vector


def _prediction_error(model, situation):
    """
    How far the input sits from the model that was supposed to predict it.

    Graded, and graded per dimension rather than as a distance from a
    remembered starting point: Kurby & Zacks (2008:163) speak of prediction
    errors that *transiently increase relative to their current baseline*,
    and the model half of that comparison keeps moving (L162, *integrating
    information over the recent past*) - it is not the first frame, frozen
    when the event opened and compared against forever.

    Returns 0.0 when the two cannot be laid against each other: no model
    yet, or a situation shaped differently.  Nothing to compare is not a
    shock.
    """
    if not model or not situation or len(model) != len(situation):
        return 0.0
    return sum(abs(a - b) for a, b in zip(model, situation)) / float(len(model))


def _integrate(state, situation, error, now, window, tau):
    """
    Fold one observation into the running event model, and let its error
    into the baseline.

    Kurby & Zacks (2008:162): event models are *maintained in a stable state
    to guide prediction, integrating information over the recent past*.  A
    leaky average is the smallest thing that does both halves of that - it
    catches up with a slow drift instead of holding the event's first frame
    up for comparison, and it refuses to be redefined by one noisy input,
    which is the same source's report that such models are *robust to
    moment-to-moment fluctuations in the perceptual input* (L174).

    The baseline is a leaky average of these very errors taken over the
    length of the event itself, so "their current baseline" means this
    event's own usual amount of being wrong rather than a number picked in
    advance.
    """
    if not situation:
        return
    last = float(state.get("model_at", state.get("started", now)))
    dt = max(now - last, 0.0)
    state["model_at"] = now

    current = state.get("model")
    if not current or len(current) != len(situation):
        # First frame of the event, or a situation of a new shape: the model
        # starts from the input in front of it - Kurby & Zacks 2008:176,
        # *reset based on the current sensory and perceptual information
        # available*, not carried over from the event before it.
        state["model"] = list(situation)
    else:
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-9))
        state["model"] = [old + alpha * (new - old)
                          for old, new in zip(current, situation)]

    if window > 0:
        alpha_baseline = 1.0 - math.exp(-dt / window)
        state["baseline"] = (
            (1.0 - alpha_baseline) * float(state.get("baseline", 0.0))
            + alpha_baseline * float(error)
        )


def _is_representation(value):
    """
    A perceptual representation rather than a set of things.

    `what` is a flat run of numbers or a list of equal-length runs: a graded
    description of a situation, with no members to enumerate.
    """
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if all(_is_number(v) for v in value):
        return True
    return (all(isinstance(v, (list, tuple)) and v
                and all(_is_number(x) for x in v) for v in value)
            and len({len(v) for v in value}) == 1)


def _merge_payload(old, new):
    """
    Fold one observation into the entry it belongs to.

    Lists of things are unioned, because the situation accumulates what was
    there.  A perceptual representation and a scalar keep the first value,
    because the entry stands for how the event started: two frames of one
    event are not two objects sitting side by side, they are two
    measurements of the same situation, and there is nothing to union in
    them.

    # ASSUMPTION: no literature specifies how to merge two observations
                  inside one event.  Union-of-sets was the least lossy rule
                  that did not grow without bound while a payload was a set
                  of tokens; a graded representation is no longer a set, and
                  every frame differs from the last by some fraction of a
                  decimal, so unioning it would append for the whole length
                  of the window.  Keeping the first is what stays bounded.
    """
    if old == new:
        return old
    if not isinstance(old, dict) or not isinstance(new, dict):
        return new
    merged = dict(old)
    for key, value in new.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(merged[key], (list, tuple)) and isinstance(value, (list, tuple)):
            if _is_representation(merged[key]) and _is_representation(value):
                continue                      # see docstring: nothing to union
            combined = list(merged[key])
            for item in value:
                if item not in combined:
                    combined.append(item)
            merged[key] = combined
        # scalars: keep the first one - see docstring
    return merged


class MemoryPipeline:
    """
    Memory pipeline: gate -> focus -> recent -> sleep -> long term / delete.
    """

    def __init__(self, data_dir="data/memory", policy=None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.policy = dict(DEFAULT_POLICY)
        if policy:
            self.policy.update(policy)

        # Three layers (OPEN-15 B)
        self.consciousness = []   # focus: [{key, kind, payload, salience, ...}]
        self.recent = []          # soft-capacity cache below the focus
        self.episodes = LongTermEpisodes(
            data_dir,
            budget=P.LONG_TERM_BUDGET,
        )

        # Segmentation state (OPEN-16 G), per kind of observation
        self.segments = {}        # kind -> {key, seq, started, model, baseline, shock_slot}
        self.segment_seq = 0      # monotonic, so keys stay unique across restarts

        # Rhythm state - OPEN-12 two-process model.  Nothing here is a
        # schedule: `asleep` flips when Process S crosses a circadian-modulated
        # threshold, and `pressure` is integrated with the exact exponential.
        self.last_sleep_ts = time.time()   # last consolidation pass
        self.state_since = self.last_sleep_ts   # when the current state began
        self.asleep = False
        self.pressure = P.H_SLEEP          # start as if just woken
        self.pressure_ts = time.time()
        self.aroused_until = 0.0           # OPEN-18b: wake effort after a startle
        self.events_since_sleep = 0
        self.total_recorded = 0
        self.total_dropped = 0
        self.total_dropped_asleep = 0      # OPEN-18b: sensed but not memorised
        self.startle_count = 0             # OPEN-18b: severe events that woke it

        # Registered long term subsystems: name -> {"consolidate", "forget"}
        # (OPEN-17: no `promote` hook any more - the pipeline owns `episodes`.)
        self.stores = {}
        self.sleep_reports = []

        self._load()

    # ------------------------------------------------------------------
    # Register long term stores
    # ------------------------------------------------------------------
    def register(self, name, consolidate=None, forget=None):
        """
        Attach a subsystem. forget() MUST really delete weak memories.

        There is deliberately no `promote` parameter: promotion goes to the
        pipeline's own long term episode store (OPEN-17).
        """
        self.stores[name] = {
            "consolidate": consolidate,
            "forget": forget,
        }

    # ------------------------------------------------------------------
    # When to memorize
    # ------------------------------------------------------------------
    def record(self, kind, payload, salience=0.5, now=None):
        """
        Try to encode one sensed event.

        Returns the entry holding it (in the focus, or in `recent` if that
        is where it lives), or None when the policy rejected it.
        """
        now = time.time() if now is None else now
        salience = max(0.0, min(1.0, float(salience)))

        # OPEN-18b: while asleep the senses keep running (PIPE-4) but routine
        # observations are not memorised - "常规事情感知不到".  A severe one
        # startles the bot awake instead of being quietly noted down.
        if self.asleep:
            if salience < self.policy["startle_salience"]:
                self.total_dropped_asleep += 1
                return None
            self._startle(now)

        # Gate 1: unimportant things are not memorized
        if salience < self.policy["min_salience"]:
            self.total_dropped += 1
            return None

        key = self._segment(kind, payload, now)

        # Gate 2: still inside the same event -> reinforce, never duplicate
        entry = self._find(key)
        if entry is not None:
            entry["hits"] += 1
            entry["salience"] = max(entry["salience"], salience)
            entry["payload"] = _merge_payload(entry["payload"], payload)
            self._rehearse(entry, now)      # a repeat is a rehearsal
            self.events_since_sleep += 1
            return entry

        entry = {
            "key": key,
            "kind": kind,
            "payload": payload,
            "salience": salience,
            "when_first": now,
            "when_last_rehearsed": now,
            "hits": 1,
        }
        self._to_focus(entry)
        self.total_recorded += 1
        self.events_since_sleep += 1
        return entry

    # -- segmentation (OPEN-16 G, OPEN-21 point B) --------------------
    def _segment(self, kind, payload, now):
        """
        Decide which event this observation belongs to, and return its key.

        Default cut: the window expired (OPEN-16 primary rule).
        Early cut:   a prediction error running above this event's own
                     baseline (OPEN-21 point B), allowed at most once per
                     window slot so that a burst of changes cannot degrade
                     back into option E (one entry per frame).
        """
        state = self.segments.get(kind)
        situation = _payload_vector(payload)
        window = float(self.policy["window_s"])
        slot = int(now // window) if window > 0 else 0

        if state is None:
            return self._open_segment(kind, payload, now, slot)

        expired = (now - state["started"]) >= window

        # Read the error *before* the model is allowed to absorb the input:
        # that ordering is the whole of "prediction error".  A slow drift
        # never gets here, because the model follows it, and frame-to-frame
        # noise never gets here, because it sits in the baseline as much as
        # it sits in the error.  What does get here is an abrupt jump.
        error = _prediction_error(state.get("model"), situation)
        raised = (
            error >= float(self.policy["shock_floor"])
            and error >= float(state.get("baseline", 0.0))
                          * float(self.policy["shock_ratio"])
        )
        early = raised and slot != state.get("shock_slot")

        _integrate(state, situation, error, now, window,
                   float(self.policy["event_model_tau_s"]))

        if expired or early:
            return self._open_segment(kind, payload, now, slot,
                                      shock_slot=slot if early else None)
        return state["key"]

    def _open_segment(self, kind, payload, now, slot, shock_slot=None):
        self.segment_seq += 1
        key = _fingerprint(kind, {"event": self.segment_seq})
        self.segments[kind] = {
            "key": key,
            "seq": self.segment_seq,
            "started": now,
            # The event model starts from the frame that opened the event
            # and its baseline starts at nothing - Kurby & Zacks 2008:176,
            # the model is *reset based on the current sensory and
            # perceptual information available*, not carried over from the
            # event before it.  A list, not a tuple: this dict is written
            # to disk.
            "model": _payload_vector(payload) or None,
            "model_at": now,
            "baseline": 0.0,
            "shock_slot": shock_slot,
        }
        return key

    # -- the two short term layers ------------------------------------
    def _find(self, key):
        """An entry lives in the focus or in `recent`, never both."""
        for entry in self.consciousness:
            if entry["key"] == key:
                return entry
        for entry in self.recent:
            if entry["key"] == key:
                return entry
        return None

    def _rehearse(self, entry, now):
        """A repeat refreshes the clock - and brings it back into focus."""
        entry["when_last_rehearsed"] = now
        for i, held in enumerate(self.recent):
            if held is entry:                 # identity, not dict equality
                self.recent.pop(i)
                self._to_focus(entry)
                return

    def _to_focus(self, entry):
        """
        Put an entry in the focus.  If the focus is full, the least recently
        rehearsed one *sinks* into `recent` - OPEN-15 D: nothing is deleted
        just because a list hit its size (the old `total_dropped` eviction).
        """
        if len(self.consciousness) >= int(self.policy["conscious_capacity"]):
            stalest = min(self.consciousness,
                          key=lambda e: e["when_last_rehearsed"])
            self.consciousness.remove(stalest)
            self.recent.append(stalest)
        self.consciousness.append(entry)

    # ------------------------------------------------------------------
    # When to sleep - OPEN-12: two-process model, not a schedule
    # ------------------------------------------------------------------
    def _advance(self, now):
        """Integrate Process S up to `now` - exact exponential, never Euler.

        A clock that moved backwards (`dt <= 0`) is re-anchored without
        touching the pressure; there is nothing to integrate into the past.
        """
        self.pressure = P.advance_pressure(
            self.pressure, now - self.pressure_ts, self.asleep)
        self.pressure_ts = now

    def _startle(self, now):
        """OPEN-18b: a severe observation wakes the bot ("剧烈情况惊醒").

        Skeldon 2025 names what keeps you awake in exactly this situation
        **wake effort**: the pressure is still above the lower threshold, so
        without `aroused_until` the very next check would decide "sleep" and
        put the bot straight back down.
        """
        if not self.asleep:
            return
        self._advance(now)      # the decay up to here belongs to that sleep
        self.asleep = False
        self.state_since = now
        self.aroused_until = now + float(self.policy["arousal_s"])
        self.startle_count += 1

    def transition(self, now=None):
        """
        The state change due right now, or None.

        Returns "fell_asleep", "woke", "cycle" (a mid-sleep pass, OPEN-18a)
        or None.  It only moves the state machine - consolidation belongs to
        sleep(), which is the one that calls this.
        """
        now = time.time() if now is None else now
        self._advance(now)
        upper, lower = P.thresholds(now)

        if self.asleep:
            if self.pressure <= lower:                     # sleep -> wake
                return "woke"
            if now - self.last_sleep_ts >= float(self.policy["sleep_cycle"]):
                return "cycle"                             # next sleep cycle
            return None

        # awake -> sleep, suppressed while the wake effort of a startle holds
        if now >= self.aroused_until and self.pressure >= upper:
            return "fell_asleep"
        return None

    def due(self, now=None):
        """True when one consolidation pass should run right now."""
        return self.transition(now) is not None

    def state(self, now=None):
        """'asleep' or 'awake', advancing the rhythm first."""
        now = time.time() if now is None else now
        self._advance(now)
        return "asleep" if self.asleep else "awake"

    def awake_seconds(self, now=None):
        """How long it has been in the *current* state, not since the pass."""
        now = time.time() if now is None else now
        return max(0.0, now - self.state_since)

    def sleep(self, now=None, reason="scheduled"):
        """
        Run one sleep pass.

        Key constraint: this is a **single non-blocking pass**. The spine and
        the senses keep running throughout - a sleeping person still reacts
        (PIPE-4).  One sleep period runs several of these: at onset, once per
        90-minute sleep cycle, and at waking (OPEN-18a).

        Phases (OPEN-15):
          1. flush the focus into `recent` - the focus is offline while asleep
          2. replay: salient entries get their clock refreshed
          3. forgetting: everything below the curve's floor is deleted
          4. transfer: what was replayed AND is >= 1 day old goes long term
          5. every registered subsystem consolidates and prunes itself
          6. the long term episode store forgets too (OPEN-14)
          7. bookkeeping
        """
        now = time.time() if now is None else now
        p = self.policy

        # --- phase 0: the rhythm decides what just happened ----------------
        transition = self.transition(now)
        if transition == "fell_asleep":
            self.asleep = True
            self.state_since = now
            self.aroused_until = 0.0
        elif transition == "woke":
            self.asleep = False
            self.state_since = now
            self.aroused_until = 0.0

        report = {
            "reason": reason,
            "started": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "awake_seconds": round(now - self.last_sleep_ts, 1),
            "state_seconds": round(now - self.state_since, 1),
            "state": "asleep" if self.asleep else "awake",
            "transition": transition,
            "pressure": round(self.pressure, 3),
            "events_processed": self.events_since_sleep,
            "focus_in": len(self.consciousness),
            "recent_in": len(self.recent),
            "promoted": [],
            "dropped_below_threshold": 0,
            "replayed": 0,
            "stores": {},
        }

        # --- phase 1: flush the focus -------------------------------------
        while self.consciousness:
            self.recent.append(self.consciousness.pop(0))

        # --- phase 2: replay refreshes the clock of salient entries --------
        # (this is what buys them time on the curve; it replaces the old
        #  `replay_boost` strength patch, which stored a parameter instead)
        replayed = []
        for entry in self.recent:
            if entry["salience"] >= p["replay_salience"]:
                entry["when_last_rehearsed"] = now
                replayed.append(entry)
                report["replayed"] += 1

        # --- phase 3: the curve decides who stays --------------------------
        # load = how crowded `recent` is: a crowded cache makes everything in
        # it effectively older (soft capacity), it does not evict anyone.
        load = len(self.recent) / float(max(1, int(p["recent_capacity"])))
        survivors = []
        for entry in self.recent:
            elapsed = max(0.0, now - entry["when_last_rehearsed"])
            if P.accessibility(elapsed, load) < float(p["forget_threshold"]):
                report["dropped_below_threshold"] += 1
                continue
            survivors.append(entry)
        self.recent = survivors

        # --- phase 4: selective transfer to long term ----------------------
        # Criterion (OPEN-15): sleep replayed it AND it is at least a day old.
        # `promote_threshold` is gone - it was a strength number, and the
        # decision is now "did sleep actually spend a replay on it".
        replayed_ids = {id(e) for e in replayed}
        still = []
        for entry in self.recent:
            age = now - entry.get("when_first", now)
            if id(entry) in replayed_ids and age >= float(p["promote_min_age"]):
                if self._promote(entry, now):
                    report["promoted"].append(entry["kind"])
                    continue        # transferred: it lives long term now
            still.append(entry)
        self.recent = still

        report["recent_out"] = len(self.recent)

        # --- phase 5: every registered subsystem consolidates and prunes ---
        for name, hooks in self.stores.items():
            store_report = {}
            try:
                if hooks["consolidate"]:
                    store_report["consolidated"] = hooks["consolidate"]()
                if hooks["forget"]:
                    store_report["forgotten"] = hooks["forget"]()
            except Exception as exc:      # one broken store must not ruin the pass
                store_report["error"] = f"{type(exc).__name__}: {exc}"
            report["stores"][name] = store_report

        # --- phase 6: long term forgets by the same curve (OPEN-14) --------
        report["episodes_forgotten"] = self.episodes.forget_pass(now=now)
        report["episodes"] = self.episodes.count()

        # --- phase 7: bookkeeping ------------------------------------------
        self.last_sleep_ts = now
        self.events_since_sleep = 0
        self.sleep_reports.append(report)
        self.sleep_reports = self.sleep_reports[-int(p["max_sleep_reports"]):]
        self._save()

        return report

    def _promote(self, entry, now=None):
        """Hand one entry to the long term episode store (OPEN-17)."""
        try:
            return bool(self.episodes.promote(entry, now=now))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def stats(self):
        now = time.time()
        self._advance(now)
        upper, lower = P.thresholds(now)
        return {
            "consciousness": len(self.consciousness),
            "conscious_capacity": self.policy["conscious_capacity"],
            "recent": len(self.recent),
            "recent_capacity": self.policy["recent_capacity"],
            "episodes": self.episodes.count(),
            "events_since_sleep": self.events_since_sleep,
            "awake_seconds": round(self.awake_seconds(now), 1),
            "state": "asleep" if self.asleep else "awake",
            "pressure": round(self.pressure, 3),
            "thresholds": [round(upper, 3), round(lower, 3)],
            "total_recorded": self.total_recorded,
            "total_dropped": self.total_dropped,
            "total_dropped_asleep": self.total_dropped_asleep,
            "startle_count": self.startle_count,
            "sleep_count": len(self.sleep_reports),
            "registered_stores": sorted(self.stores),
            "last_sleep": self.sleep_reports[-1]["started"] if self.sleep_reports else None,
        }

    def last_report(self):
        return self.sleep_reports[-1] if self.sleep_reports else None

    # ------------------------------------------------------------------
    # Persistence - only the pipeline's own ledger lives here; subsystem data
    # is written by the subsystems themselves.
    # ------------------------------------------------------------------
    def _state_path(self):
        return os.path.join(self.data_dir, "pipeline_state.json")

    def _load(self):
        path = self._state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.last_sleep_ts = state.get("last_sleep_ts", self.last_sleep_ts)
            self.total_recorded = state.get("total_recorded", 0)
            self.total_dropped = state.get("total_dropped", 0)
            self.total_dropped_asleep = state.get("total_dropped_asleep", 0)
            self.startle_count = state.get("startle_count", 0)
            self.events_since_sleep = state.get("events_since_sleep", 0)
            self.sleep_reports = state.get("sleep_reports", [])
            self.consciousness = state.get("consciousness", [])
            self.recent = state.get("recent", [])
            self.segments = state.get("segments", {})
            self.segment_seq = state.get("segment_seq", 0)
            # OPEN-12: the rhythm itself survives a restart - a sleeping bot
            # that is switched off is still asleep when it comes back.
            self.asleep = state.get("asleep", self.asleep)
            self.pressure = float(state.get("pressure", self.pressure))
            self.pressure_ts = state.get("pressure_ts", self.pressure_ts)
            self.aroused_until = state.get("aroused_until", 0.0)
            self.state_since = state.get("state_since", self.last_sleep_ts)

            # Older ledgers called the layer below the focus `short_term`.
            legacy = state.get("short_term")
            if legacy:
                self.recent = self.recent + [e for e in legacy
                                             if isinstance(e, dict)]

            # Entries written before OPEN-15 did not carry the two clocks.
            for entry in self.recent + self.consciousness:
                ts = entry.setdefault("when_last_rehearsed",
                                      entry.get("ts", time.time()))
                entry.setdefault("when_first", ts)
                entry.setdefault("hits", 1)

            # Time passed while the bot was "unconscious": everything decays
            # on the curve (no more multiplying `strength` by a constant).
            load = len(self.recent) / float(max(1, int(self.policy["recent_capacity"])))
            floor = float(self.policy["forget_threshold"])
            now = time.time()
            self.recent = [
                e for e in self.recent
                if P.accessibility(max(0.0, now - e["when_last_rehearsed"]),
                                   load) >= floor
            ]
            self.consciousness = [
                e for e in self.consciousness
                if P.accessibility(max(0.0, now - e["when_last_rehearsed"]),
                                   load) >= floor
            ]

            # Catch the rhythm up after the downtime: pressure kept moving
            # while the machine was off.  H- is always below H+, so the two
            # conditions can never both hold - this settles in one flip.
            for _ in range(4):
                self._advance(now)
                upper, lower = P.thresholds(now)
                if self.asleep and self.pressure <= lower:
                    self.asleep = False
                    self.state_since = now
                    continue
                if (not self.asleep and now >= self.aroused_until
                        and self.pressure >= upper):
                    self.asleep = True
                    self.state_since = now
                    continue
                break
        except Exception:
            pass

    def _save(self):
        path = self._state_path()
        state = {
            "last_sleep_ts": self.last_sleep_ts,
            "state_since": self.state_since,
            "asleep": self.asleep,
            "pressure": self.pressure,
            "pressure_ts": self.pressure_ts,
            "aroused_until": self.aroused_until,
            "total_recorded": self.total_recorded,
            "total_dropped": self.total_dropped,
            "total_dropped_asleep": self.total_dropped_asleep,
            "startle_count": self.startle_count,
            "events_since_sleep": self.events_since_sleep,
            "consciousness": self.consciousness,
            "recent": self.recent,
            "segments": self.segments,
            "segment_seq": self.segment_seq,
            "sleep_reports": self.sleep_reports,
            "last_updated": datetime.now().isoformat(),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
