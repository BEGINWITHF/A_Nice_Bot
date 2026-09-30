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

Two invariants kept from the original:
  1. The sensing/behaviour loops are never blocked by sleep. sleep() runs a
     single pass and returns immediately - a sleeping person still hears.
  2. Forgetting really deletes. Below the threshold the entry is removed,
     instead of being multiplied by 0.9 forever.

Everything numeric lives in core/parameters.py (OPEN-13), so a whole source
can be swapped in one place.
"""

import hashlib
import json
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

    # --- when to sleep (still the placeholder rhythm; OPEN-12 pending) ---
    "sleep_after_seconds": P.SLEEP_AFTER_SECONDS,
    "sleep_after_events": P.SLEEP_AFTER_EVENTS,

    # --- what sleep does ---
    "replay_salience": P.REPLAY_SALIENCE,   # salient enough to be replayed
    "forget_threshold": P.FORGET_THRESHOLD, # below this: really deleted
    "promote_min_age": P.PROMOTE_MIN_AGE_S, # Cepeda 2006: >= 1 day before
                                            # sleep may transfer it
    "max_sleep_reports": P.MAX_SLEEP_REPORTS,

    # --- event segmentation (OPEN-16, decided: G) ---
    "window_s": P.WINDOW_S,             # default cut on timeout
    "shock_threshold": P.SHOCK_THRESHOLD, # content shock may cut early
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

    A frame showing nothing new scores below DEFAULT_POLICY["min_salience"]
    and is therefore never memorised.
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

def _set_fields(payload):
    """The set-valued fields of a payload - what a 'situation' is made of."""
    if not isinstance(payload, dict):
        return frozenset()
    tokens = set()
    for key, value in payload.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                tokens.add(f"{key}={item}")
        # scalars (e.g. volume_band) are readings, not situation: a changing
        # scalar is a degree, not a new event - Zacks et al. (2010) coded
        # character/spatial/goal/object changes as the situational dimensions.
    return frozenset(tokens)


def _shock_distance(old, new):
    """1 - Jaccard similarity between two situations. 1.0 = total change."""
    old, new = set(old or ()), set(new or ())
    if not old or not new:
        return 0.0            # nothing to compare: not a shock
    return 1.0 - (len(old & new) / float(len(old | new)))


def _merge_payload(old, new):
    """
    Fold one observation into the entry it belongs to.

    Lists are unioned (the situation accumulates what was there), scalars
    keep the first value (the entry stands for how the event started).
    # ASSUMPTION: no literature specifies how to merge two observations
                  inside one event; union-of-sets is the least lossy rule
                  that does not grow without bound.
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
        self.segments = {}        # kind -> {seq, started, situation, shock_slot}
        self.segment_seq = 0      # monotonic, so keys stay unique across restarts

        # Rhythm state
        self.last_sleep_ts = time.time()
        self.events_since_sleep = 0
        self.total_recorded = 0
        self.total_dropped = 0

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

    # -- segmentation (OPEN-16 G) -------------------------------------
    def _segment(self, kind, payload, now):
        """
        Decide which event this observation belongs to, and return its key.

        Default cut: the window expired (OPEN-16 primary rule).
        Early cut:   the situation changed enough (G's exception), allowed
                     at most once per window slot so that a burst of changes
                     cannot degrade back into option E (one entry per frame).
        """
        state = self.segments.get(kind)
        situation = _set_fields(payload)
        window = float(self.policy["window_s"])
        shock = float(self.policy["shock_threshold"])
        slot = int(now // window) if window > 0 else 0

        if state is None:
            return self._open_segment(kind, payload, now, slot)

        expired = (now - state["started"]) >= window
        changed = _shock_distance(state["situation"], situation) >= shock
        early = changed and slot != state["shock_slot"]

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
            # a sorted list, not a frozenset: this dict is written to disk
            "situation": sorted(_set_fields(payload)),
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
    # When to sleep
    # ------------------------------------------------------------------
    def due(self, now=None):
        """Awake time or event budget reached -> one sleep pass is due."""
        now = time.time() if now is None else now
        awake = now - self.last_sleep_ts
        return (
            awake >= self.policy["sleep_after_seconds"]
            or self.events_since_sleep >= self.policy["sleep_after_events"]
        )

    def awake_seconds(self, now=None):
        now = time.time() if now is None else now
        return max(0.0, now - self.last_sleep_ts)

    def sleep(self, now=None, reason="scheduled"):
        """
        Run one sleep pass.

        Key constraint: this is a **single non-blocking pass**. The spine and
        the senses keep running throughout - a sleeping person still reacts.

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

        report = {
            "reason": reason,
            "started": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "awake_seconds": round(now - self.last_sleep_ts, 1),
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
        return {
            "consciousness": len(self.consciousness),
            "conscious_capacity": self.policy["conscious_capacity"],
            "recent": len(self.recent),
            "recent_capacity": self.policy["recent_capacity"],
            "episodes": self.episodes.count(),
            "events_since_sleep": self.events_since_sleep,
            "awake_seconds": round(self.awake_seconds(), 1),
            "total_recorded": self.total_recorded,
            "total_dropped": self.total_dropped,
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
            self.events_since_sleep = state.get("events_since_sleep", 0)
            self.sleep_reports = state.get("sleep_reports", [])
            self.consciousness = state.get("consciousness", [])
            self.recent = state.get("recent", [])
            self.segments = state.get("segments", {})
            self.segment_seq = state.get("segment_seq", 0)

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
        except Exception:
            pass

    def _save(self):
        path = self._state_path()
        state = {
            "last_sleep_ts": self.last_sleep_ts,
            "total_recorded": self.total_recorded,
            "total_dropped": self.total_dropped,
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
