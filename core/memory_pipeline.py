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
budgets and a circadian rhythm. No neural network takes part in it.
It does not understand anything - it only allocates attention.

Three invariants:
  1. The sensing/behaviour loops are never blocked by sleep. sleep() runs a
     single pass and returns immediately - a sleeping person still hears.
  2. Every memory has a cap. Each registered store gets a capacity budget.
  3. Forgetting really deletes. Strength below threshold is removed, instead
     of being multiplied by 0.9 forever.
"""

import hashlib
import json
import os
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Policy parameters (all thresholds, no models)
# ---------------------------------------------------------------------------
DEFAULT_POLICY = {
    # --- when to memorize ---
    "min_salience": 0.25,          # below this salience: never enters the buffer
    "short_term_capacity": 128,    # max entries held in working memory at once
    "reinforce_on_repeat": True,   # meeting it again strengthens, not duplicates
    "repeat_salience_boost": 0.08, # strength gained per repeat

    # --- when to sleep (daily rhythm) ---
    "sleep_after_seconds": 900,    # awake for 15 min -> time for one sleep pass
    "sleep_after_events": 200,     # or after 200 sensed events

    # --- what sleep does ---
    "decay_per_sleep": 0.85,       # strength decay applied on every pass
    "replay_boost": 0.05,          # salient memories are replayed, hence stronger
    "replay_salience": 0.6,        # salience needed to count as "worth replaying"
    "promote_threshold": 0.6,      # strength reaching this -> handed to long term
    "forget_threshold": 0.08,      # strength below this -> really deleted
    "max_sleep_reports": 20,       # how many sleep reports the ledger keeps

    # --- the dedup index cannot grow forever either ---
    "seen_key_max": 4096,          # max dedup index entries
    "seen_key_retention": 604800,  # keep dedup index for 7 days; after that the
                                   # same thing feels "new" again
}


def _fingerprint(kind, payload):
    """Stable fingerprint used to answer: have I seen this before?"""
    if isinstance(payload, str):
        body = payload.strip().lower()
    else:
        body = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(f"{kind}|{body}".encode("utf-8", "replace")).hexdigest()[:16]


class MemoryPipeline:
    """
    Memory pipeline: gate (encode) -> short term buffer -> sleep -> long term / delete.
    """

    def __init__(self, data_dir="data/memory", policy=None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.policy = dict(DEFAULT_POLICY)
        if policy:
            self.policy.update(policy)

        # Working memory
        self.short_term = []          # [{key, kind, payload, salience, strength, ts, hits}]
        self.seen_keys = {}           # key -> last seen ts (kept across sleeps, for dedup)

        # Rhythm state
        self.last_sleep_ts = time.time()
        self.events_since_sleep = 0
        self.total_recorded = 0
        self.total_dropped = 0

        # Registered long term stores: name -> {"consolidate": fn, "forget": fn, "promote": fn}
        self.stores = {}
        self.sleep_reports = []

        self._load()

    # ------------------------------------------------------------------
    # Register long term stores
    # ------------------------------------------------------------------
    def register(self, name, consolidate=None, forget=None, promote=None):
        """Attach a subsystem. forget(budget) MUST actually delete weak memories."""
        self.stores[name] = {
            "consolidate": consolidate,
            "forget": forget,
            "promote": promote,
        }

    # ------------------------------------------------------------------
    # When to memorize
    # ------------------------------------------------------------------
    def record(self, kind, payload, salience=0.5, now=None):
        """
        Try to encode one sensed event.

        Returns the short term entry, or None when the policy rejected it.
        """
        now = time.time() if now is None else now
        salience = max(0.0, min(1.0, float(salience)))

        # Gate 1: unimportant things are not memorized
        if salience < self.policy["min_salience"]:
            self.total_dropped += 1
            return None

        key = _fingerprint(kind, payload)

        # Gate 2: repeats are not stored twice, only reinforced
        existing = self._find(key)
        if existing is not None:
            existing["hits"] += 1
            existing["strength"] = min(1.0, existing["strength"] + self.policy["repeat_salience_boost"])
            existing["salience"] = max(existing["salience"], salience)
            existing["ts"] = now
            self.seen_keys[key] = now
            self.events_since_sleep += 1
            return existing

        entry = {
            "key": key,
            "kind": kind,
            "payload": payload,
            "salience": salience,
            "strength": 0.5,
            "ts": now,
            "hits": 1,
        }

        # Gate 3: buffer full -> the weakest/oldest is forgotten on the spot
        if len(self.short_term) >= self.policy["short_term_capacity"]:
            self._evict_weakest()

        self.short_term.append(entry)
        self.seen_keys[key] = now
        self.total_recorded += 1
        self.events_since_sleep += 1
        return entry

    def _prune_seen_keys(self, now):
        """Drop expired and oversized dedup entries. Returns how many were removed."""
        cutoff = now - self.policy["seen_key_retention"]
        before = len(self.seen_keys)
        self.seen_keys = {k: ts for k, ts in self.seen_keys.items() if ts >= cutoff}

        if len(self.seen_keys) > self.policy["seen_key_max"]:
            overflow = sorted(self.seen_keys.items(), key=lambda kv: kv[1])
            drop = len(self.seen_keys) - self.policy["seen_key_max"]
            for k, _ in overflow[:drop]:
                del self.seen_keys[k]

        return before - len(self.seen_keys)

    def _find(self, key):
        for e in self.short_term:
            if e["key"] == key:
                return e
        return None

    def _evict_weakest(self):
        """When the buffer overflows, drop the least strong, least recent entry."""
        if not self.short_term:
            return
        victim = min(self.short_term, key=lambda e: (e["strength"], e["ts"]))
        self.short_term.remove(victim)
        self.total_dropped += 1

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
        """
        now = time.time() if now is None else now
        p = self.policy

        report = {
            "reason": reason,
            "started": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "awake_seconds": round(now - self.last_sleep_ts, 1),
            "events_processed": self.events_since_sleep,
            "short_term_in": len(self.short_term),
            "promoted": [],
            "dropped_below_threshold": 0,
            "evicted": 0,
            "replayed": 0,
            "stores": {},
        }

        # --- phase 1: decay (strength naturally falls asleep) ---
        for e in self.short_term:
            e["strength"] *= p["decay_per_sleep"]

        # --- phase 2/3/4: replay, promote, drop ---
        survivors = []
        for e in self.short_term:
            if e["salience"] >= p["replay_salience"]:
                e["strength"] = min(1.0, e["strength"] + p["replay_boost"])
                report["replayed"] += 1

            if e["strength"] >= p["promote_threshold"]:
                if self._promote(e):
                    report["promoted"].append(e["kind"])
                survivors.append(e)
            elif e["strength"] < p["forget_threshold"]:
                report["dropped_below_threshold"] += 1
            else:
                survivors.append(e)

        report["short_term_out"] = len(survivors)
        self.short_term = survivors

        # --- phase 5: every long term store consolidates and prunes itself ---
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

        # --- phase 6: the dedup index has a budget too ---
        pruned_keys = self._prune_seen_keys(now)
        if pruned_keys:
            report["seen_keys_pruned"] = pruned_keys

        # --- finish: reset the rhythm ---
        self.last_sleep_ts = now
        self.events_since_sleep = 0
        self.sleep_reports.append(report)
        self.sleep_reports = self.sleep_reports[-p["max_sleep_reports"]:]
        self._save()

        return report

    def _promote(self, entry):
        hooks = self.stores.get(entry["kind"])
        if hooks and hooks["promote"]:
            try:
                hooks["promote"](entry)
                return True
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def stats(self):
        return {
            "short_term": len(self.short_term),
            "capacity": self.policy["short_term_capacity"],
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
            self.seen_keys = state.get("seen_keys", {})
            self.sleep_reports = state.get("sleep_reports", [])
            self.short_term = state.get("short_term", [])
            # Time passed while the bot was "unconscious": working memory decays
            elapsed = time.time() - self.last_sleep_ts
            stale = min(0.9, elapsed / max(1.0, self.policy["sleep_after_seconds"]))
            for e in self.short_term:
                e["strength"] *= (1.0 - stale)
            self.short_term = [
                e for e in self.short_term
                if e.get("strength", 0) >= self.policy["forget_threshold"]
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
            "seen_keys": self.seen_keys,
            "sleep_reports": self.sleep_reports,
            "short_term": self.short_term,
            "last_updated": datetime.now().isoformat(),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
