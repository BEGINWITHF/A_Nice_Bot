"""
Long term episodic store - the layer the memory pipeline transfers into.

Clause OPEN-17 (author's answer 2026-09-29: "new long term consciousness
store"): promotion no longer asks each subsystem for a `promote` hook.  The
pipeline owns this store directly, and what it keeps is deliberately small:

    {when_first, when_last_rehearsed, where, what, kind}

Five fields, and each of them has to pass DATA-5 - it must also be a thing a
human brain has (a time it first happened, a time it was last recalled, a
place, the content, and which sense it came through).  Deliberately NOT
stored here: `times_seen`, `memory_strength`, `salience` - those are
engineering parameters, not parts of an episode (DATA-4 / DATA-5).

Clause OPEN-14: long term entries are forgotten by the same curve as
everything else, and only during sleep.  There is no "kept forever" layer.
"""

import json
import os
import time

from core.parameters import (
    FORGET_THRESHOLD,
    LONG_TERM_BUDGET,
    accessibility,
)


def _episode_key(kind, what):
    """Identity of an episode: same sense + same content = same episode."""
    if isinstance(what, str):
        body = what.strip().lower()
    else:
        body = json.dumps(what, sort_keys=True, default=str, ensure_ascii=False)
    return f"{kind}|{body}"


class LongTermEpisodes:
    """
    Holds episodes.  Three jobs, all of them called by sleep() and nowhere
    else (OPEN-14: forgetting happens during sleep only).
    """

    def __init__(self, data_dir="data/memory", budget=LONG_TERM_BUDGET):
        self.data_dir = data_dir
        self.budget = int(budget)
        self.episodes = []          # [{when_first, when_last_rehearsed, where, what, kind}]
        self._index = {}            # key -> position, rebuilt on load
        os.makedirs(data_dir, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Write side - called by the pipeline during sleep
    # ------------------------------------------------------------------
    def promote(self, entry, now=None):
        """
        Move one short term entry into long term.

        Re-meeting something does NOT create a second episode: the existing
        one is refreshed instead (its `when_last_rehearsed` moves forward,
        which is what buys it more time on the forgetting curve).  Returns
        True when an episode now holds this content, False if it was refused.
        """
        now = time.time() if now is None else now
        if not isinstance(entry, dict):
            return False

        kind = entry.get("kind")
        what = entry.get("payload")
        if kind is None or what is None:
            return False

        key = _episode_key(kind, what)
        pos = self._index.get(key)
        if pos is not None:
            # Re-seeing it is a rehearsal: same episode, later clock.
            self.episodes[pos]["when_last_rehearsed"] = now
            self._save()
            return True

        payload_where = None
        if isinstance(what, dict):
            payload_where = what.get("where")

        self._index[key] = len(self.episodes)
        self.episodes.append({
            "when_first": entry.get("when_first", now),
            "when_last_rehearsed": now,
            "where": payload_where,
            "what": what,
            "kind": kind,
        })
        self._save()
        return True

    def forget_pass(self, now=None):
        """
        Forget what the curve has taken away, then respect the budget.

        Only called from sleep() - clause OPEN-14.  Returns how many were
        deleted, because a store that reports 0 forever is not forgetting.
        """
        now = time.time() if now is None else now
        # One crowded layer makes everything in it effectively older (soft
        # capacity, OPEN-15 D) instead of evicting the weakest entry.
        load = len(self.episodes) / float(max(1, self.budget))

        kept = []
        for ep in self.episodes:
            elapsed = max(0.0, now - ep.get("when_last_rehearsed", now))
            if accessibility(elapsed, load) < FORGET_THRESHOLD:
                continue
            kept.append(ep)

        deleted = len(self.episodes) - len(kept)

        # Budget is a hard engineering ceiling, not a memory rule: if the
        # curve alone did not get us under it, drop the least recently
        # rehearsed (oldest `when_last_rehearsed` first).
        if len(kept) > self.budget:
            kept.sort(key=lambda e: e.get("when_last_rehearsed", 0))
            deleted += len(kept) - self.budget
            kept = kept[-self.budget:]      # keep the most recently rehearsed
            kept.sort(key=lambda e: e.get("when_first", 0))

        self.episodes = kept
        self._reindex()
        if deleted:
            self._save()
        return deleted

    # ------------------------------------------------------------------
    # Read side - reports and tests
    # ------------------------------------------------------------------
    def count(self):
        return len(self.episodes)

    def find(self, kind, what):
        """The episode holding this content, or None."""
        pos = self._index.get(_episode_key(kind, what))
        return self.episodes[pos] if pos is not None else None

    def accessibility_of(self, ep, now=None):
        """Current availability of one episode (nothing is stored, DATA-4)."""
        now = time.time() if now is None else now
        load = len(self.episodes) / float(max(1, self.budget))
        return accessibility(max(0.0, now - ep.get("when_last_rehearsed", now)),
                             load)

    def stats(self):
        return {
            "episodes": len(self.episodes),
            "budget": self.budget,
            "kinds": sorted({e.get("kind") for e in self.episodes
                             if e.get("kind")}),
        }

    # ------------------------------------------------------------------
    # Persistence - the pipeline's own ledger, written here
    # ------------------------------------------------------------------
    def _state_path(self):
        return os.path.join(self.data_dir, "episodes.json")

    def _reindex(self):
        self._index = {_episode_key(e.get("kind"), e.get("what")): i
                       for i, e in enumerate(self.episodes)}

    def _load(self):
        path = self._state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = data.get("episodes", []) if isinstance(data, dict) else []
            self.episodes = [e for e in loaded if isinstance(e, dict)]
            self._reindex()
        except (ValueError, OSError):
            self.episodes = []
            self._index = {}

    def _save(self):
        path = self._state_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"episodes": self.episodes}, f,
                      ensure_ascii=False, indent=1)
        os.replace(tmp, path)
