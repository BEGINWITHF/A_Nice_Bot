"""
Truly Pure Learning System - Sensory Experience Only
No text input, no text output - like a real human baby
"""

import json
import os
from datetime import datetime
from core.human_like import HumanLikeSystem
from core.parameters import FORGET_THRESHOLD, accessibility


def _seconds_since(stamp, now):
    """
    Seconds between an ISO timestamp and `now`.

    Anything unreadable counts as "an eternity ago", which on the forgetting
    curve means the entry is already gone: a corrupt timestamp must never
    make a memory immortal.
    """
    if not isinstance(stamp, str):
        return float("inf")
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return float("inf")
    if moment.tzinfo is not None:
        moment = moment.replace(tzinfo=None)
    return max(0.0, (now - moment).total_seconds())


class PureLearningSystem:
    """
    A learning system that behaves like a real human.
    No external dependencies.

    What a person has and what a person does not have, applied literally
    (DATA-7, reading A: no natural language may appear in memory):

        no    a list of words with indices and frequencies - nobody keeps
              one of those in their head, and it is the shape of an ASR
              vocabulary rather than of a mind
        no    a stored English sentence meaning a word - meaning is grounded
              in perception (Barsalou's perceptual symbol system), not held
              as a gloss in the language it explains
        no    a machine whose input and output are word indices - both of
              the networks that used to live here were that, and without a
              vocabulary neither can even be constructed
        yes   sequential pattern learning - people do pick up what follows
              what, from experience rather than from a list
        yes   emotion, personality, preference and social state - HumanLike
              System, already stripped of its text store by OPEN-19

    `core/pure_network.py` is gone with them rather than kept empty: nothing
    referenced it, and a network belongs to the day STEP-2 has perception to
    point one at (OPEN-9 - the old no longer matches the present, so it goes).
    """

    def __init__(self, data_dir="data/pure"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Human-like system
        self.human = HumanLikeSystem(data_dir=os.path.join(data_dir, "human"))

        # No sensors of its own: seeing and hearing belong to SensorySystem,
        # which is the store the memory pipeline registers. A second copy here
        # would hold memories that no sleep pass ever reaches (OPEN-9).

        # Learned transitions. This is the part that is not language: a pair
        # is a pair of anything, and STEP-2 will hand it perception.
        self.patterns = []

        # Load existing state
        self._load_state()

    def _load_state(self):
        """Load learning state from disk"""
        state_file = os.path.join(self.data_dir, "learning_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    # An older ledger may still carry word_to_index,
                    # index_to_word, word_frequency, vocabulary_size and
                    # concepts.  They are read past and never assigned: the
                    # language is not migrated in, it is dropped, and the
                    # next save is what actually erases it from disk.
                    # DATA-5: no pattern may carry a score.  Older ledgers
                    # stored `strength`; dropping it here is what migrates them.
                    self.patterns = [
                        {"from": p["from"], "to": p["to"],
                         "learned_at": p.get("learned_at")
                         or datetime.now().isoformat()}
                        for p in state.get("patterns", [])
                        if isinstance(p, dict) and "from" in p and "to" in p
                    ]
            except Exception as e:
                print(f"Error loading state: {e}")

    def _save_state(self):
        """Save learning state to disk"""
        state_file = os.path.join(self.data_dir, "learning_state.json")
        state = {
            "patterns": self.patterns,
            "last_updated": datetime.now().isoformat()
        }

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def learn_pattern(self, sequence):
        """
        Learn a pattern from a sequence of whatever just happened.

        Re-meeting a pair is a rehearsal, not a higher score: it pushes
        `learned_at` forward, and availability is derived from that clock
        (DATA-5 - a memory may not carry a parameter of its own; the curve is
        computed, never stored, per parameter-values section 4).
        """
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])

            found = False
            for p in self.patterns:
                if p["from"] == pair[0] and p["to"] == pair[1]:
                    p["learned_at"] = datetime.now().isoformat()
                    found = True
                    break

            if not found:
                self.patterns.append({
                    "from": pair[0],
                    "to": pair[1],
                    "learned_at": datetime.now().isoformat()
                })

        self._save_state()

    def get_stats(self):
        """Get learning statistics"""
        return {
            "patterns_learned": len(self.patterns),
            "human_like": self.human.get_stats()
        }

    # ------------------------------------------------------------------
    # Forgetting during sleep (called by MemoryPipeline.sleep)
    # ------------------------------------------------------------------
    def forget_pass(self, max_patterns=2000):
        """
        Sleep-time pruning, called by MemoryPipeline.sleep().

        Nothing is compared against a stored score - there is none left to
        compare with.  Availability is computed from `learned_at` on the same
        two-stage curve the other layers use (OPEN-11b: forgetting is a curve,
        not a step; OPEN-14: it runs during sleep only).  The budget stays,
        because it is an engineering ceiling rather than a memory rule;
        when it bites, the oldest `learned_at` goes first.

        The old `forget_threshold=0.05` parameter is gone too: it was a
        magic number with no source (DATA-6), and the floor it used to hold
        is now FORGET_THRESHOLD from core/parameters.py.

        There used to be a second pass here that pruned `concepts` and ranked
        the survivors by `word_frequency` - "the brain does track which words
        are common".  It is gone with them: frequency of occurrence is a
        property of a corpus, not of a person.
        """
        now = datetime.now()
        was_patterns = len(self.patterns)

        load = len(self.patterns) / float(max(1, max_patterns))
        kept = [p for p in self.patterns
                if accessibility(_seconds_since(p.get("learned_at"), now), load)
                >= FORGET_THRESHOLD]
        if len(kept) > max_patterns:
            kept.sort(key=lambda p: p.get("learned_at", ""))   # oldest first
            kept = kept[len(kept) - max_patterns:]             # newest survive
        self.patterns = kept

        report = {"patterns": was_patterns - len(self.patterns)}
        if report["patterns"]:
            self._save_state()

        return report
