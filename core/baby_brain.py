"""
Baby Brain - Internal Processing Only
No text input, no text output - like a real human baby
"""

import json
import os
from datetime import datetime
from enum import Enum

class DevelopmentalStage(Enum):
    """Stages of AI development"""
    BIRTH = 0
    SENSORY = 1
    BABBLING = 2
    FIRST_WORDS = 3
    SENTENCES = 4
    REASONING = 5
    PERSONALITY = 6

class BabyBrain:
    """
    A baby's head before it has anything in it.

    DATA-7 read A applies here exactly as it applies to the rest of memory:
    no natural language may be stored.  So there is no word list, no
    concept-to-definition table and no semantic store keyed by a word -
    a baby does not come with those, and it does not build them, because
    meaning is grounded in what it perceives rather than in a gloss.  What
    it does build is attachment (trust, who spoke to it), an emotional
    history, and a record of how many things have happened to it.

    The stage names are English, but they are not held as data: what goes to
    disk is `stage.value`, and the labels are the human developmental
    sequence itself - babbling before first words before sentences.
    """

    def __init__(self, data_dir="data/baby"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Core state - starts empty
        self.stage = DevelopmentalStage.BIRTH
        self.age = 0
        self.experience_count = 0

        # Memory systems - start empty
        self.episodic_memory = []
        self.procedural_memory = {}
        self.patterns = []
        self.curiosities = []

        # Social bonding - starts empty
        self.trust_level = 0
        self.known_people = {}
        self.communication_attempts = []

        # Emotional state
        self.emotional_state = "neutral"
        self.emotional_history = []

        # Load existing state if any
        self._load_state()

    def _load_state(self):
        """Load baby's state from disk"""
        state_file = os.path.join(self.data_dir, "brain_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.stage = DevelopmentalStage(state.get("stage", 0))
                    self.age = state.get("age", 0)
                    self.experience_count = state.get("experience_count", 0)
                    # An older ledger may still carry `vocabulary` and
                    # `concepts`.  They are read past, never assigned: the
                    # language is not migrated in, it is dropped, and the
                    # next save is what erases it from disk.
                    self.trust_level = state.get("trust_level", 0)
                    self.known_people = state.get("known_people", {})
                    self.emotional_state = state.get("emotional_state", "neutral")
            except:
                pass

    def _save_state(self):
        """Save baby's state to disk"""
        state_file = os.path.join(self.data_dir, "brain_state.json")
        state = {
            "stage": self.stage.value,
            "age": self.age,
            "experience_count": self.experience_count,
            "trust_level": self.trust_level,
            "known_people": self.known_people,
            "emotional_state": self.emotional_state,
            "last_updated": datetime.now().isoformat()
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def record_experience(self, experience_type, content, context=None):
        """Record a new experience"""
        experience = {
            "type": experience_type,
            "content": content,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "stage": self.stage.value
        }
        self.episodic_memory.append(experience)
        self.experience_count += 1
        self.age += 1

        self._check_development()
        self._save_state()

        return experience

    def _check_development(self):
        """Check if ready to advance stage"""
        stage_requirements = {
            DevelopmentalStage.BIRTH: 1,
            DevelopmentalStage.SENSORY: 5,
            DevelopmentalStage.BABBLING: 15,
            DevelopmentalStage.FIRST_WORDS: 30,
            DevelopmentalStage.SENTENCES: 60,
            DevelopmentalStage.REASONING: 100,
        }

        if self.stage in stage_requirements:
            required = stage_requirements[self.stage]
            if self.experience_count >= required:
                next_stage = DevelopmentalStage(self.stage.value + 1)
                self.stage = next_stage
                self._save_state()

    def add_trust(self, amount=1):
        """Build trust with a person"""
        self.trust_level = min(100, self.trust_level + amount)
        self._save_state()

    def add_known_person(self, name, relationship="unknown"):
        """Remember a person"""
        self.known_people[name] = {
            "relationship": relationship,
            "interactions": 0,
            "trust": 0
        }
        self._save_state()

    def update_emotion(self, emotion):
        """Update emotional state"""
        self.emotional_state = emotion
        self.emotional_history.append({
            "emotion": emotion,
            "timestamp": datetime.now().isoformat()
        })
        self._save_state()

    def get_state_description(self):
        """Get description of current state"""
        return f"Stage: {self.stage.name}, Age: {self.age}"

    def get_memory_stats(self):
        """Get statistics"""
        return {
            "stage": self.stage.name,
            "age": self.age,
            "experiences": len(self.episodic_memory),
            "trust_level": self.trust_level,
            "known_people": len(self.known_people)
        }

    def internal_process(self, input_text, username):
        """
        Take in a social interaction without ever writing down what was said.

        DATA-7 read A: no natural language in memory.  A baby forms an
        attachment from being talked *to* - who spoke, how often, how that
        felt - and none of that is the sentence.  `input_text` is handed in
        only so the call site can say an interaction happened; nothing in
        here reads it, and nothing here writes it.
        """
        self.record_experience("social", None, {"from": username})

        # Update known person
        if username not in self.known_people:
            self.add_known_person(username, "parent")
        self.known_people[username]["interactions"] += 1
        self.add_trust(0.5)

        # Internal state changes happen silently
        return None
