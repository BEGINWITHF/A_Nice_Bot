"""
Baby Brain - Internal Processing Only
No text input, no text output - like a real human baby
"""

import json
import os
import random
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
    def __init__(self, data_dir="data/baby"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Core state - starts empty
        self.stage = DevelopmentalStage.BIRTH
        self.age = 0
        self.experience_count = 0
        
        # Memory systems - start empty
        self.episodic_memory = []
        self.semantic_memory = {}
        self.procedural_memory = {}
        
        # Learning state - starts empty
        self.vocabulary = set()
        self.concepts = {}
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
                    self.vocabulary = set(state.get("vocabulary", []))
                    self.concepts = state.get("concepts", {})
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
            "vocabulary": list(self.vocabulary),
            "concepts": self.concepts,
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
    
    def learn_word(self, word, meaning=None):
        """Learn a new word"""
        self.vocabulary.add(word)
        if meaning:
            self.semantic_memory[word] = meaning
        self._save_state()
    
    def learn_concept(self, concept, definition):
        """Learn an abstract concept"""
        self.concepts[concept] = definition
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
            "vocabulary_size": len(self.vocabulary),
            "concepts_learned": len(self.concepts),
            "trust_level": self.trust_level,
            "known_people": len(self.known_people)
        }
    
    def internal_process(self, input_text, username):
        """
        Process input internally - no text output
        The AI's thoughts are not readable
        """
        # Record interaction
        self.record_experience("social", input_text, {"from": username})
        
        # Update known person
        if username not in self.known_people:
            self.add_known_person(username, "parent")
        self.known_people[username]["interactions"] += 1
        self.add_trust(0.5)
        
        # Learn words from input
        words = input_text.lower().split()
        for word in words:
            if len(word) > 2:
                self.learn_word(word)
        
        # Internal state changes happen silently
        return None
