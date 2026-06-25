"""
Truly Pure Learning System - Sensory Experience Only
No text input, no text output - like a real human baby
"""

import math
import random
import json
import os
from datetime import datetime
from core.pure_network import PureNeuralNetwork
from core.human_like import HumanLikeSystem
from core.hearing import HearingSystem
from core.seeing import SeeingSystem

class PureLearningSystem:
    """
    A learning system that behaves like a real human.
    No external dependencies - only Python's math module.
    """
    
    def __init__(self, data_dir="data/pure"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Human-like system
        self.human = HumanLikeSystem(data_dir=os.path.join(data_dir, "human"))
        
        # Sensory systems
        self.hearing = HearingSystem(data_dir=os.path.join(data_dir, "hearing"))
        self.seeing = SeeingSystem(data_dir=os.path.join(data_dir, "seeing"))
        
        # Vocabulary - starts empty
        self.word_to_index = {"<UNK>": 0, "<PAD>": 1, "<START>": 2, "<END>": 3}
        self.index_to_word = {0: "<UNK>", 1: "<PAD>", 2: "<START>", 3: "<END>"}
        self.word_frequency = {}
        self.vocabulary_size = 4
        
        # Neural networks
        self.meaning_network = PureNeuralNetwork(
            input_size=100,
            hidden_size=64,
            output_size=10,
            learning_rate=0.01
        )
        
        self.prediction_network = PureNeuralNetwork(
            input_size=20,
            hidden_size=32,
            output_size=100,
            learning_rate=0.005
        )
        
        # Learned patterns
        self.patterns = []
        self.concepts = {}
        
        # Training data
        self.interactions = []
        
        # Load existing state
        self._load_state()
    
    def _load_state(self):
        """Load learning state from disk"""
        state_file = os.path.join(self.data_dir, "learning_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.word_to_index = state.get("word_to_index", self.word_to_index)
                    self.index_to_word = {int(k): v for k, v in state.get("index_to_word", {}).items()}
                    self.word_frequency = state.get("word_frequency", {})
                    self.vocabulary_size = state.get("vocabulary_size", 4)
                    self.patterns = state.get("patterns", [])
                    self.concepts = state.get("concepts", {})
                    
                # Load neural networks
                self.meaning_network.load(os.path.join(self.data_dir, "meaning_network.json"))
                self.prediction_network.load(os.path.join(self.data_dir, "prediction_network.json"))
            except Exception as e:
                print(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save learning state to disk"""
        state_file = os.path.join(self.data_dir, "learning_state.json")
        state = {
            "word_to_index": self.word_to_index,
            "index_to_word": {str(k): v for k, v in self.index_to_word.items()},
            "word_frequency": self.word_frequency,
            "vocabulary_size": self.vocabulary_size,
            "patterns": self.patterns,
            "concepts": self.concepts,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        # Save neural networks
        self.meaning_network.save(os.path.join(self.data_dir, "meaning_network.json"))
        self.prediction_network.save(os.path.join(self.data_dir, "prediction_network.json"))
    
    def learn_word(self, word):
        """Learn a new word"""
        word = word.lower().strip()
        
        if word not in self.word_to_index:
            index = self.vocabulary_size
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            self.vocabulary_size += 1
        
        self.word_frequency[word] = self.word_frequency.get(word, 0) + 1
        return self.word_to_index[word]
    
    def encode_word(self, word):
        """Encode a word into a fixed-size vector"""
        index = self.learn_word(word)
        encoding = [0.0] * 100
        if index < 100:
            encoding[index] = 1.0
        encoding = [x + random.gauss(0, 0.01) for x in encoding]
        return encoding
    
    def get_meaning(self, word):
        """Get the meaning of a word"""
        encoding = self.encode_word(word)
        meaning = self.meaning_network.predict(encoding)
        return meaning
    
    def learn_meaning(self, word, meaning_context):
        """Learn the meaning of a word from context"""
        encoding = self.encode_word(word)
        
        target = [0.0] * 10
        context_words = meaning_context.lower().split()
        for i, w in enumerate(context_words[:10]):
            if w in self.word_to_index:
                target[i % 10] += 1.0
        
        total = sum(target)
        if total > 0:
            target = [x / total for x in target]
        
        self.meaning_network.learn_from_interaction(encoding, target)
        
        self.concepts[word] = {
            "context": meaning_context,
            "learned_at": datetime.now().isoformat()
        }
        
        self._save_state()
    
    def predict_next_word(self, context_words):
        """Predict the next word given context"""
        context = [0.0] * 20
        for i, word in enumerate(context_words[-5:]):
            if word in self.word_to_index:
                idx = self.word_to_index[word]
                context[i * 4] = idx / self.vocabulary_size
        
        predictions = self.prediction_network.predict(context)
        
        indexed_predictions = [(i, p) for i, p in enumerate(predictions)]
        indexed_predictions.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, confidence in indexed_predictions[:5]:
            if idx in self.index_to_word:
                word = self.index_to_word[idx]
                results.append({"word": word, "confidence": confidence})
        
        return results
    
    def learn_pattern(self, sequence):
        """Learn a pattern from a sequence of words"""
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            
            found = False
            for p in self.patterns:
                if p["from"] == pair[0] and p["to"] == pair[1]:
                    p["strength"] += 1
                    found = True
                    break
            
            if not found:
                self.patterns.append({
                    "from": pair[0],
                    "to": pair[1],
                    "strength": 1,
                    "learned_at": datetime.now().isoformat()
                })
        
        self._save_state()
    
    def internal_process(self, input_text):
        """
        Process input internally - no text output
        Like a human thinking silently
        """
        # Learn words from input
        input_words = input_text.lower().split()
        for word in input_words:
            self.learn_word(word)
        
        # Process through human-like system
        self.human.process_input(input_text, "world")
        
        # Internal state changes happen silently
        # No text response generated
        return None
    
    def hear_word(self, sound):
        """
        Process a sound heard through microphone
        Like a baby hearing sounds
        """
        # Learn the sound
        self.learn_word(sound)
        
        # Process through hearing system
        self.hearing.hear_sound(sound)
        
        # Internal state changes happen silently
        return None
    
    def get_stats(self):
        """Get learning statistics"""
        return {
            "vocabulary_size": self.vocabulary_size,
            "patterns_learned": len(self.patterns),
            "concepts_learned": len(self.concepts),
            "total_interactions": len(self.interactions),
            "knowledge_level": self.meaning_network.knowledge_level,
            "human_like": self.human.get_stats()
        }
