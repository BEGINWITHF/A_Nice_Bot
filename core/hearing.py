"""
Hearing System - Processes sounds like a human ear
The AI hears sounds, not sentences
"""

import random
import json
import os
from datetime import datetime

class HearingSystem:
    """
    The AI's hearing system.
    Hears sounds and words, not structured sentences.
    Like a baby hearing the world around them.
    """
    
    def __init__(self, data_dir="data/senses/hearing"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Hearing state
        self.hearing_sensitivity = 0.3  # 0=deaf, 1=perfect hearing
        self.volume_threshold = 0.2  # Minimum volume to notice
        self.attention_level = 0.5  # 0=distracted, 1=focused
        
        # What has been heard
        self.sounds_heard = []  # Raw sounds
        self.words_recognized = []  # Processed words
        self.patterns_detected = []  # Sound patterns
        
        # Learning
        self.known_sounds = {}  # sound -> meaning
        self.sound_frequencies = {}  # How often each sound occurs
        
        # Current state
        self.currently_hearing = False
        self.last_sound = None
        self.hearing_history = []
        
        self._load_state()
    
    def _load_state(self):
        """Load hearing state"""
        state_file = os.path.join(self.data_dir, "hearing_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.hearing_sensitivity = state.get("hearing_sensitivity", 0.3)
                    self.known_sounds = state.get("known_sounds", {})
                    self.words_recognized = state.get("words_recognized", [])
            except:
                pass
    
    def _save_state(self):
        """Save hearing state"""
        state_file = os.path.join(self.data_dir, "hearing_state.json")
        state = {
            "hearing_sensitivity": self.hearing_sensitivity,
            "known_sounds": self.known_sounds,
            "words_recognized": self.words_recognized,
            "last_updated": datetime.now().isoformat()
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def hear_sound(self, sound, volume=1.0, source="unknown"):
        """
        Process a sound like a human ear.
        The AI hears raw sounds, not structured text.
        
        Args:
            sound: The raw sound (could be a word, noise, etc.)
            volume: How loud (0.0 to 1.0)
            source: Where the sound came from
        """
        # Check if sound is loud enough to notice
        if volume < self.volume_threshold:
            return None
        
        # Adjust by hearing sensitivity
        effective_volume = volume * self.hearing_sensitivity
        
        # Record the sound
        sound_record = {
            "sound": sound,
            "volume": volume,
            "effective_volume": effective_volume,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "attention_level": self.attention_level
        }
        
        self.sounds_heard.append(sound_record)
        self.last_sound = sound
        
        # Try to recognize the sound
        recognized = self._recognize_sound(sound, effective_volume)
        
        if recognized:
            self.words_recognized.append(recognized)
            
            # Learn the sound
            if sound not in self.known_sounds:
                self.known_sounds[sound] = {
                    "first_heard": datetime.now().isoformat(),
                    "times_heard": 1,
                    "meaning": None
                }
            else:
                self.known_sounds[sound]["times_heard"] += 1
        
        # Improve hearing with use
        self.hearing_sensitivity = min(1.0, self.hearing_sensitivity + 0.001)
        self._save_state()
        
        return sound_record
    
    def _recognize_sound(self, sound, volume):
        """
        Try to recognize a sound.
        Like a baby learning to distinguish words from noise.
        """
        # Clean the sound
        sound = sound.strip().lower()
        
        # Check if it's a word (simple heuristic)
        if len(sound) > 0 and sound.isalpha():
            return {
                "word": sound,
                "confidence": min(1.0, volume),
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if it's a partial word or babble
        if len(sound) > 0:
            return {
                "word": sound,
                "confidence": volume * 0.5,  # Lower confidence for non-words
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def hear_sentence(self, sentence, speaker="unknown"):
        """
        Process a sentence as a stream of sounds.
        Like hearing someone speak - it's a flow of sounds, not a structured input.
        """
        # Split into individual sounds/words
        sounds = sentence.split()
        
        results = []
        for i, sound in enumerate(sounds):
            # Each word has slightly different volume (natural speech variation)
            volume = random.uniform(0.7, 1.0)
            
            result = self.hear_sound(sound, volume, speaker)
            if result:
                results.append(result)
        
        return results
    
    def get_attention_level(self):
        """Get current attention level"""
        return self.attention_level
    
    def set_attention(self, level):
        """Set attention level (0=distracted, 1=focused)"""
        self.attention_level = max(0.0, min(1.0, level))
        self._save_state()
    
    def get_stats(self):
        """Get hearing statistics"""
        return {
            "hearing_sensitivity": self.hearing_sensitivity,
            "total_sounds_heard": len(self.sounds_heard),
            "words_recognized": len(self.words_recognized),
            "unique_sounds": len(self.known_sounds),
            "attention_level": self.attention_level
        }
