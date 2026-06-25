"""
Hearing System - Processes sounds like a human ear with biological auditory processing
The AI hears sounds, not sentences

Biological auditory processing:
1. Cochlea: Frequency decomposition (tonotopic organization)
2. Brainstem: Basic sound processing
3. Auditory Cortex: Complex sound recognition
4. Wernicke's Area: Language comprehension
"""

import random
import json
import os
import math
from datetime import datetime

class HearingSystem:
    """
    The AI's hearing system with biological auditory processing.
    Hears sounds and words, not structured sentences.
    Like a baby hearing the world around them.
    
    Biological processing:
    - Cochlea: Decomposes sound into frequencies (tonotopic)
    - Brainstem: Processes timing, location
    - Auditory Cortex: Recognizes patterns
    - Wernicke's Area: Language understanding
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
        
        # Biological auditory processing
        self.cochlea_filters = self._init_cochlea()  # Frequency decomposition
        self.frequency_map = {}  # tonotopic map
        self.auditory_memory = {}  # sound patterns in memory
        
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
    
    def _init_cochlea(self):
        """
        Initialize cochlea-like frequency decomposition
        Like the basilar membrane in the inner ear
        Different regions respond to different frequencies
        """
        # 24 frequency bands (like critical bands in human hearing)
        # From 50Hz to 8000Hz logarithmically spaced
        min_freq = 50
        max_freq = 8000
        num_bands = 24
        
        filters = []
        for i in range(num_bands):
            # Center frequency (logarithmic spacing)
            center_freq = min_freq * (max_freq / min_freq) ** (i / (num_bands - 1))
            # Bandwidth (ERB-like)
            bandwidth = center_freq * 0.1 + 50
            
            filters.append({
                "center": center_freq,
                "bandwidth": bandwidth,
                "activation": 0.0
            })
        
        return filters
    
    def _cochlea_process(self, sound):
        """
        Process sound through cochlea-like frequency decomposition
        Like the basilar membrane vibrating at different locations
        """
        # Convert sound to frequency representation
        # Simple FFT-like analysis using DFT
        if isinstance(sound, str):
            # Convert text to frequency pattern
            freq_pattern = self._text_to_frequency(sound)
        else:
            freq_pattern = sound
        
        # Apply cochlear filters
        activations = []
        for i, filt in enumerate(self.cochlea_filters):
            # Simple resonance - each filter responds to nearby frequencies
            activation = 0.0
            for freq, magnitude in freq_pattern.items():
                distance = abs(freq - filt["center"])
                if distance < filt["bandwidth"]:
                    activation += magnitude * (1 - distance / filt["bandwidth"])
            
            activations.append(activation)
            self.cochlea_filters[i]["activation"] = activation
        
        return activations
    
    def _text_to_frequency(self, text):
        """
        Convert text to frequency pattern
        Each character maps to a frequency range
        """
        freq_map = {}
        for i, char in enumerate(text):
            # Map character to frequency based on position and ASCII value
            base_freq = 100 + (ord(char) % 26) * 100
            # Add harmonics
            for harmonic in range(1, 4):
                freq = base_freq * harmonic
                if freq not in freq_map:
                    freq_map[freq] = 0
                freq_map[freq] += 1.0 / harmonic
        
        return freq_map
    
    def _brainstem_process(self, cochlea_output):
        """
        Brainstem processing: Basic sound analysis
        Like inferior colliculus processing
        """
        processed = {
            "onset": sum(cochlea_output) > 0.1,  # Sound started
            "loudness": sum(cochlea_output) / len(cochlea_output),
            "pitch_centroid": 0,
            "frequency_spread": 0
        }
        
        # Calculate pitch centroid (average frequency)
        total_activation = sum(cochlea_output)
        if total_activation > 0:
            for i, activation in enumerate(cochlea_output):
                processed["pitch_centroid"] += activation * self.cochlea_filters[i]["center"]
            processed["pitch_centroid"] /= total_activation
        
        # Calculate frequency spread
        if total_activation > 0:
            mean_freq = processed["pitch_centroid"]
            for i, activation in enumerate(cochlea_output):
                processed["frequency_spread"] += activation * (self.cochlea_filters[i]["center"] - mean_freq) ** 2
            processed["frequency_spread"] = math.sqrt(processed["frequency_spread"] / total_activation)
        
        return processed
    
    def _auditory_cortex_process(self, brainstem_output, cochlea_output):
        """
        Auditory cortex processing: Pattern recognition
        Like superior temporal gyrus
        """
        patterns = []
        
        # Detect temporal patterns (rhythm)
        recent_sounds = self.sounds_heard[-10:] if self.sounds_heard else []
        if len(recent_sounds) > 2:
            intervals = []
            for i in range(1, len(recent_sounds)):
                t1 = datetime.fromisoformat(recent_sounds[i-1]["timestamp"])
                t2 = datetime.fromisoformat(recent_sounds[i]["timestamp"])
                intervals.append((t2 - t1).total_seconds())
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval < 1.0:  # Fast speech
                    patterns.append("speech")
                elif avg_interval > 2.0:  # Slow, deliberate
                    patterns.append("careful_speech")
        
        # Detect frequency patterns
        active_filters = [i for i, a in enumerate(cochlea_output) if a > 0.1]
        if len(active_filters) > 3:
            patterns.append("complex_sound")
        elif len(active_filters) == 1:
            patterns.append("pure_tone")
        
        return patterns
    
    def _wernicke_process(self, sound, auditory_memory):
        """
        Wernicke's area: Language comprehension
        Try to match sound to known words
        """
        # Clean the sound
        clean_sound = sound.strip().lower()
        
        # Check if we've heard this before
        if clean_sound in auditory_memory:
            memory = auditory_memory[clean_sound]
            return {
                "word": clean_sound,
                "confidence": memory.get("strength", 0.5),
                "meaning": memory.get("meaning"),
                "times_heard": memory.get("times_heard", 0)
            }
        
        # Try to match to similar sounds (phonological similarity)
        best_match = None
        best_similarity = 0
        
        for known_word, memory in auditory_memory.items():
            similarity = self._calculate_phonological_similarity(clean_sound, known_word)
            if similarity > best_similarity and similarity > 0.6:
                best_similarity = similarity
                best_match = known_word
        
        if best_match:
            return {
                "word": best_match,
                "confidence": best_similarity * 0.8,
                "meaning": auditory_memory[best_match].get("meaning"),
                "similar_to": best_match
            }
        
        return None
    
    def _calculate_phonological_similarity(self, word1, word2):
        """
        Calculate phonological similarity between two words
        Based on shared phonemes and sound patterns
        """
        if not word1 or not word2:
            return 0.0
        
        # Simple character-based similarity
        common = 0
        for i, c in enumerate(word1):
            if i < len(word2) and c == word2[i]:
                common += 1
        
        max_len = max(len(word1), len(word2))
        return common / max_len if max_len > 0 else 0.0
    
    def hear_sound(self, sound, volume=1.0, source="unknown"):
        """
        Process a sound like a human ear with biological processing.
        The AI hears raw sounds, not structured text.
        
        Biological pipeline:
        1. Cochlea: Frequency decomposition
        2. Brainstem: Basic processing
        3. Auditory Cortex: Pattern recognition
        4. Wernicke's Area: Language comprehension
        """
        # Check if sound is loud enough to notice
        if volume < self.volume_threshold:
            return None
        
        # Adjust by hearing sensitivity
        effective_volume = volume * self.hearing_sensitivity
        
        # 1. Cochlea processing (frequency decomposition)
        cochlea_output = self._cochlea_process(sound)
        
        # 2. Brainstem processing (basic analysis)
        brainstem_output = self._brainstem_process(cochlea_output)
        
        # 3. Auditory cortex processing (pattern recognition)
        cortex_patterns = self._auditory_cortex_process(brainstem_output, cochlea_output)
        
        # 4. Wernicke's area (language comprehension)
        wernicke_result = self._wernicke_process(sound, self.auditory_memory)
        
        # Record the sound
        sound_record = {
            "sound": sound,
            "volume": volume,
            "effective_volume": effective_volume,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "attention_level": self.attention_level,
            "cochlea": cochlea_output,
            "brainstem": brainstem_output,
            "cortex_patterns": cortex_patterns,
            "wernicke": wernicke_result
        }
        
        self.sounds_heard.append(sound_record)
        self.last_sound = sound
        
        # Try to recognize the sound
        recognized = self._recognize_sound(sound, effective_volume)
        
        if recognized:
            self.words_recognized.append(recognized)
            
            # Learn the sound - store in auditory memory
            clean_sound = sound.strip().lower()
            if clean_sound not in self.auditory_memory:
                self.auditory_memory[clean_sound] = {
                    "first_heard": datetime.now().isoformat(),
                    "times_heard": 1,
                    "meaning": None,
                    "cochlea_pattern": cochlea_output,
                    "strength": 0.5
                }
            else:
                self.auditory_memory[clean_sound]["times_heard"] += 1
                # Strengthen memory with each hearing (Hebbian learning)
                self.auditory_memory[clean_sound]["strength"] = min(1.0,
                    self.auditory_memory[clean_sound]["strength"] + 0.05)
            
            # Also store in known_sounds for compatibility
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
            "attention_level": self.attention_level,
            "auditory_memory_size": len(self.auditory_memory)
        }
    
    def consolidate_auditory_memory(self):
        """
        Auditory Memory Consolidation
        Like sleep-dependent memory consolidation
        Reviews and strengthens important sound memories
        """
        consolidated = 0
        
        for sound, data in self.auditory_memory.items():
            times_heard = data.get("times_heard", 0)
            
            # Calculate memory importance
            importance = times_heard / 10.0  # More heard = more important
            
            # Strengthen based on importance
            if importance > 0.3:
                data["strength"] = min(1.0, data.get("strength", 0.5) + 0.01 * importance)
                consolidated += 1
            
            # Decay weak memories
            if data.get("strength", 0.5) < 0.1 and times_heard < 3:
                data["strength"] *= 0.9
        
        self._save_state()
        return consolidated
    
    def recognize_sound_pattern(self, cochlea_pattern):
        """
        Recognize a sound pattern from cochlea activation
        Like auditory cortex pattern matching
        """
        best_match = None
        best_similarity = 0
        
        for sound, data in self.auditory_memory.items():
            stored_pattern = data.get("cochlea_pattern")
            if stored_pattern and len(stored_pattern) == len(cochlea_pattern):
                # Calculate pattern similarity
                similarity = 0
                for i in range(len(cochlea_pattern)):
                    similarity += 1.0 - abs(cochlea_pattern[i] - stored_pattern[i])
                similarity /= len(cochlea_pattern)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = sound
        
        return best_match, best_similarity
