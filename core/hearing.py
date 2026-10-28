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

from core import categories

# Attention's numbers, imported from vision rather than copied.  The hold
# and release rates were sourced from studies of attention itself, not of
# sight (Dugue et al. 2020: endogenous attention takes about 300 ms to
# deploy and can be sustained, exogenous about 100 ms and is transient), so
# they belong to attention and not to one sense.  The relaxation was
# measured once, in vision; there is no auditory measurement in the library,
# and making up a second number would be a second unmeasured number rather
# than a better one.
from core.seeing import (
    FOCUS_HOLD_TAU_S,
    FOCUS_RELEASE_TAU_S,
    FOCUS_THRESHOLD_RELAXATION,
)

# ---------------------------------------------------------------------------
# OPEN-22: from PCM to a `what`  -  Diary/articles/2026-10-28-hearing-design.md
# ---------------------------------------------------------------------------
#
# D1 hearing segregates into streams instead of only noticing change:
# Sussman 2017 L51 - "stream segregation occurs automatically and sets the
# basis for auditory event formation"; Calcus 2024 L37 - the mechanisms are
# "functional from birth", and L16 - newborns parse with "the same cues
# adults use".  That is hearing's IO-7: it happens without learning and
# without thinking, which is the only admission test IO-7 sets.
#
# D2 order of work: segregate first, describe afterwards.  Sussman 2017 L37
# - "Stream segregation occurs first, and then sound events are detected and
# identified on the already sorted streams."  So a stream is hearing's
# region, and the four numbers below are its descriptor.
#
# D3 the cue that binds bands into one stream is temporal coherence:
# Calcus 2024 L12 - "binding together coherent acoustic (spectral, spatial,
# and/or temporal) features into streams".
#
# D7 four descriptors, not nineteen, and the reason is Burns 2019 L37's own
# redundancy rule - 19 objective measures cut to 7 by dropping one of any
# pair correlating at |r| >= 0.45.  What each of the four is, and the paper
# that puts it there, sits on `_describe_stream`.
#
# D8 no name for any of it.  A stream gets an index from the shared
# induction rule (core/categories.py) and nothing else; DATA-7 and IO-7
# already said so, and Burns 2019 L86 adds that identifying a sound's source
# "appears only partly to rely on objective measures" anyway.

# # ASSUMPTION: framing.  2048 samples is about 46 ms at 44.1 kHz; 512 hops
# give seven-eighths overlap.  Nothing in the library states a window for
# stream segregation, so both numbers are ours - swap them freely, no other
# line reads them.
FRAME_SAMPLES = 2048
HOP_SAMPLES = 512

# # ASSUMPTION: how much of the mixture a band or a stream must carry to
# count at all.  Calcus 2024 L9 gives frequency distance as the segregation
# cue and no cut; below this line a band is leftovers of the decomposition
# rather than something that caught the ear.  One number doing both jobs is
# deliberate - "worth hearing" should not mean two different things.
STREAM_MIN_SHARE = 0.05

# How different two sounds must be to be two categories.
#
# Not vision's INDUCTION_TAU, and the reason is arithmetic rather than taste:
# `categories.similarity` returns 1 - *mean* absolute difference, so what a
# shared 0.85 asks for is an average gap of 0.15 across the components.  The
# two descriptors are not the same shape - vision's carries raw magnitudes
# over nineteen components, hearing's four values bounded to 0..1 - so the
# same number cannot mean the same thing on both.  Sharing the rule does not
# mean sharing its calibration.
#
# # ASSUMPTION: 0.91.  Barsalou offers "similar enough" and no number, so
# this stays an assumption; what is not assumed is *where the edges are*.
# Measured on 2026-10-28 against the four cases that matter, using pure
# tones built by the test file:
#
#   must separate  440 Hz from 3000 Hz, alone and inside a mixture -> 0.9022
#   must not split the same pitch at two loudnesses (20000 vs 6000) -> 0.9263
#
# which leaves the window (0.9022, 0.9263], and 0.91 sits in it.  0.90 sat
# one thousandth below the upper edge and collapsed a mixture's two tones
# into a single category, which is how the window was found.
#
# The window does not stretch to every pair: 3000 Hz against 8000 Hz inside
# a three-tone mixture reads 0.9565, so those two stay in one category.  The
# reason is the mean over four components diluting pitch - say so rather
# than paper over it.
HEARING_TAU = 0.91

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

        # OPEN-22: the category ledger, running the very rule vision runs
        # (core/categories.py).  A key is a content-free index - OPEN-21 C -
        # so nothing in this file can name a sound.
        self.auditory_ledger = {}
        self._next_category = 0
        self.last_induction = []
        self.what = []

        # OPEN-22 D4: attention in hearing, the same three rules vision
        # runs.  Focus belongs to this act of hearing and to no record: it is
        # never written to disk and never read by induction (R3), and what it
        # moves is the bar a stream has to clear to be heard at all.
        self.focus_level = 0.0        # 0 spread over everything, 1 held
        self._attention_band = None   # the band attention was last on
        self._attention_at = None
        
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
                    # 睡眠整理的就是这两样，不落盘的话睡眠等于白做
                    self.auditory_memory = state.get("auditory_memory", self.auditory_memory)
                    self.sound_frequencies = state.get("sound_frequencies", self.sound_frequencies)
                    # OPEN-22: without these, every restart would re-open
                    # every category and the ledger would grow forever.
                    self.auditory_ledger = state.get("auditory_ledger", self.auditory_ledger)
                    self._next_category = int(state.get("next_category", 0))
            except:
                pass
    
    def _save_state(self):
        """Save hearing state"""
        state_file = os.path.join(self.data_dir, "hearing_state.json")
        state = {
            "hearing_sensitivity": self.hearing_sensitivity,
            "known_sounds": self.known_sounds,
            "words_recognized": self.words_recognized[-500:],
            "auditory_memory": self.auditory_memory,
            "sound_frequencies": self.sound_frequencies,
            "auditory_ledger": self.auditory_ledger,
            "next_category": self._next_category,
            "last_updated": datetime.now().isoformat()
        }
        tmp = state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, state_file)
    
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

    def forget_pass(self, max_auditory=400, max_words=500, max_sounds=1000,
                    forget_threshold=0.05):
        """
        Sleep-time pruning, called by MemoryPipeline.sleep().

        This is where forgetting really happens: weak entries are deleted and
        every list is held to a budget. consolidate_auditory_memory() only
        adjusts strength - strength alone never removes anything.
        """
        dropped = {"auditory_memory": 0, "known_sounds": 0,
                   "sounds_heard": 0, "words_recognized": 0,
                   "auditory_ledger": 0}

        # Auditory patterns: delete the weak ones, then cap what is left
        before = len(self.auditory_memory)
        weak = [
            s for s, d in self.auditory_memory.items()
            if d.get("strength", 0.5) < forget_threshold
            and d.get("times_heard", 0) < 3
        ]
        for s in weak:
            del self.auditory_memory[s]
        dropped["auditory_memory"] += before - len(self.auditory_memory)

        if len(self.auditory_memory) > max_auditory:
            ranked = sorted(
                self.auditory_memory.items(),
                key=lambda kv: (kv[1].get("strength", 0.5), kv[1].get("times_heard", 0)),
                reverse=True,
            )
            keep = dict(ranked[:max_auditory])
            dropped["auditory_memory"] += len(self.auditory_memory) - len(keep)
            self.auditory_memory = keep

        # OPEN-22: the category ledger gains a row for every sound shape it
        # has never met, and no other hook prunes it, so without this it
        # grows without end.  Evicted by how often a category was met rather
        # than by strength - a category carries no strength of its own
        # (DATA-5), only the times it was matched.
        before = len(self.auditory_ledger)
        if before > max_auditory:
            ranked = sorted(
                self.auditory_ledger.items(),
                key=lambda kv: kv[1].get("times_seen", 0),
                reverse=True,
            )
            keep = dict(ranked[:max_auditory])
            dropped["auditory_ledger"] = before - len(keep)
            self.auditory_ledger = keep

        # Known sounds: never heard again -> gone
        before = len(self.known_sounds)
        self.known_sounds = {
            s: d for s, d in self.known_sounds.items()
            if d.get("times_heard", d.get("count", 1)) >= 2
            or d.get("strength", 0.5) >= forget_threshold
        }
        dropped["known_sounds"] = before - len(self.known_sounds)

        # Raw buffers are capped: hearing is not a recording device
        if len(self.sounds_heard) > max_sounds:
            dropped["sounds_heard"] = len(self.sounds_heard) - max_sounds
            self.sounds_heard = self.sounds_heard[-max_sounds:]
        if len(self.words_recognized) > max_words:
            dropped["words_recognized"] = len(self.words_recognized) - max_words
            self.words_recognized = self.words_recognized[-max_words:]
        if len(self.patterns_detected) > max_sounds:
            self.patterns_detected = self.patterns_detected[-max_sounds:]

        if any(dropped.values()):
            self._save_state()
        return dropped

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

    # ------------------------------------------------------------------
    # OPEN-22: one act of hearing, from PCM to a `what`
    # ------------------------------------------------------------------

    def _update_attention(self, band, now):
        """
        Whether attention has stayed where it was - D4's half of the rule.

        Held while the loudest band is the same band it was, released when
        it moves.  An exact exponential over real time, so the value means
        the same thing however often the microphone is polled.

        R3 applies to hearing as it does to sight: this never reaches disk
        and induction never reads it.  Focus is a property of this act of
        hearing, and the moment it is written down it starts describing
        records it did not shape.
        """
        previous, last = self._attention_band, self._attention_at
        holding = (previous is not None and last is not None
                   and band is not None and band == previous)
        dt = 0.0 if last is None else max(0.0, (now - last).total_seconds())
        tau = FOCUS_HOLD_TAU_S if holding else FOCUS_RELEASE_TAU_S
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-9))
        target = 1.0 if holding else 0.0
        self.focus_level = max(0.0, min(
            1.0, self.focus_level + alpha * (target - self.focus_level)))
        self._attention_band = band
        self._attention_at = now

    def hear_samples(self, samples, sample_rate=44100, now=None):
        """
        One act of hearing: split the mixture into streams, then describe.

        D2 fixes the order (Sussman 2017 L37 - segregate first, describe
        after), so a stream is hearing's region and `_describe_stream`
        gives it the four numbers.  The shape it returns matches what
        seeing hands to memory, which is the point: the one gate downstream
        must not have to know which sense arrived.

        D8 again - `what` is a flat list of four floats per stream, in the
        order each stream carries the mixture.  Order is priority, the same
        rule vision orders `what` by (OPEN-21 A).
        """
        import numpy as np

        if samples is None or len(samples) < FRAME_SAMPLES:
            return None

        bands = self._band_powers(samples, sample_rate)
        if bands is None:
            return None

        total = float(bands.sum())
        if total <= 0.0:
            self.what = []
            self.last_induction = []
            self._update_attention(None, datetime.now())
            return {"what": [], "streams": 0, "loudness": 0.0}

        # Attention first, exactly as vision does it: where it is has to be
        # settled before anything is let in.  The loudest band is a surround
        # difference with no category in it, so this much stays
        # pre-attentive (IO-7).  `now` is a parameter for the same reason
        # vision's is: focus is a rate over real time, and anything that has
        # to be tested has to be able to drive the clock.
        if now is None:
            now = datetime.now()
        peak_band = int(bands.mean(axis=0).argmax())
        self._update_attention(peak_band, now)

        # The one thing focus moves is how much of the mixture a stream has
        # to carry to be heard at all - D4, and the citation is Sussman 2017
        # L40: listeners doing an active task got two streams apart at a
        # *smaller frequency separation* than the same listeners did
        # passively.  Attention widens what gets in; nothing else in this
        # pipeline moves, and nothing about a record already written does
        # either (R3).
        bar = STREAM_MIN_SHARE * (
            1.0 - FOCUS_THRESHOLD_RELAXATION * self.focus_level)

        described = []
        for stream in self._segregate(bands, bar):
            carried = float(bands[:, list(stream)].sum())
            descriptor = self._describe_stream(bands, stream, sample_rate)
            if descriptor is not None:
                described.append((carried, descriptor))

        described.sort(key=lambda item: -item[0])
        what = [descriptor for _carried, descriptor in described]

        keys = []
        next_key = self._next_category
        for descriptor in what:
            key, next_key = categories.induct(
                descriptor, self.auditory_ledger, next_key,
                tau=HEARING_TAU, first_seen=datetime.now().isoformat())
            keys.append(key)
        self._next_category = next_key

        self.what = what
        self.last_induction = keys
        self._save_state()

        frames = max(len(bands), 1)
        return {
            "what": what,
            "streams": len(what),
            # The same correction `_describe_stream` needed for `level`: an
            # average across frames, or a longer snippet reads as a louder
            # one playing the very same sound.
            "loudness": float(np.sqrt(total / frames) / FRAME_SAMPLES
                              / 32768.0),
        }

    def _band_powers(self, samples, sample_rate):
        """
        Power in each cochlear band, one row per frame: (frames, bands).

        The bands are the basilar-membrane model this class always had
        (`_init_cochlea` - 24 logarithmic bands, 50 Hz to 8 kHz).  Nothing
        about the decomposition changed; what is new is that its output now
        goes somewhere.
        """
        import numpy as np
        length = int(len(samples))
        if length < FRAME_SAMPLES:
            return None

        window = np.hanning(FRAME_SAMPLES)
        freqs = np.fft.rfftfreq(FRAME_SAMPLES, 1.0 / float(sample_rate))
        # The filters as declared (centre * 0.1 + 50) leave a gap between
        # neighbours - measured, not guessed: a 3000 Hz tone sat in one and
        # only its window leakage was recorded, which is how a whistle could
        # have become silence.  The basilar membrane has no holes in it, so
        # the spectrum is cut at the geometric midpoint between consecutive
        # centres instead and every bin belongs to exactly one band.
        centres = [band["center"] for band in self.cochlea_filters]
        edges = [0.0]
        for left, right in zip(centres, centres[1:]):
            edges.append(math.sqrt(left * right))
        edges.append(float(freqs[-1]) + 1.0)
        spans = [(freqs >= low) & (freqs < high)
                 for low, high in zip(edges, edges[1:])]

        rows = []
        for start in range(0, length - FRAME_SAMPLES + 1, HOP_SAMPLES):
            frame = np.asarray(samples[start:start + FRAME_SAMPLES],
                               dtype=float) * window
            spectrum = np.abs(np.fft.rfft(frame)) ** 2
            rows.append([float(spectrum[span].sum()) for span in spans])

        return np.asarray(rows, dtype=float) if rows else None

    @staticmethod
    def _segregate(bands, min_share):
        """
        Which bands belong to which stream: the bands carrying enough of the
        mixture, grouped by sitting next to each other.

        The half of D3 this implements is Calcus 2024 L9 - "the larger the
        frequency distance between A and B, the more likely participants are
        to experience stream segregation".  A run of neighbouring bands with
        energy in them is one stream; a stretch carrying nothing is a
        frequency distance, and therefore two.

        The other half of D3, temporal coherence (Calcus 2024 L12), is NOT
        implemented, and the reason is measured rather than principled: on
        steady sounds the band envelopes are flat, so a correlation between
        them is answered by where the analysis window happened to land
        rather than by anything in the scene - two unrelated steady tones
        came out correlated.  Cutting that off needs a number the literature
        does not give, so the cue that does not need one is used instead.
        Recorded as a deviation in articles/2026-10-28-hearing-design.md.
        """
        mean_power = bands.mean(axis=0)
        total = float(mean_power.sum())
        if total <= 0.0:
            return []

        active = [index for index, value in enumerate(mean_power)
                  if value / total >= min_share]

        streams = []
        for index in active:
            if streams and index == streams[-1][-1] + 1:
                streams[-1].append(index)
            else:
                streams.append([index])
        return [tuple(group) for group in streams]

    def _describe_stream(self, bands, stream, sample_rate):
        """
        The four numbers (D7), and the paper that put each one there.

        spectral centroid - Burns 2019 L38, "the best individual descriptor"
            of both naming accuracy and complexity
        spectral flatness - Burns 2019 L21 lists it among the measures, and
            it stands in for HNR: the two are that paper's documented
            correlate (L36, r = -0.822; restated L79), while HNR needs a
            fundamental we do not estimate
        level - Burns 2019 L21's power in discrete frequency ranges,
            log-squashed against s16 full scale so a shout and a whisper are
            nearer than their raw amplitudes are
        attack - where the envelope peaks inside the window; Ettlinger 2013
            L140 has attack rise-time among the features that track timbre

        Every one of them is bounded to about 0..1 so that no single
        dimension can swamp the mean absolute difference the induction rule
        measures similarity with.
        """
        import numpy as np
        column = bands[:, list(stream)]
        power = column.sum(axis=0)
        if float(power.sum()) <= 0.0:
            return None

        centres = np.array([self.cochlea_filters[i]["center"]
                            for i in stream])
        centroid = float((centres * power).sum() / power.sum())
        # Log position across the cochlear range, not hertz over the sample
        # rate.  The ear's own map is logarithmic - _init_cochlea spaces the
        # bands that way - and dividing by the sample rate would squeeze the
        # whole audible range into the bottom fifth of this one component,
        # where 440 Hz and 3000 Hz differ by 0.058 and the induction rule
        # could not tell a hum from a whistle.
        low = self.cochlea_filters[0]["center"]
        high = min(self.cochlea_filters[-1]["center"], sample_rate / 2.0)
        pitch = math.log(max(centroid, low) / low) / math.log(high / low)
        centroid = max(0.0, min(1.0, pitch))

        # Over the bands this stream occupies.  A stream of a single band
        # comes out at 1.0 for arithmetic reasons - one value is trivially
        # its own geometric and arithmetic mean - so this carries
        # information only where a stream spans several bands, which is
        # exactly when the scene is broadband.  Said rather than hidden: it
        # separates a hiss from a tone and never separates two tones.
        floor = float(power.max()) * 1e-9 + 1e-300
        flatness = float(np.exp(np.mean(np.log(power + floor)))
                         / (float(np.mean(power)) + floor))
        # Clamped: the ratio can land a few ULPs above 1.0 when the sum
        # rounds under the numerator, and a descriptor documented as bounded
        # has to actually be bounded - the induction rule reads it either
        # way, but nothing downstream expects 1.0000000000000024.
        flatness = max(0.0, min(1.0, flatness))

        envelope = column.sum(axis=1)
        # First rise past nine tenths of the peak, not the peak itself.  On
        # an envelope with no onset in it the argmax is whichever frame won
        # by quantisation, and that number moved between two identical
        # plays of the same sound - measured, then replaced.
        peak = float(envelope.max())
        if peak <= 0.0:
            attack = 0.0
        else:
            above = np.nonzero(envelope >= 0.9 * peak)[0]
            attack = float(above[0]) / float(max(len(envelope) - 1, 1))

        # Parseval against s16 full scale: 32768 is a fact about the input
        # format, and the log squash only maps 0..1 onto 0..1 - neither is a
        # psychological quantity, so neither carries a # ASSUMPTION.  The
        # power has to be an average across frames first: summing every
        # frame pushed every loud sound to exactly 1.0, which made a shout
        # and a whisper come out as the same number.
        per_frame = float(column.sum()) / float(max(len(column), 1))
        rms = math.sqrt(2.0 * per_frame) / FRAME_SAMPLES / 32768.0
        rms = max(0.0, min(1.0, rms))
        level = math.log10(1.0 + 9.0 * rms)

        return [centroid, flatness, level, attack]
