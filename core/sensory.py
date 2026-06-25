"""
Sensory System - Pure sensory experience only
The AI experiences the world ONLY through hearing and seeing
No text input, no text output - like a real human baby
"""

import json
import os
from datetime import datetime
from core.hearing import HearingSystem
from core.seeing import SeeingSystem

class SensorySystem:
    """
    The AI's complete sensory system.
    Integrates hearing and seeing.
    Like a human experiencing the world through senses.
    """
    
    def __init__(self, data_dir="data/senses"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize sense systems
        self.hearing = HearingSystem(os.path.join(data_dir, "hearing"))
        self.seeing = SeeingSystem(os.path.join(data_dir, "seeing"))
        
        # Overall sensory state
        self.awareness_level = 0.3  # 0=unaware, 1=fully aware
        self.sensory_load = 0.0  # 0=no input, 1=overloaded
        self.fatigue_level = 0.0  # 0=rested, 1=exhausted
        
        # Sensory history
        self.sensory_history = []
        
        self._load_state()
    
    def _load_state(self):
        """Load sensory state"""
        state_file = os.path.join(self.data_dir, "sensory_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.awareness_level = state.get("awareness_level", 0.3)
                    self.sensory_load = state.get("sensory_load", 0.0)
                    self.fatigue_level = state.get("fatigue_level", 0.0)
            except:
                pass
    
    def _save_state(self):
        """Save sensory state"""
        state_file = os.path.join(self.data_dir, "sensory_state.json")
        state = {
            "awareness_level": self.awareness_level,
            "sensory_load": self.sensory_load,
            "fatigue_level": self.fatigue_level,
            "last_updated": datetime.now().isoformat()
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def process_sensory_input(self, input_type, data):
        """
        Process any sensory input.
        Like a brain integrating different senses.
        """
        result = None
        
        if input_type == "hearing":
            result = self.hearing.hear_sound(data.get("sound", ""), data.get("volume", 1.0))
            self.sensory_load += 0.1
        
        elif input_type == "seeing":
            result = self.seeing.see_visual(data.get("description", ""), data.get("details"))
            self.sensory_load += 0.15
        
        # Update awareness based on sensory input
        self.awareness_level = min(1.0, self.awareness_level + 0.05)
        
        # Add to history
        self.sensory_history.append({
            "type": input_type,
            "data": data,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check for sensory overload
        if self.sensory_load > 0.8:
            self.awareness_level = max(0.1, self.awareness_level - 0.2)
        
        self._save_state()
        return result
    
    def see_camera(self):
        """
        Capture and process camera input.
        Like opening your eyes.
        """
        result = self.seeing.see_camera()
        self.sensory_load += 0.2
        self.awareness_level = min(1.0, self.awareness_level + 0.1)
        self._save_state()
        return result
    
    def hear_sound(self, sound, volume=1.0):
        """
        Process hearing input.
        Like hearing a sound.
        """
        result = self.hearing.hear_sound(sound, volume)
        self.sensory_load += 0.1
        self.awareness_level = min(1.0, self.awareness_level + 0.05)
        self._save_state()
        return result
    
    def rest(self):
        """
        Rest and recover from sensory fatigue.
        Like closing your eyes and resting.
        """
        self.fatigue_level = max(0.0, self.fatigue_level - 0.3)
        self.sensory_load = max(0.0, self.sensory_load - 0.5)
        self.awareness_level = min(1.0, self.awareness_level + 0.2)
        self._save_state()
    
    def sleep_like_consolidation(self):
        """
        Sleep-like memory consolidation
        Consolidates memories from all sensory systems
        Like the brain during sleep
        """
        # Consolidate hearing memories
        hearing_consolidated = self.hearing.consolidate_auditory_memory()
        
        # Consolidate visual memories
        seeing_consolidated = self.seeing.consolidate_visual_memory()
        
        # Reset sensory fatigue
        self.fatigue_level = max(0.0, self.fatigue_level - 0.5)
        self.sensory_load = 0.0
        self.awareness_level = min(1.0, self.awareness_level + 0.3)
        
        self._save_state()
        
        return {
            "hearing_consolidated": hearing_consolidated,
            "seeing_consolidated": seeing_consolidated,
            "fatigue_reset": self.fatigue_level,
            "awareness_restored": self.awareness_level
        }
    
    def get_overall_state(self):
        """Get overall sensory state"""
        return {
            "awareness_level": self.awareness_level,
            "sensory_load": self.sensory_load,
            "fatigue_level": self.fatigue_level,
            "hearing": self.hearing.get_stats(),
            "seeing": self.seeing.get_stats()
        }
    
    def get_stats(self):
        """Get statistics"""
        return self.get_overall_state()
