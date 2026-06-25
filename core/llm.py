"""
Pure AI - Internal Processing Only
No text input, no text output - like a real human's mind
"""

import os
import json
from datetime import datetime
from core.pure_learning import PureLearningSystem

# Initialize the pure learning system
pure_ai = PureLearningSystem()

def internal_process(input_text):
    """
    Process input internally - no text output
    The AI's thoughts are not readable
    """
    return pure_ai.internal_process(input_text)

def get_mood():
    """Get current mood (for internal use only)"""
    return pure_ai.human.mood

def get_energy():
    """Get current energy level (for internal use only)"""
    return pure_ai.human.energy_level

def wants_to_talk():
    """Check if AI wants to respond (free will)"""
    return pure_ai.human.should_respond()

def consolidate_memories():
    """Consolidate memories during sleep"""
    return pure_ai.human.consolidate_memories()

def sleep():
    """Sleep-like consolidation process"""
    return consolidate_memories()

def get_internal_state():
    """Get internal state (not shown to user)"""
    return {
        "mood": pure_ai.human.mood,
        "energy": pure_ai.human.energy_level,
        "vocabulary": len(pure_ai.vocabulary),
        "memories": len(pure_ai.human.interaction_memories)
    }
