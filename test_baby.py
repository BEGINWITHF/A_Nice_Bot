"""
Test script for Baby AI system
Tests the core functionality without requiring Ollama
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.baby_brain import BabyBrain, DevelopmentalStage

def test_baby_brain():
    """Test the baby brain system"""
    print("Testing Baby Brain System...")
    print("=" * 50)
    
    # Create a baby brain
    baby = BabyBrain(data_dir="data/test_baby")
    
    print(f"Initial state: {baby.get_state_description()}")
    print(f"Stage: {baby.stage.name}")
    print(f"Age: {baby.age}")
    print()
    
    # Simulate some interactions
    test_inputs = [
        "Hello baby!",
        "What is your name?",
        "I am your parent.",
        "Do you understand me?",
        "Let me teach you something.",
        "The sky is blue.",
        "The sun is warm.",
        "Water is wet.",
        "Fire is hot.",
        "I love you.",
    ]
    
    for i, input_text in enumerate(test_inputs):
        print(f"Interaction {i+1}: '{input_text}'")
        response = baby.generate_response(input_text, "TestUser")
        print(f"Baby responds: '{response}'")
        print(f"Current stage: {baby.stage.name}")
        print()
    
    print("=" * 50)
    print("Final Statistics:")
    stats = baby.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_baby_brain()
