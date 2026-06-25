"""
Test script for Pure AI system
Tests learning without any pre-trained knowledge
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pure_learning import PureLearningSystem

def test_pure_ai():
    """Test the pure AI learning system"""
    print("Testing Pure AI System...")
    print("=" * 50)
    
    # Create a pure AI
    ai = PureLearningSystem(data_dir="data/test_pure")
    
    print("Initial state - knows nothing")
    print(f"Vocabulary: {ai.vocabulary_size} words")
    print(f"Patterns: {len(ai.patterns)}")
    print(f"Concepts: {len(ai.concepts)}")
    print()
    
    # Simulate learning through interaction
    test_conversations = [
        ("Hello", "Hi there"),
        ("What is your name?", "I am learning"),
        ("I am your teacher", "Teacher"),
        ("The sky is blue", "Blue sky"),
        ("The sun is warm", "Warm sun"),
        ("I love you", "Love"),
        ("You are smart", "Smart"),
        ("Learning is fun", "Fun learning"),
        ("Hello again", "Hello"),
        ("Goodbye", "Bye"),
    ]
    
    for i, (input_text, expected_context) in enumerate(test_conversations):
        print(f"Interaction {i+1}: '{input_text}'")
        
        # Learn from this interaction
        words = input_text.lower().split()
        for word in words:
            ai.learn_word(word)
        
        # Learn meaning from context
        if len(words) >= 2:
            for word in words:
                ai.learn_meaning(word, input_text)
        
        # Learn pattern
        ai.learn_pattern(words)
        
        # Internal processing
        ai.internal_process(input_text)
        print(f"AI processes internally (no text output)")
        print(f"Vocabulary: {ai.vocabulary_size} words")
        print()
    
    print("=" * 50)
    print("Final Statistics:")
    stats = ai.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_pure_ai()
