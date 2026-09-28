"""
Test script for Pure AI system
Tests learning without any pre-trained knowledge

pytest runs the demo against a temporary directory and asserts on the result.
Running this file directly still prints the demo into data/test_pure.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pure_learning import PureLearningSystem

TEST_CONVERSATIONS = [
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


def run_demo(data_dir):
    """Learn from ten conversations. Returns the PureLearningSystem."""
    print("Testing Pure AI System...")
    print("=" * 50)

    # Create a pure AI
    ai = PureLearningSystem(data_dir=data_dir)

    print("Initial state - knows nothing")
    print(f"Vocabulary: {ai.vocabulary_size} words")
    print(f"Patterns: {len(ai.patterns)}")
    print(f"Concepts: {len(ai.concepts)}")
    print()

    # Simulate learning through interaction
    for i, (input_text, _expected_context) in enumerate(TEST_CONVERSATIONS):
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
        print("AI processes internally (no text output)")
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
    return ai


def test_pure_ai(tmp_path):
    """The system must learn: vocabulary, patterns, concepts, and persist."""
    ai = run_demo(str(tmp_path / "pure"))

    # starts at 4 special tokens, must have grown
    assert ai.vocabulary_size > 4
    assert len(ai.word_to_index) == ai.vocabulary_size
    assert len(ai.index_to_word) == ai.vocabulary_size

    # word pairs were learned from the conversations
    assert len(ai.patterns) > 0
    # every conversation of two words or more produced a concept
    assert len(ai.concepts) > 0

    stats = ai.get_stats()
    assert stats["vocabulary_size"] == ai.vocabulary_size
    assert stats["patterns_learned"] == len(ai.patterns)
    assert stats["concepts_learned"] == len(ai.concepts)

    # encoding round trip: a word maps to a vector of the right size
    assert len(ai.encode_word("hello")) == 100

    # all of it survives a restart
    again = PureLearningSystem(str(tmp_path / "pure"))
    assert again.vocabulary_size == ai.vocabulary_size
    assert again.word_to_index == ai.word_to_index
    assert len(again.patterns) == len(ai.patterns)
    assert len(again.concepts) == len(ai.concepts)


if __name__ == "__main__":
    run_demo("data/test_pure")
