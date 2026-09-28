"""
Test script for Baby AI system
Tests the core functionality without requiring Ollama

pytest runs the demo against a temporary directory and asserts on the result.
Running this file directly still prints the demo into data/test_baby.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.baby_brain import BabyBrain, DevelopmentalStage

TEST_INPUTS = [
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


def run_demo(data_dir):
    """Run the baby through ten interactions. Returns the BabyBrain."""
    print("Testing Baby Brain System...")
    print("=" * 50)

    # Create a baby brain
    baby = BabyBrain(data_dir=data_dir)

    print(f"Initial state: {baby.get_state_description()}")
    print(f"Stage: {baby.stage.name}")
    print(f"Age: {baby.age}")
    print()

    for i, input_text in enumerate(TEST_INPUTS):
        print(f"Interaction {i+1}: '{input_text}'")
        baby.internal_process(input_text, "TestUser")
        print("   Baby processes internally (no text output)")
        print(f"Current stage: {baby.stage.name}")
        print()

    print("=" * 50)
    print("Final Statistics:")
    stats = baby.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print()
    print("Test completed successfully!")
    return baby


def test_baby_brain(tmp_path):
    """The baby must actually develop: age, stage, trust, vocabulary."""
    baby = run_demo(str(tmp_path / "baby"))

    # one recorded experience per interaction
    assert baby.experience_count == len(TEST_INPUTS)
    assert baby.age == len(TEST_INPUTS)

    # stage requirements: BIRTH(1) -> SENSORY(5) -> BABBLING(15)
    # ten experiences is enough for BABBLING but not for FIRST_WORDS
    assert baby.stage == DevelopmentalStage.BABBLING
    assert baby.stage.value == 2

    # it remembers who talked to it, and trust grew with every interaction
    assert baby.known_people["TestUser"]["interactions"] == len(TEST_INPUTS)
    assert baby.trust_level == 5.0

    # words longer than two characters were learned
    assert len(baby.vocabulary) > 15

    # and all of it survives a restart
    again = BabyBrain(str(tmp_path / "baby"))
    assert again.age == baby.age
    assert again.stage == baby.stage
    assert again.experience_count == baby.experience_count
    assert len(again.vocabulary) == len(baby.vocabulary)
    assert again.known_people["TestUser"]["interactions"] == len(TEST_INPUTS)
    assert again.trust_level == baby.trust_level


if __name__ == "__main__":
    run_demo("data/test_baby")
