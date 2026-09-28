"""
Sensory system - demo script and smoke test.

IMPORTANT: the demo below used to run at module level, which meant every
pytest collection pass opened the real camera, wrote capture_*.jpg and
overwrote the real brain state in data/senses/ - while contributing zero
tests. It is now behind a __main__ guard; pytest only sees the test.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sensory import SensorySystem
from core.pure_learning import PureLearningSystem
from core.seeing import SeeingSystem


def run_demo():
    """Manual demo - uses the real camera and the real data/ directory.

    Run it with:  python test_sensory.py
    """
    print("=== Testing Sensory System ===")
    print()

    # Initialize
    senses = SensorySystem()
    brain = PureLearningSystem()

    # Hearing
    print("1. HEARING:")
    senses.hear_sound("hello", 1.0)
    print("   I heard: hello")
    senses.hear_sound("world", 0.8)
    print("   I heard: world")
    print()

    # Camera
    print("2. CAMERA:")
    if senses.seeing.camera_available:
        result = senses.see_camera()
        if "error" in result:
            print("   Error:", result["error"])
        else:
            print("   I see through camera")
            if result.get("analysis"):
                analysis = result["analysis"]
                if analysis.get("colors"):
                    print("   Colors:", ", ".join(analysis["colors"]))
                if analysis.get("objects"):
                    print("   Objects:", ", ".join(analysis["objects"]))
    else:
        print("   Camera not available (need OpenCV)")
    print()

    # Stats
    print("3. STATS:")
    stats = senses.get_stats()
    print("   Awareness:", stats["awareness_level"])
    print("   Sensory Load:", stats["sensory_load"])
    print("   Fatigue:", stats["fatigue_level"])
    print("   Hearing:", stats["hearing"]["total_sounds_heard"], "sounds heard")
    print("   Camera:", "available" if stats["seeing"]["camera_available"] else "not available")
    print()

    # Brain processing
    print("4. BRAIN PROCESSING:")
    brain.learn_word("hello")
    brain.learn_word("world")
    brain.internal_process("hello world")
    print("   I learned: hello, world")
    print("   (Internal processing - no text output)")
    print()

    print("=== Test Complete ===")


def test_sensory_smoke(tmp_path, monkeypatch):
    """Hearing and brain wiring, with hardware and real data/ neutralised."""
    monkeypatch.setattr(SeeingSystem, "_check_camera", lambda self: None)

    senses = SensorySystem(str(tmp_path / "senses"))
    brain = PureLearningSystem(str(tmp_path / "brain"))

    senses.hear_sound("hello", 1.0)
    senses.hear_sound("world", 0.8)

    stats = senses.get_stats()
    assert stats["hearing"]["total_sounds_heard"] == 2
    assert 0.0 <= stats["awareness_level"] <= 1.0
    assert stats["sensory_load"] > 0.0
    assert stats["seeing"]["camera_available"] is False

    # vocabulary starts at 4 special tokens, then learns two words
    brain.learn_word("hello")
    brain.learn_word("world")
    brain.internal_process("hello world")
    assert brain.vocabulary_size == 6
    # counted twice: once explicitly, once again through internal_process
    assert brain.word_frequency["hello"] == 2
    assert brain.word_frequency["world"] == 2


if __name__ == "__main__":
    run_demo()
