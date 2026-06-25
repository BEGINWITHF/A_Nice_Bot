"""
Test Sensory System (Hearing and Camera only)
"""

from core.sensory import SensorySystem
from core.pure_learning import PureLearningSystem

# Initialize
senses = SensorySystem()
brain = PureLearningSystem()

print("=== Testing Sensory System ===")
print()

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
response = brain.generate_response("hello world")
print("   I learned: hello, world")
print("   I respond:", response)
print()

print("=== Test Complete ===")
