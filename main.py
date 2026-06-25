"""
Pure AI - Sensory Experience
The AI experiences the world through hearing and seeing (camera)
"""

from core.sensory import SensorySystem
from core.pure_learning import PureLearningSystem
from ui.cli_loader import LoadingAnimation

# Initialize systems
senses = SensorySystem()
brain = PureLearningSystem()

def main():
    """
    Main entry point.
    The AI experiences the world through hearing and camera.
    """
    print("=" * 50)
    print("PURE AI - Sensory Experience")
    print("=" * 50)
    print("I experience the world through hearing and seeing.")
    print()
    print("Commands:")
    print("  hear [sound]     - I hear a sound")
    print("  see              - I look through camera")
    print("  rest             - I rest and recover")
    print("  stats            - Show my sensory stats")
    print("  quit             - Exit")
    print("=" * 50)
    print()
    
    while True:
        user_input = input("Sense: ")
        
        if user_input.lower() == "quit":
            print("Goodbye! I will remember what I experienced.")
            break
        
        if user_input.lower() == "stats":
            stats = senses.get_stats()
            print("\n--- Sensory Statistics ---")
            print("Awareness:", stats["awareness_level"])
            print("Sensory Load:", stats["sensory_load"])
            print("Fatigue:", stats["fatigue_level"])
            print("Hearing:", stats["hearing"]["total_sounds_heard"], "sounds heard")
            print("Camera:", "available" if stats["seeing"]["camera_available"] else "not available")
            print("Things seen:", stats["seeing"]["total_things_seen"])
            print()
            continue
        
        if user_input.lower() == "rest":
            senses.rest()
            print("I rest and recover...")
            print()
            continue
        
        if user_input.lower() == "see":
            print("Looking through camera...")
            result = senses.see_camera()
            
            if "error" in result:
                print("Error:", result["error"])
            else:
                print("I see:")
                if result.get("analysis"):
                    analysis = result["analysis"]
                    if analysis.get("colors"):
                        print("  Colors:", ", ".join(analysis["colors"]))
                    if analysis.get("objects"):
                        print("  Objects:", ", ".join(analysis["objects"]))
                    print("  Brightness:", analysis.get("brightness", 0))
                print("  Saved to:", result.get("filename", "unknown"))
            print()
            continue
        
        if user_input.lower().startswith("hear "):
            sound = user_input[5:]
            result = senses.hear_sound(sound)
            print("I heard:", sound)
            
            # Process through brain
            brain.learn_word(sound)
            response = brain.generate_response(sound)
            if response:
                print("I respond:", response)
        
        else:
            # Default: treat as hearing
            result = senses.hear_sound(user_input)
            print("I heard:", user_input)
            
            # Process through brain
            brain.learn_word(user_input)
            response = brain.generate_response(user_input)
            if response:
                print("I respond:", response)
        
        print()

if __name__ == "__main__":
    main()
