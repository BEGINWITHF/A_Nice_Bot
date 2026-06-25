"""
Clean All Memories
One tap to erase everything - fresh start
"""

import os
import shutil

def clean_all():
    """Delete all memories and data"""
    folders_to_clean = [
        "data",
        "data/senses",
        "data/senses/hearing",
        "data/senses/seeing",
        "data/pure",
        "data/pure/human",
        "data/pure/hearing",
        "data/pure/seeing",
        "data/baby",
        "data/memory",
        "data/test_pure",
        "data/test_baby"
    ]
    
    files_to_clean = [
        "data/sensory_state.json",
        "data/senses/sensory_state.json",
        "data/senses/hearing/hearing_state.json",
        "data/senses/seeing/seeing_state.json",
        "data/pure/learning_state.json",
        "data/pure/human/human_state.json",
        "data/pure/hearing/hearing_state.json",
        "data/pure/seeing/seeing_state.json",
        "data/baby/brain_state.json"
    ]
    
    cleaned = 0
    
    # Delete files
    for filepath in files_to_clean:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                cleaned += 1
                print(f"Deleted: {filepath}")
            except:
                pass
    
    # Delete capture images
    if os.path.exists("data/senses/seeing"):
        for file in os.listdir("data/senses/seeing"):
            if file.startswith("capture_") and file.endswith(".jpg"):
                try:
                    os.remove(os.path.join("data/senses/seeing", file))
                    cleaned += 1
                except:
                    pass
    
    # Delete entire data folder (clean slate)
    if os.path.exists("data"):
        try:
            shutil.rmtree("data")
            cleaned += 1
            print("Deleted: data/ folder")
        except:
            pass
    
    return cleaned

if __name__ == "__main__":
    print("=" * 40)
    print("CLEAN ALL MEMORIES")
    print("=" * 40)
    print()
    
    confirm = input("Delete ALL memories? (yes/no): ")
    
    if confirm.lower() == "yes":
        cleaned = clean_all()
        print()
        print(f"Cleaned {cleaned} items")
        print("All memories erased.")
        print("AI starts fresh - like a newborn baby.")
    else:
        print("Cancelled.")
    
    print()
