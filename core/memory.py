import json
import os
from configs.settings import MEMORY_FILE_PATH

# In-memory conversation history
conversation_memory = []

def initialize_memory():
    """Load memory from file on startup."""
    global conversation_memory
    try:
        if os.path.exists(MEMORY_FILE_PATH):
            with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
                conversation_memory = json.load(f)
        else:
            conversation_memory = []
    except:
        conversation_memory = []

def add_memory(role, content):
    """Add a new message to memory and save to disk."""
    conversation_memory.append({
        "role": role,
        "content": content
    })
    save_memory()

def get_memory():
    """Return full conversation history."""
    return conversation_memory

def save_memory():
    """Save memory to JSON file."""
    os.makedirs(os.path.dirname(MEMORY_FILE_PATH), exist_ok=True)
    with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(conversation_memory, f, ensure_ascii=False, indent=2)

# Auto-load memory when module is imported
initialize_memory()