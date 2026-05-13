import os
import json

MEMORY_PATH = "data/memory/user_memories.json"

def load_all_memories():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_user_memory(username, facts):
    memories = load_all_memories()
    memories[username] = facts
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)

def get_user_facts(username):
    memories = load_all_memories()
    return memories.get(username, [])