import os
import json

MEMORY_DIR = "data/memory"

def get_user_memory_path(username):
    return os.path.join(MEMORY_DIR, f"user_{username}.json")

def load_user_memory(username):
    path = get_user_memory_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_user_memory(username, facts):
    path = get_user_memory_path(username)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)

def get_user_facts(username):
    return load_user_memory(username)