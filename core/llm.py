import os
import json
import ollama
from configs.settings import MODEL_NAME
from core.prompt_manager import load_system_prompt
from core.memory_manager import get_user_facts, save_user_memory

MEMORY_DIR = "data/memory"
GLOBAL_HISTORY_PATH = os.path.join(MEMORY_DIR, "global_chat_history.json")
MAX_HISTORY_LENGTH = 20

if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

def load_global_history():
    if os.path.exists(GLOBAL_HISTORY_PATH):
        with open(GLOBAL_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history[-MAX_HISTORY_LENGTH:]
    return []

def save_global_history(history):
    trimmed = history[-MAX_HISTORY_LENGTH:]
    with open(GLOBAL_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)

def chat(user_input, username):
    history = load_global_history()
    base_prompt = load_system_prompt()
    user_facts = get_user_facts(username)

    facts_str = "\n".join([f"- {f}" for f in user_facts]) if user_facts else "No known facts yet."
    
    full_system_prompt = f"""
{base_prompt}

You are in a group chat.
Current user: {username}

IMPORTANT FACTS ABOUT THIS USER:
{facts_str}

You have long-term memory. Always use these facts.
""".strip()

    messages = [
        {"role": "system", "content": full_system_prompt},
        *history,
        {"role": "user", "content": f"[{username}] {user_input}"}
    ]

    try:
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        raw = response["message"]["content"].strip()

        if "" in raw and "" in raw:
            parts = raw.split("</think>")
            reply = parts[-1].strip()
        else:
            reply = raw

    except Exception as e:
        return "System error."

    try:
        update_prompt = f"""This is a notecard for {username}. Update it. Keep old info if no conflict. Replace if changed. Keep short.
Current: {user_facts}
Input: {user_input}
Output only new list."""
        res = ollama.chat(model=MODEL_NAME, messages=[{"role":"user","content":update_prompt}])
        updated = res["message"]["content"].strip()
        
        if "[" in updated and "]" in updated:
            new_list = eval(updated)
        else:
            new_list = user_facts
            
        save_user_memory(username, new_list)
    except:
        pass

    history.append({"role": "user", "content": f"[{username}] {user_input}"})
    history.append({"role": "assistant", "content": reply})
    save_global_history(history)

    return reply