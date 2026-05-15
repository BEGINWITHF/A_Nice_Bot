import os
import json
import ollama
from configs.settings import MODEL_NAME
from core.prompt_manager import load_system_prompt
from core.memory_manager import get_user_facts, save_user_memory
from core.chat_history import get_global_history, add_global_message

MEMORY_DIR = "data/memory"
MAX_HISTORY_LENGTH = 20

if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

def chat(user_input, username):
    history = get_global_history(limit=MAX_HISTORY_LENGTH)
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

    add_global_message("user", f"[{username}] {user_input}")
    add_global_message("assistant", reply)

    return reply