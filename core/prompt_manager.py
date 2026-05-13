def load_system_prompt():
    path = "configs/system_prompt.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "System prompt loading failed. You are in an abnormal state."