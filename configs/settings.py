"""Paths shared by the rest of the project.

Deliberately small: there is no model name here. Design clause COG-5 says the
model is one we train ourselves, so nothing in this repo points at an external
LLM. Chat-era settings (system prompt, chat DB, chat history) were removed with
the modules that used them.
"""

import json
import os

# Used by tools/screen.py (take_screenshot / take_photo).
SCREENSHOT_SAVE_PATH = "data/screenshots/"

# Backups live next to the bot's mind, outside git (IO-5).
BACKUP_CFG = "configs/backup.json"
DEF_BACKUP = {"backup_path": "data/backups"}


def load_bk():
    os.makedirs("configs", exist_ok=True)
    if not os.path.exists(BACKUP_CFG):
        with open(BACKUP_CFG, "w") as f:
            json.dump(DEF_BACKUP, f)
    with open(BACKUP_CFG) as f:
        return json.load(f)


def save_bk(d):
    with open(BACKUP_CFG, "w") as f:
        json.dump(d, f, indent=2)


bk = load_bk()


def get_backup_path():
    # `or` instead of a plain .get default: an existing empty string in
    # configs/backup.json would otherwise win and send backups to "".
    return bk.get("backup_path") or "data/backups"


def set_backup_path(p):
    bk["backup_path"] = p
    save_bk(bk)
