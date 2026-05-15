MODEL_NAME = "deepseek-r1:14b"
SCREENSHOT_SAVE_PATH = "data/screenshots/"
MEMORY_FILE_PATH = "data/memory/history.json"
LOG_DIRECTORY = "data/logs/"

import json
import os

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
    return bk.get("backup_path", "data/backups")

def set_backup_path(p):
    bk["backup_path"] = p
    save_bk(bk)

CONFIG_FILE = "configs/app_config.json"

def load_app_config():
    os.makedirs("configs", exist_ok=True)
    if not os.path.exists(CONFIG_FILE):
        default_cfg = {
            "db_path": "data/memory/chat_history.db"
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_cfg, f, indent=2)
        return default_cfg
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db_path(path):
    cfg = load_app_config()
    cfg["db_path"] = path
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def get_db_path():
    return load_app_config().get("db_path", "data/memory/chat_history.db")