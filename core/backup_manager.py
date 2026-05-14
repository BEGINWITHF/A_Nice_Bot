import os
import shutil
from datetime import datetime
from configs.settings import get_backup_path

def one_click_backup():
    root = get_backup_path()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(root, f"backup_{ts}")
    os.makedirs(target, exist_ok=True)
    src = "data/memory"
    dst = os.path.join(target, "memory")
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return f"Success!\n{target}"