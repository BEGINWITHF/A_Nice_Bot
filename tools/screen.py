import mss
import time
import os
from configs.settings import SCREENSHOT_SAVE_PATH

def screenshot():
    """
    Capture the full screen and save as PNG.
    Return the path to the saved image.
    """
    os.makedirs(SCREENSHOT_SAVE_PATH, exist_ok=True)
    filename = f"{SCREENSHOT_SAVE_PATH}screen_{int(time.time())}.png"

    with mss.MSS() as sct:
        screen = sct.grab(sct.monitors[1])
        mss.tools.to_png(screen.rgb, screen.size, output=filename)
    
    return filename