from pynput.keyboard import Controller, Key

keyboard_controller = Controller()

def type_text(text):
    """Type given text as keyboard input."""
    keyboard_controller.type(text)

def press_enter():
    """Press and release the Enter key."""
    keyboard_controller.press(Key.enter)
    keyboard_controller.release(Key.enter)