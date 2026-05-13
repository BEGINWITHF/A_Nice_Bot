from pynput.mouse import Controller, Button

mouse_controller = Controller()

def mouse_move(x, y):
    """Move mouse to absolute (x, y) coordinates."""
    mouse_controller.position = (x, y)

def mouse_click(x, y):
    """Move to (x, y) and perform left click."""
    mouse_controller.position = (x, y)
    mouse_controller.click(Button.left, 1)