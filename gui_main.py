import sys
from PySide6.QtWidgets import QApplication
from ui.gui_chat import ChatWindow

def start_gui():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    start_gui()