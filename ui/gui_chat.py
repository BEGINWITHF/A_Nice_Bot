import threading
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QCursor
from ui.bubbles import BubbleWidget, NameLabel
from core.llm import chat
from core.backup_manager import one_click_backup
from configs.settings import get_backup_path, set_backup_path

class HoverButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setStyleSheet("""
            QPushButton {
                background: #2b2d34;
                color: white;
                padding: 11px;
                border-radius: 8px;
                border: none;
                font-size: 10pt;
            }
            QPushButton:hover {
                background: #4a4d55;
                padding-left: 14px;
            }
            QPushButton:pressed {
                background: #007acc;
            }
        """)

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(480, 180)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(QLabel("Backup Folder Path:"))

        self.path_edit = QLineEdit()
        self.path_edit.setText(get_backup_path())
        layout.addWidget(self.path_edit)

        btn_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        btn_layout.addWidget(browse_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_path)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Backup Folder")
        if folder:
            self.path_edit.setText(folder)

    def save_path(self):
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Warning", "Path cannot be empty!")
            return
        set_backup_path(path)
        QMessageBox.information(self, "Success", "Backup path saved!")
        self.close()

class ChatWindow(QMainWindow):
    ai_response_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diles F Chat")
        self.resize(900, 650)
        self.ai_name = "Diles F"
        self.settings_dialog = SettingsDialog(self)
        self.init_ui()
        self.ai_response_ready.connect(self.show_ai_message)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setStyleSheet("background-color:#1a1c20; border-right:1px solid #333;")
        self.sidebar.setFixedWidth(210)
        sl = QVBoxLayout(self.sidebar)
        sl.setContentsMargins(10,16,10,10)
        sl.setSpacing(8)

        title = QLabel("Diles F Chat")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:17px; color:#eee;")
        sl.addWidget(title)
        sl.addWidget(QLabel("Active User"))
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter username...")
        self.user_input.setStyleSheet("""
            QLineEdit {background:#2b2d34; color:#fff; padding:8px; border:none; border-radius:7px;}
            QLineEdit:focus {border:1px solid #007acc; background:#32343a;}
        """)
        sl.addWidget(self.user_input)

        backup_btn = HoverButton("One-Click Backup")
        backup_btn.clicked.connect(self.do_backup)
        sl.addWidget(backup_btn)

        sl.addWidget(HoverButton("Chat History"))

        self.settings_btn = HoverButton("Settings")
        self.settings_btn.clicked.connect(self.settings_dialog.show)
        sl.addWidget(self.settings_btn)

        sl.addWidget(HoverButton("About"))
        sl.addStretch()
        main_layout.addWidget(self.sidebar)

        chat_area = QVBoxLayout()
        chat_area.setContentsMargins(14,14,14,14)
        chat_area.setSpacing(10)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border:none; background:#25272c; border-radius:12px;")
        self.msg_container = QWidget()
        self.msg_layout = QVBoxLayout(self.msg_container)
        self.msg_layout.setSpacing(6)
        self.msg_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.msg_container)
        chat_area.addWidget(self.scroll)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your message...")
        self.input_box.setStyleSheet("padding:12px; border-radius:14px; background:#2b2d34; color:#fff; border:none;")
        self.send_btn = QPushButton("Send")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setStyleSheet("""
            QPushButton {background:#007acc; color:white; padding:12px 18px; border-radius:14px; border:none;}
            QPushButton:hover {background:#0088e0;}
        """)
        self.send_btn.clicked.connect(self.send_message)
        self.input_box.returnPressed.connect(self.send_message)
        input_row.addWidget(self.input_box, stretch=10)
        input_row.addWidget(self.send_btn)
        chat_area.addLayout(input_row)

        self.loading = QLabel("")
        self.loading.setAlignment(Qt.AlignCenter)
        self.loading.setStyleSheet("color:#ccc;")
        chat_area.addWidget(self.loading)

        main_layout.addLayout(chat_area, stretch=1)

    def get_username(self):
        return self.user_input.text().strip()

    def send_message(self):
        username = self.get_username()
        if not username:
            QMessageBox.warning(self, "Warning", "Please enter a username first!")
            return

        txt = self.input_box.text().strip()
        if not txt:
            return

        self.add_message(txt, is_user=True)
        self.input_box.clear()
        self.loading.setText("Diles F is thinking...")
        self.send_btn.setEnabled(False)
        threading.Thread(target=self.worker, args=(txt, username), daemon=True).start()

    def do_backup(self):
        result = one_click_backup()
        QMessageBox.information(self, "Backup Complete", result)

    def add_message(self, text, is_user):
        available_width = self.scroll.viewport().width()
        max_w = int(available_width * 0.6)

        msg_widget = QWidget()
        v_layout = QVBoxLayout(msg_widget)
        v_layout.setContentsMargins(0,0,0,10)
        v_layout.setSpacing(4)

        name = self.get_username() if is_user else self.ai_name
        v_layout.addWidget(NameLabel(name, is_user))
        v_layout.addWidget(BubbleWidget(text, is_user, max_w))

        row = QHBoxLayout()
        if is_user:
            row.addStretch()
            row.addWidget(msg_widget)
        else:
            row.addWidget(msg_widget)
            row.addStretch()

        self.msg_layout.addLayout(row)
        QTimer.singleShot(10, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def worker(self, text, username):
        try:
            reply = chat(text, username)
        except:
            reply = "Error"
        self.ai_response_ready.emit(reply)

    def show_ai_message(self, text):
        self.loading.setText("")
        self.send_btn.setEnabled(True)
        self.add_message(text, is_user=False)