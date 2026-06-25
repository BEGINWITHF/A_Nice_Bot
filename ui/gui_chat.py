"""
Pure AI Chat Interface
Shows the AI's learning progress and allows interaction
No pre-trained models - just pure learning
"""

import threading
import sys
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QCursor, QFont
from ui.bubbles import BubbleWidget, NameLabel
from core.llm import chat, get_ai_state, get_ai_stats
from core.backup_manager import one_click_backup
from configs.settings import get_backup_path, set_backup_path, get_db_path, save_db_path

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

class PureAIStatusWidget(QWidget):
    """Widget showing the pure AI's learning status"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)  # Update every second
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Pure AI Status")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; color: #007acc; font-weight: bold;")
        layout.addWidget(title)
        
        # Mood
        self.mood_label = QLabel("Mood: neutral")
        self.mood_label.setStyleSheet("color: #eee; font-size: 12px;")
        layout.addWidget(self.mood_label)
        
        # Energy
        self.energy_label = QLabel("Energy: normal")
        self.energy_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.energy_label)
        
        # Want to talk
        self.talk_label = QLabel("Wants to talk: yes")
        self.talk_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.talk_label)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #333;")
        layout.addWidget(line)
        
        # Stats
        stats_title = QLabel("Learning Statistics")
        stats_title.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
        layout.addWidget(stats_title)
        
        self.vocab_label = QLabel("Vocabulary: 0 words")
        self.vocab_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.vocab_label)
        
        self.patterns_label = QLabel("Patterns: 0 learned")
        self.patterns_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.patterns_label)
        
        self.concepts_label = QLabel("Concepts: 0 understood")
        self.concepts_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.concepts_label)
        
        self.interactions_label = QLabel("Interactions: 0")
        self.interactions_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.interactions_label)
        
        self.knowledge_label = QLabel("Knowledge: 0.00%")
        self.knowledge_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.knowledge_label)
        
        layout.addStretch()
    
    def update_status(self):
        """Update the status display"""
        try:
            stats = get_ai_stats()
            human_stats = stats.get("human_like", {})
            
            # Update mood
            mood = human_stats.get("mood", "neutral")
            mood_desc = human_stats.get("mood_description", "calm")
            self.mood_label.setText(f"Mood: {mood} ({mood_desc})")
            
            # Update energy
            energy = human_stats.get("energy_level", 0.5)
            energy_desc = human_stats.get("energy_description", "normal")
            self.energy_label.setText(f"Energy: {energy_desc} ({energy:.0%})")
            
            # Update want to talk
            want_to_talk = human_stats.get("want_to_talk", True)
            self.talk_label.setText(f"Wants to talk: {'yes' if want_to_talk else 'no'}")
            
            # Update learning stats
            self.vocab_label.setText(f"Vocabulary: {stats['vocabulary_size']} words")
            self.patterns_label.setText(f"Patterns: {stats['patterns_learned']} learned")
            self.concepts_label.setText(f"Concepts: {stats['concepts_learned']} understood")
            self.interactions_label.setText(f"Interactions: {stats['total_interactions']}")
            
            knowledge_pct = stats['knowledge_level'] * 100
            self.knowledge_label.setText(f"Knowledge: {knowledge_pct:.2f}%")
        except:
            pass

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(580, 260)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(16)

        layout.addWidget(QLabel("Backup Folder Path:"))
        backup_row = QHBoxLayout()
        self.backup_edit = QLineEdit()
        self.backup_edit.setText(get_backup_path())
        backup_row.addWidget(self.backup_edit)
        browse_backup = QPushButton("Browse")
        browse_backup.clicked.connect(self.browse_backup_folder)
        backup_row.addWidget(browse_backup)
        layout.addLayout(backup_row)

        layout.addWidget(QLabel("Global Chat Database Path:"))
        db_row = QHBoxLayout()
        self.db_edit = QLineEdit()
        self.db_edit.setText(get_db_path())
        db_row.addWidget(self.db_edit)
        browse_db = QPushButton("Browse")
        browse_db.clicked.connect(self.browse_db_file)
        db_row.addWidget(browse_db)
        layout.addLayout(db_row)

        save_row = QHBoxLayout()
        save_row.addStretch()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_all_settings)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)

    def browse_backup_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Backup Folder")
        if folder:
            self.backup_edit.setText(folder)

    def browse_db_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select DB File", get_db_path(), "SQLite DB (*.db)")
        if path:
            self.db_edit.setText(path)

    def save_all_settings(self):
        backup_path = self.backup_edit.text().strip()
        db_path = self.db_edit.text().strip()

        if not backup_path or not db_path:
            QMessageBox.warning(self, "Warning", "Paths cannot be empty!")
            return

        set_backup_path(backup_path)
        save_db_path(db_path)

        choice = QMessageBox.question(
            self,
            "Settings Saved",
            "Settings have been updated!\n\n"
            "Save & Restart: Apply changes immediately\n"
            "Cancel: Exit without changes",
            QMessageBox.Yes | QMessageBox.Cancel
        )

        if choice == QMessageBox.Yes:
            os.execl(sys.executable, sys.executable, *sys.argv)
        else:
            self.close()

class ChatWindow(QMainWindow):
    ai_response_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pure AI - Sensory Experience")
        self.resize(1000, 700)
        self.ai_name = "Pure AI"
        self.settings_dialog = SettingsDialog(self)
        self.init_ui()
        self.ai_response_ready.connect(self.show_ai_message)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # Left sidebar - AI status
        self.sidebar = QFrame()
        self.sidebar.setStyleSheet("background-color:#1a1c20; border-right:1px solid #333;")
        self.sidebar.setFixedWidth(220)
        sl = QVBoxLayout(self.sidebar)
        sl.setContentsMargins(10,16,10,10)
        sl.setSpacing(8)

        # AI status widget
        self.ai_status = PureAIStatusWidget()
        sl.addWidget(self.ai_status)

        # User input
        sl.addWidget(QLabel("Your Name (Teacher):"))
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter your name...")
        self.user_input.setStyleSheet("""
            QLineEdit {background:#2b2d34; color:#fff; padding:8px; border:none; border-radius:7px;}
            QLineEdit:focus {border:1px solid #007acc; background:#32343a;}
        """)
        sl.addWidget(self.user_input)

        # Camera button
        self.camera_btn = HoverButton("Look Through Camera")
        self.camera_btn.clicked.connect(self.look_through_camera)
        sl.addWidget(self.camera_btn)

        # Buttons
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

        # Chat area
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
        self.input_box.setPlaceholderText("Say something for the AI to hear...")
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

    def look_through_camera(self):
        """AI looks through camera"""
        self.loading.setText("Looking through camera...")
        threading.Thread(target=self.camera_worker, daemon=True).start()

    def camera_worker(self):
        """Camera capture worker"""
        try:
            from core.sensory import SensorySystem
            senses = SensorySystem()
            result = senses.see_camera()
            
            if "error" in result:
                reply = f"Camera error: {result['error']}"
            else:
                analysis = result.get("analysis", {})
                colors = ", ".join(analysis.get("colors", [])) if analysis.get("colors") else "none"
                objects = ", ".join(analysis.get("objects", [])) if analysis.get("objects") else "none"
                reply = f"I see: colors({colors}), objects({objects})"
            
            self.ai_response_ready.emit(reply)
        except Exception as e:
            self.ai_response_ready.emit(f"Error: {str(e)}")

    def get_username(self):
        return self.user_input.text().strip()

    def send_message(self):
        username = self.get_username()
        if not username:
            QMessageBox.warning(self, "Warning", "Please enter your name first!")
            return

        txt = self.input_box.text().strip()
        if not txt:
            return

        self.add_message(txt, is_user=True)
        self.input_box.clear()
        self.loading.setText("AI is learning...")
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
        except Exception as e:
            reply = f"Error: {str(e)}"
        self.ai_response_ready.emit(reply)

    def show_ai_message(self, text):
        self.loading.setText("")
        self.send_btn.setEnabled(True)
        self.add_message(text, is_user=False)
