"""
Pure AI - Sensory Experience GUI
Shows the AI's sensory experience - no text input
"""

import threading
import sys
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QCursor, QFont
from ui.bubbles import BubbleWidget, NameLabel
from core.sensory import SensorySystem
from core.pure_learning import PureLearningSystem

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

class SensoryStatusWidget(QWidget):
    """Widget showing the AI's sensory status"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.senses = SensorySystem()
        self.init_ui()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Sensory Experience")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; color: #007acc; font-weight: bold;")
        layout.addWidget(title)
        
        # Awareness
        self.awareness_label = QLabel("Awareness: 0.30")
        self.awareness_label.setStyleSheet("color: #eee; font-size: 12px;")
        layout.addWidget(self.awareness_label)
        
        # Sensory Load
        self.load_label = QLabel("Load: 0.00")
        self.load_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.load_label)
        
        # Fatigue
        self.fatigue_label = QLabel("Fatigue: 0.00")
        self.fatigue_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.fatigue_label)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #333;")
        layout.addWidget(line)
        
        # Hearing stats
        hearing_title = QLabel("Hearing")
        hearing_title.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
        layout.addWidget(hearing_title)
        
        self.hearing_label = QLabel("Sounds heard: 0")
        self.hearing_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.hearing_label)
        
        # Seeing stats
        seeing_title = QLabel("Seeing")
        seeing_title.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
        layout.addWidget(seeing_title)
        
        self.camera_label = QLabel("Camera: checking...")
        self.camera_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.camera_label)
        
        self.things_seen_label = QLabel("Things seen: 0")
        self.things_seen_label.setStyleSheet("color: #eee; font-size: 11px;")
        layout.addWidget(self.things_seen_label)
        
        layout.addStretch()
    
    def update_status(self):
        """Update the status display"""
        try:
            stats = self.senses.get_overall_state()
            
            # Update awareness
            awareness = stats.get("awareness_level", 0)
            self.awareness_label.setText(f"Awareness: {awareness:.2f}")
            
            # Update load
            load = stats.get("sensory_load", 0)
            self.load_label.setText(f"Load: {load:.2f}")
            
            # Update fatigue
            fatigue = stats.get("fatigue_level", 0)
            self.fatigue_label.setText(f"Fatigue: {fatigue:.2f}")
            
            # Update hearing
            hearing_stats = stats.get("hearing", {})
            sounds = hearing_stats.get("total_sounds_heard", 0)
            self.hearing_label.setText(f"Sounds heard: {sounds}")
            
            # Update seeing
            seeing_stats = stats.get("seeing", {})
            camera_avail = seeing_stats.get("camera_available", False)
            self.camera_label.setText(f"Camera: {'Ready' if camera_avail else 'Not available'}")
            things = seeing_stats.get("total_things_seen", 0)
            self.things_seen_label.setText(f"Things seen: {things}")
        except:
            pass

class SensoryExperienceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pure AI - Sensory Experience")
        self.resize(400, 500)
        self.senses = SensorySystem()
        self.brain = PureLearningSystem()
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Status widget
        self.status = SensoryStatusWidget()
        layout.addWidget(self.status)

        # Camera button
        self.camera_btn = HoverButton("Look Through Camera")
        self.camera_btn.clicked.connect(self.look_through_camera)
        layout.addWidget(self.camera_btn)

        # Sleep button
        self.sleep_btn = HoverButton("Sleep (Consolidate Memories)")
        self.sleep_btn.clicked.connect(self.sleep)
        layout.addWidget(self.sleep_btn)

        # Status message
        self.message_label = QLabel("")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #007acc; font-size: 12px;")
        layout.addWidget(self.message_label)

        layout.addStretch()

    def look_through_camera(self):
        """AI looks through camera"""
        self.message_label.setText("Looking through camera...")
        threading.Thread(target=self.camera_worker, daemon=True).start()

    def camera_worker(self):
        """Camera capture worker"""
        try:
            result = self.senses.see_camera()
            
            if "error" in result:
                self.message_label.setText(f"Camera error: {result['error']}")
            else:
                self.message_label.setText("I saw something new")
        except Exception as e:
            self.message_label.setText(f"Error: {str(e)}")

    def sleep(self):
        """Consolidate memories"""
        self.message_label.setText("Sleeping and consolidating memories...")
        threading.Thread(target=self.sleep_worker, daemon=True).start()

    def sleep_worker(self):
        """Sleep worker"""
        try:
            result = self.senses.sleep_like_consolidation()
            self.message_label.setText("Memories consolidated")
        except Exception as e:
            self.message_label.setText(f"Error: {str(e)}")
