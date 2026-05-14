from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QMessageBox, QFileDialog
from configs.settings import get_backup_path, set_backup_path

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(480, 180)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Backup Folder Path:"))

        self.path_edit = QLineEdit()
        self.path_edit.setText(get_backup_path())
        layout.addWidget(self.path_edit)

        btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_folder)
        btn_layout.addWidget(self.browse_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_path)
        btn_layout.addWidget(self.save_btn)
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