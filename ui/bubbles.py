from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics

class BubbleWidget(QWidget):
    def __init__(self, text, is_user, max_width):
        super().__init__()
        self.setMaximumWidth(max_width)
        self.setMinimumWidth(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        padding_total = 32
        safe_margin = 18
        available_width = max_width - padding_total - safe_margin

        wrapped_text = self.wrap_text(text, available_width, QFont("Arial", 11))

        label = QLabel(wrapped_text)
        label.setWordWrap(False)
        label.setFont(QFont("Arial", 11))
        label.setTextInteractionFlags(Qt.NoTextInteraction)

        if is_user:
            label.setStyleSheet("""
                QLabel {
                    background-color: #007acc;
                    color: white;
                    padding: 12px 16px;
                    border-radius: 20px;
                }
            """)
        else:
            label.setStyleSheet("""
                QLabel {
                    background-color: #3a3d44;
                    color: white;
                    padding: 12px 16px;
                    border-radius: 20px;
                }
            """)

        layout.addWidget(label)

    def wrap_text(self, text, max_line_width, font):
        fm = QFontMetrics(font)
        lines = []
        current = ""
        current_w = 0

        for c in text:
            w = fm.horizontalAdvance(c)
            if current_w + w <= max_line_width:
                current += c
                current_w += w
            else:
                lines.append(current)
                current = c
                current_w = w

        if current:
            lines.append(current)

        return "\n".join(lines)


class NameLabel(QLabel):
    def __init__(self, name, is_user):
        super().__init__(name)
        self.setFont(QFont("Arial", 9))
        self.setStyleSheet("color: #aaaaaa;")
        self.setAlignment(Qt.AlignRight if is_user else Qt.AlignLeft)