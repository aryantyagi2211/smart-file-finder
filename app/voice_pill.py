"""Non-focusing hold-to-talk status pill."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QWidget


class VoicePill(QWidget):
    def __init__(self, anchor=None):
        super().__init__()
        self.anchor = anchor
        self._level_index = 0
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.label = QLabel("Listening...", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            color: #f1f4f8;
            background-color: rgba(38, 43, 52, 238);
            border: 1px solid rgba(255, 255, 255, 35);
            border-radius: 16px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 700;
        """)
        self.label.resize(180, 34)
        self.setFixedSize(180, 34)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def show_listening(self):
        self._level_index = 0
        self._tick()
        self._timer.start(350)
        self._position()
        self.show()

    def show_transcribing(self):
        self._timer.stop()
        self.label.setText("Transcribing...")
        self._position()
        self.show()

    def show_error(self, message):
        self._timer.stop()
        self.label.setText(message)
        self._position()
        self.show()

    def hide_status(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        bars = ("▂", "▃", "▄", "▅", "▆")
        self.label.setText(
            f"Listening... {bars[self._level_index % len(bars)]}"
        )
        self._level_index += 1

    def _position(self):
        if self.anchor is not None and self.anchor.isVisible():
            pos = self.anchor.frameGeometry().topLeft()
            pos.setY(pos.y() - self.height() - 8)
        else:
            screen = QApplication.primaryScreen().availableGeometry()
            pos = screen.topLeft()
            pos.setX(screen.x() + (screen.width() - self.width()) // 2)
            pos.setY(screen.y() + screen.height() - 90)
        self.move(pos)
