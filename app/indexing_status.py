"""
app/indexing_status.py

Indexing status indicator: a small, frameless, borderless, top-right
corner overlay. Shows real (not estimated) progress -- current folder,
a short version of the file currently being processed, and a real X/Y
percentage -- plus a distinct "loading model" state for when a model is
being loaded (which can itself take a few seconds), and a red error
indicator when a file fails to index. This replaces the earlier
fake-climbing-percentage version after real user testing showed the fake
version gave no way to tell "still working" from "silently stuck."
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QApplication
from PyQt6.QtCore import Qt, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QColor, QPainter, QBrush
from PyQt6.QtGui import QIcon


class PulsingDot(QWidget):
    """A small circle that smoothly fades in/out -- green while working
    normally, switches to red on an error."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(12, 12)
        self._opacity = 1.0
        self._color = QColor(70, 220, 110)  # green by default

        self._animation = QPropertyAnimation(self, b"dot_opacity")
        self._animation.setDuration(900)
        self._animation.setStartValue(1.0)
        self._animation.setEndValue(0.3)
        self._animation.setLoopCount(-1)
        self._animation.finished.connect(self._reverse)
        self._animation.start()

    def _reverse(self):
        start = self._animation.endValue()
        end = self._animation.startValue()
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def get_opacity(self):
        return self._opacity

    def set_opacity(self, value):
        self._opacity = value
        self.update()

    dot_opacity = pyqtProperty(float, get_opacity, set_opacity)

    def set_error_state(self, is_error):
        self._color = QColor(230, 70, 70) if is_error else QColor(70, 220, 110)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(self._color)
        color.setAlphaF(self._opacity)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 12, 12)


class IndexingStatusWidget(QWidget):
    """
    Frameless, borderless, top-right-corner overlay shown while indexing
    runs -- icon, live status text (model loading / current file / real
    percentage), floating directly on the desktop with no box around it.
    """
    def __init__(self):
        super().__init__()
        from app_paths import get_resource_path
        self.setWindowIcon(QIcon(str(get_resource_path("app_icon.ico"))))
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(320, 30)
        self._had_error = False
        self._setup_ui()
        self._position_top_right()

    def _setup_ui(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(8)

        self.dot = PulsingDot()
        row.addWidget(self.dot)

        self.status_label = QLabel("Starting...")
        self.status_label.setStyleSheet("""
            color: #eaeaea;
            font-size: 12px;
            font-weight: 700;
            background: transparent;
        """)
        row.addWidget(self.status_label)

        self.percent_label = QLabel("")
        self.percent_label.setStyleSheet("""
            color: #6ee88a;
            font-size: 12px;
            font-weight: 700;
            background: transparent;
        """)
        row.addWidget(self.percent_label)
        row.addStretch(1)

    def _position_top_right(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.x() + screen.width() - self.width() - 16
        y = screen.y() + 16
        self.move(x, y)

    def show_loading_model(self):
        """Distinct state for when a model is being loaded (can itself
        take a few seconds) -- separate from actively indexing files."""
        self.status_label.setText("Loading model...")
        self.percent_label.setText("")

    def show_progress(self, folder_name, filename_short, files_done, files_total):
        """Real progress shown as a count: X/Y files in the current folder."""
        self.status_label.setText(f"Indexing ({folder_name}): {filename_short}")
        self.percent_label.setText(f"{files_done}/{files_total}")

    def show_error(self, filename_short):
        """A file failed to index -- dot turns red, text shows which file."""
        self._had_error = True
        self.dot.set_error_state(True)
        self.status_label.setText(f"Error: {filename_short}")

    def mark_complete(self):
        """Called once all indexing is fully done, right before this
        widget is hidden. Resets the error state for next time."""
        self.dot.set_error_state(False)
        self._had_error = False
        self.status_label.setText("Indexing complete")
        self.percent_label.setText("100%")