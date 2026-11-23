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

from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QApplication,
)
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
        self.setFixedSize(560, 58)
        self._standard_width = 560
        self._had_error = False
        self._setup_ui()
        self._position_top_right()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        tier_row = QHBoxLayout()
        tier_row.setContentsMargins(0, 0, 0, 0)
        tier_row.setSpacing(6)
        self.tier_label = QLabel("Reading PC specifications...")
        self.tier_label.setStyleSheet("""
            color: #8fc7ff;
            font-size: 12px;
            font-weight: 700;
            background: transparent;
        """)
        tier_row.addWidget(self.tier_label)
        tier_row.addStretch(1)
        layout.addLayout(tier_row)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        self.dot = PulsingDot()
        status_row.addWidget(self.dot)
        self.status_label = QLabel("Reading RAM, graphics, and storage...")
        self.status_label.setStyleSheet("""
            color: #eaeaea;
            font-size: 12px;
            font-weight: 700;
            background: transparent;
        """)
        status_row.addWidget(self.status_label)

        self.percent_label = QLabel("")
        self.percent_label.setStyleSheet("""
            color: #6ee88a;
            font-size: 12px;
            font-weight: 700;
            background: transparent;
        """)
        status_row.addWidget(self.percent_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

    def set_performance_tier(self, tier, mode):
        from app.hardware_profile import get_tier_display_name

        selection = "automatic" if mode == "auto" else "manual"
        self.tier_label.setText(
            f"Performance: {get_tier_display_name(tier)} ({selection})"
        )
        self.tier_label.setToolTip(
            "The selected tier controls the local vision and voice models."
        )

    def show_hardware_scan(self):
        self.tier_label.setText("Reading PC specifications...")
        self.tier_label.setToolTip(
            "Checking this PC's memory, graphics, processor, and storage."
        )
        self.show_status("Reading RAM, graphics, and available storage...")

    def _position_top_right(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.x() + screen.width() - self.width() - 16
        y = screen.y() + 16
        self.move(x, y)

    def show_status(self, message):
        if self.width() != 560 or self.height() != 58:
            self.setFixedSize(560, 58)
            self._position_top_right()
        self.status_label.setText(message)
        self.percent_label.clear()

    def show_loading_model(self):
        """Distinct state for when a model is being loaded (can itself
        take a few seconds) -- separate from actively indexing files."""
        self.show_status("Loading models...")

    def show_progress(self, folder_name, filename_short, files_done, files_total):
        """Real progress shown as a count: X/Y files in the current folder."""
        self.show_status(f"Indexing ({folder_name}): {filename_short}")
        self.percent_label.setText(f"{files_done}/{files_total}")

    def show_error(self, filename_short):
        """A file failed to index -- dot turns red, text shows which file."""
        self._had_error = True
        self.dot.set_error_state(True)
        self.show_status(f"Error: {filename_short}")

    def show_notice(self, message):
        """Show a non-error indexing notice such as filename-only image mode."""
        self.show_status(message)

    def mark_complete(self):
        """Show the persistent post-scan watcher state."""
        self.dot.set_error_state(False)
        self._had_error = False
        self.show_status("Watching for changes")