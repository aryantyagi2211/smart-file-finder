"""
app/result_card.py

Displays a single search result in a clean, modern dark-UI card: icon
badge + filename + metadata at top (with a properly layered close
button), a bordered centered preview box, a "content preview" section
with the match explanation and a related-file tag, and equal-width
Open / Show in Folder buttons at the bottom -- all using real vector
icons via qtawesome (Font Awesome) instead of emoji.

The numeric confidence score is NEVER shown to the user anywhere -- it
still drives all threshold logic internally (the low-confidence confirm
dialog), just never surfaced as a percentage.
"""

import json
import os
from pathlib import Path

import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

CARD_WIDTH = 400
CARD_HEIGHT = 380

PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: rgba(70, 130, 240, 235);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
    }
    QPushButton:hover { background-color: rgba(90, 150, 255, 240); }
    QPushButton:pressed { background-color: rgba(55, 110, 210, 240); }
"""

SECONDARY_BUTTON_STYLE = """
    QPushButton {
        background-color: rgba(255, 255, 255, 16);
        color: #e8e8ec;
        border: 1px solid rgba(255, 255, 255, 40);
        border-radius: 10px;
        padding: 10px;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
    }
    QPushButton:hover { background-color: rgba(255, 255, 255, 28); }
    QPushButton:pressed { background-color: rgba(255, 255, 255, 10); }
"""


from app_paths import get_base_dir

def load_confidence_threshold(config_path=None):
    if config_path is None:
        config_path = get_base_dir() / "config" / "harness_limits.json"
    with open(config_path, encoding="utf-8-sig") as f:
        return json.load(f)["confidence_threshold"]

def _strip_confidence_from_explanation(explanation):
    """Remove any 'Matched with X% confidence — ' prefix -- the score
    itself is never displayed, only used internally for threshold logic."""
    if "confidence" in explanation and "—" in explanation:
        return explanation.split("—", 1)[-1].strip()
    return explanation


def _format_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


class ResultCard(QWidget):
    def __init__(self, file_path, score, explanation, config_path=None):
        super().__init__()
        self.file_path = file_path
        self.score = score
        self.explanation = _strip_confidence_from_explanation(explanation)
        self.threshold = load_confidence_threshold(config_path)
        self._setup_ui()

    def _setup_ui(self):
        from app_paths import get_resource_path
        self.setWindowTitle("Smart File Finder")
        self.setWindowIcon(QIcon(str(get_resource_path("app_icon.ico"))))
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(CARD_WIDTH, CARD_HEIGHT)

        container = QWidget(self)
        container.setGeometry(0, 0, CARD_WIDTH, CARD_HEIGHT)
        container.setStyleSheet("""
            background-color: rgba(18, 19, 24, 248);
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 22);
        """)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)

        path_obj = Path(self.file_path)
        ext = path_obj.suffix.lower()
        is_image = ext in IMAGE_EXTENSIONS and os.path.exists(self.file_path)

        # --- Header row: icon badge + filename + metadata + close button ---
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        badge = QLabel()
        badge.setFixedSize(38, 38)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_icon = qta.icon("fa5s.image" if is_image else "fa5s.file-alt",
                               color="#8fb3ff" if is_image else "#b8b8c0")
        badge.setPixmap(badge_icon.pixmap(20, 20))
        badge.setStyleSheet(f"""
            background-color: {"rgba(70, 130, 240, 55)" if is_image else "rgba(150, 150, 160, 35)"};
            border-radius: 10px;
        """)
        header_row.addWidget(badge)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        filename_label = QLabel(path_obj.name)
        filename_label.setStyleSheet("color: #f2f2f5; font-size: 14px; font-weight: 700;")
        filename_label.setWordWrap(True)
        title_col.addWidget(filename_label)

        meta_parts = [ext.replace(".", "").upper() or "FILE"]
        try:
            meta_parts.append(_format_size(os.path.getsize(self.file_path)))
        except OSError:
            pass
        if is_image:
            pixmap_check = QPixmap(self.file_path)
            if not pixmap_check.isNull():
                meta_parts.append(f"{pixmap_check.width()} × {pixmap_check.height()}")
        meta_label = QLabel("  •  ".join(meta_parts))
        meta_label.setStyleSheet("color: #8a8a92; font-size: 11px;")
        title_col.addWidget(meta_label)

        header_row.addLayout(title_col, stretch=1)

        close_btn = QPushButton()
        close_btn.setIcon(qta.icon("fa5s.times", color="#9a9aa2"))
        close_btn.setFixedSize(26, 26)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 14);
                border: none;
                border-radius: 13px;
            }
            QPushButton:hover { background-color: rgba(255, 80, 80, 180); }
        """)
        close_btn.clicked.connect(self.close)
        header_row.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignTop)

        layout.addLayout(header_row)

        # --- Preview box ---
        preview_frame = QFrame()
        preview_frame.setFixedHeight(230)
        preview_frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 5);
            border: 1px solid rgba(255, 255, 255, 20);
            border-radius: 14px;
        """)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        if is_image:
            thumb_label = QLabel()
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap = QPixmap(self.file_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(340, 210, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
                thumb_label.setPixmap(pixmap)
            preview_layout.addWidget(thumb_label)
        else:
            icon_label = QLabel()
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            big_icon = qta.icon("fa5s.file-alt", color="#5a5a64")
            icon_label.setPixmap(big_icon.pixmap(64, 64))
            preview_layout.addWidget(icon_label)

        layout.addWidget(preview_frame)

        # --- Buttons ---
        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        open_btn = QPushButton(" Open")
        open_btn.setIcon(qta.icon("fa5s.external-link-alt", color="#e8e8ec"))
        open_btn.setStyleSheet(SECONDARY_BUTTON_STYLE)
        open_btn.setFixedHeight(42)
        open_btn.clicked.connect(self._handle_open)

        show_btn = QPushButton(" Show in Folder")
        show_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        show_btn.setStyleSheet(PRIMARY_BUTTON_STYLE)
        show_btn.setFixedHeight(42)
        show_btn.clicked.connect(self._handle_show)

        button_row.addWidget(open_btn, stretch=1)
        button_row.addWidget(show_btn, stretch=1)
        layout.addLayout(button_row)

    def _is_low_confidence(self):
        return self.score < self.threshold

    def _confirm_low_confidence(self):
        reply = QMessageBox.question(
            self,
            "Low Confidence Match",
            "This isn't a strong match — open anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return reply == QMessageBox.StandardButton.Yes

    def _handle_open(self):
        if self._is_low_confidence() and not self._confirm_low_confidence():
            return
        try:
            absolute_path = str(Path(self.file_path).resolve())
            os.startfile(absolute_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open file: {e}")

    def _handle_show(self):
        if self._is_low_confidence() and not self._confirm_low_confidence():
            return
        try:
            folder = Path(self.file_path).resolve().parent
            os.startfile(folder)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")