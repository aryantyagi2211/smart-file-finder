"""
app/result_card.py

Displays confident single-file results and uncertain multi-file pick
lists using the existing dark card style and qtawesome icons.

The single-file card omits its score; the pick list shows each option's
score as a percentage.
"""

import os
from pathlib import Path

import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame
)
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut
from PyQt6.QtCore import Qt

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


def _strip_confidence_from_explanation(explanation):
    """Keep the explanation text separate from the score shown in pick lists."""
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
    def __init__(self, file_path, score, explanation):
        super().__init__()
        self.file_path = file_path
        self.score = score
        self.explanation = _strip_confidence_from_explanation(explanation)
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

    def _handle_open(self):
        try:
            absolute_path = str(Path(self.file_path).resolve())
            os.startfile(absolute_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open file: {e}")

    def _handle_show(self):
        try:
            folder = Path(self.file_path).resolve().parent
            os.startfile(folder)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")


class OptionsResultCard(QWidget):
    """Pick list shown when several files are plausible matches."""

    def __init__(self, options):
        super().__init__()
        self.options = list(options[:3])
        self._setup_ui()
        for index in range(len(self.options)):
            shortcut = QShortcut(QKeySequence(str(index + 1)), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda selected=index: self.open_option(selected))
        close_shortcut = QShortcut(QKeySequence("Esc"), self)
        close_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        close_shortcut.activated.connect(self.close)

    def _setup_ui(self):
        from app_paths import get_resource_path

        self.setWindowTitle("Smart File Finder")
        self.setWindowIcon(QIcon(str(get_resource_path("app_icon.ico"))))
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        width = 660
        height = 94 + 112 * len(self.options)
        self.setFixedSize(width, height)

        container = QWidget(self)
        container.setGeometry(0, 0, width, height)
        container.setStyleSheet("""
            background-color: rgba(18, 19, 24, 248);
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 22);
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 14, 20, 16)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Not sure - pick the right file")
        title.setStyleSheet("color: #f2f2f5; font-size: 15px; font-weight: 700;")
        header.addWidget(title, stretch=1)
        close_button = QPushButton()
        close_button.setIcon(qta.icon("fa5s.times", color="#9a9aa2"))
        close_button.setFixedSize(26, 26)
        close_button.clicked.connect(self.close)
        header.addWidget(close_button)
        layout.addLayout(header)

        for index, option in enumerate(self.options):
            self._add_option_row(layout, index, option)

    def _add_option_row(self, parent_layout, index, option):
        file_path = Path(option["path"]).resolve()
        score = float(option["score"])
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 7);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 10px;
            }
        """)
        row = QHBoxLayout(frame)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        thumbnail = QLabel()
        thumbnail.setFixedSize(54, 54)
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            pixmap = QPixmap(str(file_path))
            if not pixmap.isNull():
                thumbnail.setPixmap(pixmap.scaled(
                    54, 54, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        if thumbnail.pixmap() is None or thumbnail.pixmap().isNull():
            icon = qta.icon(
                "fa5s.image" if file_path.suffix.lower() in IMAGE_EXTENSIONS
                else "fa5s.file-alt",
                color="#8fb3ff",
            )
            thumbnail.setPixmap(icon.pixmap(26, 26))
        row.addWidget(thumbnail)

        details = QVBoxLayout()
        filename = QLabel(f"{index + 1}. {file_path.name}")
        filename.setStyleSheet("color: #f2f2f5; font-size: 13px; font-weight: 700;")
        details.addWidget(filename)
        score_label = QLabel(f"{score:.0%} match")
        score_label.setStyleSheet("color: #8fb3ff; font-size: 11px;")
        details.addWidget(score_label)
        explanation = QLabel(
            _strip_confidence_from_explanation(
                option.get("explanation", "")
            ).replace("\n", " ")
        )
        explanation.setStyleSheet("color: #aaaab2; font-size: 11px;")
        explanation.setMaximumHeight(18)
        details.addWidget(explanation)
        row.addLayout(details, stretch=1)

        open_button = QPushButton("Open")
        open_button.setStyleSheet(SECONDARY_BUTTON_STYLE)
        open_button.clicked.connect(
            lambda checked=False, selected=index: self.open_option(selected)
        )
        show_button = QPushButton("Show")
        show_button.setStyleSheet(PRIMARY_BUTTON_STYLE)
        show_button.clicked.connect(
            lambda checked=False, selected=index: self.show_option(selected)
        )
        row.addWidget(open_button)
        row.addWidget(show_button)
        parent_layout.addWidget(frame)

    def open_option(self, index):
        try:
            os.startfile(str(Path(self.options[index]["path"]).resolve()))
        except Exception as error:
            QMessageBox.warning(self, "Error", f"Could not open file: {error}")

    def show_option(self, index):
        try:
            folder = Path(self.options[index]["path"]).resolve().parent
            os.startfile(str(folder))
        except Exception as error:
            QMessageBox.warning(self, "Error", f"Could not open folder: {error}")