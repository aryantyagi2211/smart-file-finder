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
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut, QFontMetrics, QCursor
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

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


def _elide_explanation(text, font, width, max_lines=2):
    metrics = QFontMetrics(font)
    words = str(text).replace("\n", " ").split()
    lines = []
    current = ""
    while words:
        word = words.pop(0)
        candidate = f"{current} {word}".strip()
        if metrics.horizontalAdvance(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and current:
        lines.append(current)
    if words and lines:
        lines[-1] = metrics.elidedText(
            lines[-1], Qt.TextElideMode.ElideRight, width
        )
    return "\n".join(lines[:max_lines])


class OptionsPopupCard(QWidget):
    """One ranked file option in the popup group."""

    def __init__(self, group, rank, option, width):
        super().__init__()
        self.group = group
        self.rank = rank
        self.option = option
        self.file_path = Path(option["path"]).resolve()
        self.setWindowTitle("Smart File Finder")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(width, group.CARD_HEIGHT)
        self._setup_ui(width)

    def _setup_ui(self, width):
        from app_paths import get_resource_path

        self.setWindowIcon(QIcon(str(get_resource_path("app_icon.ico"))))
        self.setStyleSheet("""
            OptionsPopupCard {
                background-color: rgba(18, 19, 24, 248);
                border: 1px solid rgba(255, 255, 255, 22);
                border-radius: 18px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()
        badge = QLabel(str(self.rank))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(26, 26)
        badge.setStyleSheet("""
            QLabel {
                color: #f2f2f5;
                background-color: rgba(70, 130, 240, 100);
                border-radius: 13px;
                font-weight: 700;
            }
        """)
        header.addWidget(badge)
        header.addWidget(QLabel("Not sure - pick a file"), stretch=1)
        self.close_button = QPushButton()
        self.close_button.setIcon(qta.icon("fa5s.times", color="#9a9aa2"))
        self.close_button.setFixedSize(24, 24)
        self.close_button.setStyleSheet("QPushButton { border: none; }")
        self.close_button.clicked.connect(self.group.close)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        details_row = QHBoxLayout()
        details_row.setSpacing(10)
        thumbnail = QLabel()
        thumbnail.setFixedSize(48, 48)
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.file_path.suffix.lower() in IMAGE_EXTENSIONS:
            pixmap = QPixmap(str(self.file_path))
            if not pixmap.isNull():
                thumbnail.setPixmap(pixmap.scaled(
                    48, 48, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        if thumbnail.pixmap() is None or thumbnail.pixmap().isNull():
            icon = qta.icon(
                "fa5s.image" if self.file_path.suffix.lower() in IMAGE_EXTENSIONS
                else "fa5s.file-alt",
                color="#8fb3ff",
            )
            thumbnail.setPixmap(icon.pixmap(28, 28))
        details_row.addWidget(thumbnail)

        details = QVBoxLayout()
        details.setSpacing(3)
        filename = QLabel()
        filename.setToolTip(self.file_path.name)
        filename.setStyleSheet(
            "color: #f2f2f5; font-size: 13px; font-weight: 700;"
        )
        name_width = max(40, width - 112)
        filename.setFixedWidth(name_width)
        filename.setText(QFontMetrics(filename.font()).elidedText(
            self.file_path.name, Qt.TextElideMode.ElideMiddle, name_width
        ))
        details.addWidget(filename)
        score = QLabel(f"{float(self.option['score']):.0%} match")
        score.setStyleSheet("color: #8fb3ff; font-size: 11px;")
        details.addWidget(score)
        details_row.addLayout(details, stretch=1)
        layout.addLayout(details_row)

        explanation_width = width - 32
        explanation = QLabel(_elide_explanation(
            _strip_confidence_from_explanation(
                self.option.get("explanation", "")
            ),
            self.font(),
            explanation_width,
        ))
        explanation.setToolTip(self.option.get("explanation", ""))
        explanation.setStyleSheet("color: #aaaab2; font-size: 11px;")
        explanation.setFixedHeight(32)
        layout.addWidget(explanation)

        folder = QLabel(f"…\\{self.file_path.parent.name}")
        folder.setToolTip(str(self.file_path.parent))
        folder.setStyleSheet("color: #85858e; font-size: 10px;")
        layout.addWidget(folder)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.open_button = QPushButton("Open")
        self.open_button.setStyleSheet(SECONDARY_BUTTON_STYLE)
        self.open_button.setFixedHeight(36)
        self.open_button.clicked.connect(
            lambda checked=False: self.group.choose(self.rank, "open")
        )
        self.show_button = QPushButton("Show")
        self.show_button.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.show_button.setFixedHeight(36)
        self.show_button.clicked.connect(
            lambda checked=False: self.group.choose(self.rank, "show")
        )
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.show_button)
        layout.addLayout(buttons)


class OptionsPopupGroup(QObject):
    """Creates and manages a coordinated set of ranked option cards."""

    choice_made = pyqtSignal(str, str)
    CARD_WIDTH = 340
    CARD_HEIGHT = 260
    GAP = 16

    def __init__(self, options, anchor=None):
        super().__init__()
        self.options = list(options[:3])
        self.anchor = anchor
        self.cards = []
        self._options_by_rank = {
            rank: option for rank, option in enumerate(self.options, 1)
        }
        screen = self._target_screen()
        geometry = screen.availableGeometry()
        count = len(self.options)
        gap = self.GAP if count > 1 else 0
        width = min(
            self.CARD_WIDTH,
            max(1, (geometry.width() - gap * (count - 1)) // count),
        )

        visual_options = (
            [(2, self.options[1]), (1, self.options[0]), (3, self.options[2])]
            if count == 3 else list(enumerate(self.options, 1))
        )
        for rank, option in visual_options:
            self.cards.append(OptionsPopupCard(self, rank, option, width))

        self._position_cards(screen, width, gap)
        if self.cards:
            for key, action in (("1", "open"), ("2", "open"), ("3", "open")):
                shortcut = QShortcut(QKeySequence(key), self.cards[0])
                shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
                shortcut.activated.connect(
                    lambda selected=key: self.choose(int(selected), "open")
                )
            escape = QShortcut(QKeySequence("Esc"), self.cards[0])
            escape.setContext(Qt.ShortcutContext.ApplicationShortcut)
            escape.activated.connect(self.close)

    def _target_screen(self):
        if self.anchor is not None:
            screen = self.anchor.screen()
            if screen is not None:
                return screen
        screen = QGuiApplication.screenAt(QCursor.pos())
        return screen or QApplication.primaryScreen()

    def _position_cards(self, screen, width, gap):
        geometry = screen.availableGeometry()
        count = len(self.cards)
        total_width = count * width + max(0, count - 1) * gap
        left = geometry.x() + (geometry.width() - total_width) // 2
        if self.anchor is not None:
            anchor_top = self.anchor.mapToGlobal(self.anchor.rect().topLeft()).y()
            top = anchor_top - self.CARD_HEIGHT - 16
        else:
            top = geometry.y() + (geometry.height() - self.CARD_HEIGHT) // 2
        top = max(geometry.y(), min(top, geometry.bottom() - self.CARD_HEIGHT + 1))
        for index, card in enumerate(self.cards):
            card.move(left + index * (width + gap), top)

    def show(self):
        for card in self.cards:
            card.show()
        if self.cards:
            self.cards[0].raise_()
            self.cards[0].activateWindow()

    def choose(self, rank, action):
        option = self._options_by_rank.get(rank)
        if option is None:
            return
        path = str(Path(option["path"]).resolve())
        self.close()
        self.choice_made.emit(action, path)

    def close(self):
        for card in self.cards:
            card.close()