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
from PyQt6.QtGui import (
    QPixmap, QIcon, QFont, QFontMetrics, QCursor, QGuiApplication, QPainter, QPainterPath,
    QPen, QColor, QMouseEvent,
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QEvent, QRectF
from PyQt6.QtWidgets import QApplication
from agent.candidate_utils import select_options_by_score
from agent.harness import load_limits

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


def _paint_card_background(widget):
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(widget.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
    path = QPainterPath()
    path.addRoundedRect(rect, 14, 14)
    painter.fillPath(path, QColor(18, 19, 24, 255))
    painter.setPen(QPen(QColor(64, 65, 73, 255), 1))
    painter.drawPath(path)


def _elide_filename(name, font, width):
    metrics = QFontMetrics(font)
    suffix = Path(name).suffix
    stem = name[:-len(suffix)] if suffix else name
    if metrics.horizontalAdvance(name) <= width:
        return name
    if not suffix or metrics.horizontalAdvance(suffix) >= width:
        return metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, width)

    middle = len(stem) // 2
    left, right = stem[:middle], stem[middle:]
    while left or right:
        candidate = f"{left}…{right}{suffix}"
        if metrics.horizontalAdvance(candidate) <= width:
            return candidate
        if len(right) >= len(left) and right:
            right = right[1:]
        elif left:
            left = left[:-1]
    return f"…{suffix}"


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
        container.setStyleSheet("background: transparent; border: none;")

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

    def paintEvent(self, event):
        _paint_card_background(self)


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
        self._drag_offset = None
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(3)

        header = QHBoxLayout()
        badge = QLabel(str(self.rank))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(26, 26)
        badge.setStyleSheet("""
            QLabel {
                color: #f2f2f5;
                font-weight: 700;
                font-size: 13px;
            }
        """)
        badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header.addWidget(badge)
        title = QLabel(
            "Not sure - is this it?"
            if len(self.group.options) == 1
            else "Not sure - pick a file"
        )
        self.header_title = title
        title.setStyleSheet("color: #f2f2f5; font-size: 12px; font-weight: 700;")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header.addWidget(title, stretch=1)
        self.close_button = QPushButton()
        self.close_button.setIcon(qta.icon("fa5s.times", color="#9a9aa2"))
        self.close_button.setFixedSize(22, 22)
        self.close_button.setStyleSheet("QPushButton { border: none; }")
        self.close_button.clicked.connect(
            lambda checked=False: self.group.close_card(self.rank)
        )
        header.addWidget(self.close_button)
        layout.addLayout(header)

        filename = QLabel()
        self.filename_label = filename
        filename.setToolTip(self.file_path.name)
        filename_font = QFont(filename.font())
        filename_font.setPixelSize(13)
        filename_font.setBold(True)
        filename.setFont(filename_font)
        filename.setStyleSheet("color: #f2f2f5;")
        name_width = width - 28
        filename.setFixedWidth(name_width)
        filename.setText(_elide_filename(
            self.file_path.name, filename.font(), name_width
        ))
        filename.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(filename)

        score = QLabel(f"{float(self.option['score']):.0%} match")
        score.setStyleSheet("color: #8fb3ff; font-size: 11px;")
        score.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(score)

        explanation = QLabel(_elide_explanation(
            _strip_confidence_from_explanation(
                self.option.get("explanation", "")
            ),
            self.font(),
            width - 28,
        ))
        explanation.setToolTip(self.option.get("explanation", ""))
        explanation.setStyleSheet("color: #aaaab2; font-size: 11px;")
        explanation.setFixedHeight(26)
        explanation.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(explanation)

        folder = QLabel(f"…\\{self.file_path.parent.name}")
        folder.setToolTip(str(self.file_path.parent))
        folder.setStyleSheet("color: #85858e; font-size: 10px;")
        folder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(folder)

        details_row = QHBoxLayout()
        details_row.setSpacing(8)
        thumbnail = QLabel()
        thumbnail.setFixedSize(30, 30)
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.file_path.suffix.lower() in IMAGE_EXTENSIONS:
            pixmap = QPixmap(str(self.file_path))
            if not pixmap.isNull():
                thumbnail.setPixmap(pixmap.scaled(
                    30, 30, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        if thumbnail.pixmap() is None or thumbnail.pixmap().isNull():
            icon = qta.icon(
                "fa5s.image" if self.file_path.suffix.lower() in IMAGE_EXTENSIONS
                else "fa5s.file-alt",
                color="#8fb3ff",
            )
            thumbnail.setPixmap(icon.pixmap(22, 22))
        thumbnail.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        details_row.addWidget(thumbnail)

        buttons = QHBoxLayout()
        buttons.setSpacing(7)
        self.open_button = QPushButton("Open")
        self.open_button.setStyleSheet(SECONDARY_BUTTON_STYLE)
        self.open_button.setFixedHeight(30)
        self.open_button.clicked.connect(
            lambda checked=False: self.group.choose(self.rank, "open")
        )
        self.show_button = QPushButton("Show")
        self.show_button.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.show_button.setFixedHeight(30)
        self.show_button.clicked.connect(
            lambda checked=False: self.group.choose(self.rank, "show")
        )
        buttons.addWidget(self.open_button, stretch=1)
        buttons.addWidget(self.show_button, stretch=1)
        details_row.addLayout(buttons, stretch=1)
        layout.addLayout(details_row)

    def paintEvent(self, event):
        _paint_card_background(self)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if (
            self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            target = event.globalPosition().toPoint() - self._drag_offset
            geometry = self.group.screen.availableGeometry()
            target.setX(max(
                geometry.left(),
                min(target.x(), geometry.right() - self.width() + 1),
            ))
            target.setY(max(
                geometry.top(),
                min(target.y(), geometry.bottom() - self.height() + 1),
            ))
            self.move(target)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event):
        self.group._card_closed()
        super().closeEvent(event)


class OptionsPopupGroup(QObject):
    """Creates and manages a coordinated set of ranked option cards."""

    choice_made = pyqtSignal(str, str)
    CARD_WIDTH = 340
    MAX_CARD_HEIGHT = 184
    GAP = 12

    def __init__(self, options, anchor=None):
        super().__init__()
        limits = load_limits()
        self.options = select_options_by_score(
            options,
            min_option_score=float(limits["min_option_score"]),
            option_score_window=float(limits["option_score_window"]),
            max_options=int(limits["max_options"]),
        )
        self.anchor = anchor
        self.cards = []
        self._app_filter_installed = False
        self._options_by_rank = {
            rank: option for rank, option in enumerate(self.options, 1)
        }
        screen = self._target_screen()
        self.screen = screen
        geometry = screen.availableGeometry()
        count = len(self.options)
        gap = min(self.GAP, max(0, (geometry.height() - 48) // max(1, count - 1)))
        self.CARD_HEIGHT = min(
            self.MAX_CARD_HEIGHT,
            max(1, (geometry.height() - 48 - gap * (count - 1)) // max(1, count)),
        )
        width = min(
            self.CARD_WIDTH,
            max(1, geometry.width() - 48),
        )

        for rank, option in enumerate(self.options, 1):
            self.cards.append(OptionsPopupCard(self, rank, option, width))

        self._position_cards(screen, width, gap)

    def _target_screen(self):
        if self.anchor is not None:
            screen = self.anchor.screen()
            if screen is not None:
                return screen
        screen = QGuiApplication.screenAt(QCursor.pos())
        return screen or QApplication.primaryScreen()

    def _position_cards(self, screen, width, gap):
        geometry = screen.availableGeometry()
        total_height = (
            len(self.cards) * self.CARD_HEIGHT
            + max(0, len(self.cards) - 1) * gap
        )
        left = geometry.x() + min(24, max(0, geometry.width() - width))
        top = geometry.y() + (geometry.height() - total_height) // 2
        for index, card in enumerate(self.cards):
            card.move(left, top + index * (self.CARD_HEIGHT + gap))

    def show(self):
        if self.cards and not self._app_filter_installed:
            QApplication.instance().installEventFilter(self)
            self._app_filter_installed = True
        for card in self.cards:
            card.show()
        if self.cards:
            self.cards[0].raise_()

    def choose(self, rank, action):
        option = self._options_by_rank.get(rank)
        if option is None:
            return
        path = str(Path(option["path"]).resolve())
        self.close()
        self.choice_made.emit(action, path)

    def close_card(self, rank):
        for card in self.cards:
            if card.rank == rank:
                card.close()
                break
        self._remove_filter_if_closed()

    def close(self):
        for card in self.cards:
            card.close()
        self._remove_filter_if_closed()

    def _card_closed(self):
        self._remove_filter_if_closed()

    def _remove_filter_if_closed(self):
        if (
            self._app_filter_installed
            and not any(card.isVisible() for card in self.cards)
        ):
            QApplication.instance().removeEventFilter(self)
            self._app_filter_installed = False

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.KeyPress or not any(
            card.isVisible() for card in self.cards
        ):
            return False
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return True
        rank = event.key() - Qt.Key.Key_0
        if rank in self._options_by_rank and 1 <= rank <= 3:
            self.choose(rank, "open")
            return True
        return False