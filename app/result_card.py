"""
app/result_card.py

Displays confident single-file results and uncertain multi-file pick
lists using the existing dark card style and qtawesome icons.

The single-file card omits its score; the pick list shows each option's
score as a percentage.
"""

import os
from pathlib import Path
import subprocess

import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame
)
from PyQt6.QtGui import (
    QPixmap, QIcon, QFont, QFontMetrics, QCursor, QGuiApplication, QPainter, QPainterPath,
    QPen, QColor, QMouseEvent, QLinearGradient,
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QEvent, QRectF, QPoint
from PyQt6.QtWidgets import QApplication
from agent.candidate_utils import select_options_by_score
from agent.harness import load_limits

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

CARD_WIDTH = 450
CARD_HEIGHT = 190
POPUP_ACCENTS = ("#bd35ff", "#1498ff", "#08d985")

COMPACT_PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #0877ff;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 3px 12px;
        font-size: 11px;
        font-weight: 600;
    }
    QPushButton:hover { background-color: #258dff; }
"""

COMPACT_SECONDARY_BUTTON_STYLE = """
    QPushButton {
        background-color: rgba(255, 255, 255, 16);
        color: white;
        border: 1px solid rgba(255, 255, 255, 40);
        border-radius: 12px;
        padding: 3px 12px;
        font-size: 11px;
        font-weight: 600;
    }
    QPushButton:hover { background-color: rgba(255, 255, 255, 28); }
"""


def _strip_confidence_from_explanation(explanation):
    """Keep the explanation text separate from the score shown in pick lists."""
    if "confidence" in explanation and "—" in explanation:
        return explanation.split("—", 1)[-1].strip()
    return explanation


def _paint_card_background(widget, accent="#168aff"):
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(widget.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
    path = QPainterPath()
    path.addRoundedRect(rect, 20, 20)
    painter.fillPath(path, QColor("#08152f"))
    gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
    gradient.setColorAt(0, QColor(38, 35, 83, 190))
    gradient.setColorAt(0.55, QColor(17, 26, 56, 170))
    gradient.setColorAt(1, QColor(7, 17, 43, 220))
    painter.fillPath(path, gradient)
    for width, alpha in ((8, 24), (4, 45)):
        painter.setOpacity(alpha / 255)
        painter.setPen(QPen(QColor(accent), width))
        painter.drawPath(path)
    painter.setOpacity(1)
    painter.setPen(QPen(QColor(accent), 1.5))
    painter.drawPath(path)
    painter.end()


def _set_drag_surface(widget):
    widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def _move_popup_to_cursor(widget, anchor=None):
    screen = anchor.screen() if anchor is not None else None
    screen = screen or QGuiApplication.screenAt(QCursor.pos())
    screen = screen or QApplication.primaryScreen()
    if screen is None:
        return None
    bounds = screen.availableGeometry()
    if anchor is not None:
        center = anchor.mapToGlobal(anchor.rect().center())
        x = center.x() - widget.width() // 2
        y = anchor.mapToGlobal(QPoint(0, 0)).y() - widget.height() - 18
    else:
        x = bounds.center().x() - widget.width() // 2
        y = bounds.center().y() - widget.height() // 2
    x = max(bounds.left(), min(x, bounds.right() - widget.width() + 1))
    y = max(bounds.top(), min(y, bounds.bottom() - widget.height() + 1))
    widget.move(x, y)
    return screen


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


def open_file(file_path):
    """Open a file using its absolute path."""
    os.startfile(str(Path(file_path).resolve()))


def reveal_in_explorer(file_path):
    """Open Explorer with the specified file selected."""
    subprocess.Popen(["explorer.exe", "/select,", str(Path(file_path).resolve())])


class ResultCard(QWidget):
    """Compact confident-result card matching the ranked popup visual style."""

    def __init__(self, file_path, score, explanation, anchor=None):
        super().__init__()
        self.file_path = str(file_path)
        self.score = float(score)
        self.explanation = _strip_confidence_from_explanation(explanation)
        self.anchor = anchor
        self._drag_offset = None
        self._screen = None
        self._setup_ui()

    def _setup_ui(self):
        from app_paths import get_resource_path

        self.setWindowTitle("Smart File Finder")
        self.setWindowIcon(QIcon(str(get_resource_path("app_icon.ico"))))
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(CARD_WIDTH, CARD_HEIGHT)

        path = Path(self.file_path)
        extension = path.suffix.lower()
        is_image = extension in IMAGE_EXTENSIONS and path.is_file()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.best_badge = QLabel("Best match")
        self.best_badge.setStyleSheet(
            "color: white; background: #0877ff; border-radius: 10px;"
            "padding: 3px 10px; font-size: 11px; font-weight: 700;"
        )
        _set_drag_surface(self.best_badge)
        header.addWidget(self.best_badge)
        header.addStretch(1)
        dots = QLabel("•••")
        dots.setStyleSheet("color: #559cff; font-size: 12px;")
        _set_drag_surface(dots)
        header.addWidget(dots)
        close_button = QPushButton()
        close_button.setIcon(qta.icon("fa5s.times", color="#dce8ff"))
        close_button.setFixedSize(24, 24)
        close_button.setStyleSheet("QPushButton { border: none; }")
        close_button.clicked.connect(self.close)
        header.addWidget(close_button)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        preview = QFrame()
        preview.setFixedSize(90, 90)
        preview.setStyleSheet(
            "QFrame { background: #075ce8; border: 1px solid #168aff;"
            "border-radius: 16px; }"
        )
        _set_drag_surface(preview)
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        preview_item = QLabel()
        preview_item.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if is_image:
            pixmap = QPixmap(self.file_path)
            if not pixmap.isNull():
                preview_item.setPixmap(pixmap.scaled(
                    80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        else:
            preview_item.setPixmap(
                qta.icon("fa5s.file-alt", color="white").pixmap(48, 48)
            )
        _set_drag_surface(preview_item)
        preview_layout.addWidget(preview_item)
        body.addWidget(preview)

        details = QVBoxLayout()
        details.setSpacing(4)
        filename = QLabel()
        self.filename_label = filename
        filename.setToolTip(path.name)
        font = QFont(filename.font())
        font.setPixelSize(14)
        font.setBold(True)
        filename.setFont(font)
        filename.setStyleSheet("color: #f4f7ff;")
        filename.setText(_elide_filename(path.name, font, 280))
        _set_drag_surface(filename)
        details.addWidget(filename)

        score_row = QHBoxLayout()
        score_row.setSpacing(8)
        score_label = QLabel(f"{self.score:.0%} match")
        score_label.setStyleSheet(
            "color: white; background: #0877ff; border-radius: 9px;"
            "padding: 3px 9px; font-size: 10px; font-weight: 700;"
        )
        _set_drag_surface(score_label)
        score_row.addWidget(score_label)
        score_bar = QFrame()
        score_bar.setFixedSize(150, 7)
        score_bar.setStyleSheet("background: #25365e; border-radius: 3px;")
        score_fill = QFrame(score_bar)
        score_fill.setGeometry(
            0, 0, int(150 * max(0, min(1, self.score))), 7
        )
        score_fill.setStyleSheet("background: #27a7ff; border-radius: 3px;")
        _set_drag_surface(score_bar)
        score_row.addWidget(score_bar)
        score_row.addStretch(1)
        details.addLayout(score_row)

        explanation = QLabel(_elide_explanation(
            self.explanation or "Content matches your query closely",
            self.font(),
            280,
            max_lines=1,
        ))
        explanation.setFixedHeight(16)
        explanation.setStyleSheet("color: #b7c9e8; font-size: 11px;")
        explanation.setToolTip(self.explanation)
        _set_drag_surface(explanation)
        details.addWidget(explanation)

        folder = QLabel(f"…\\{path.parent.name}")
        folder.setToolTip(str(path.parent))
        folder.setStyleSheet("color: #b5c7e4; font-size: 10px;")
        _set_drag_surface(folder)
        details.addWidget(folder)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        open_button = QPushButton("Open")
        open_button.setStyleSheet(COMPACT_SECONDARY_BUTTON_STYLE)
        open_button.setFixedHeight(28)
        open_button.clicked.connect(self._handle_open)
        show_button = QPushButton("Show")
        show_button.setStyleSheet(COMPACT_PRIMARY_BUTTON_STYLE)
        show_button.setFixedHeight(28)
        show_button.clicked.connect(self._handle_show)
        buttons.addWidget(open_button)
        buttons.addWidget(show_button)
        details.addLayout(buttons)
        body.addLayout(details, stretch=1)
        layout.addLayout(body)
        self._screen = _move_popup_to_cursor(self, self.anchor)

    def _handle_open(self):
        try:
            open_file(self.file_path)
        except Exception as error:
            QMessageBox.warning(self, "Error", f"Could not open file: {error}")

    def _handle_show(self):
        try:
            reveal_in_explorer(self.file_path)
        except Exception as error:
            QMessageBox.warning(
                self, "Error", f"Could not show file in Explorer: {error}"
            )

    def paintEvent(self, event):
        _paint_card_background(self, "#168aff")

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
        if self._drag_offset is not None and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            target = event.globalPosition().toPoint() - self._drag_offset
            screen = self._screen or self.screen() or QApplication.primaryScreen()
            if screen is not None:
                bounds = screen.availableGeometry()
                target.setX(max(
                    bounds.left(),
                    min(target.x(), bounds.right() - self.width() + 1),
                ))
                target.setY(max(
                    bounds.top(),
                    min(target.y(), bounds.bottom() - self.height() + 1),
                ))
                self.move(target)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_offset = None
        super().mouseReleaseEvent(event)


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
    """Compact, accent-colored option card matching the supplied UI reference."""

    def __init__(self, group, rank, option, width):
        super().__init__()
        self.group = group
        self.rank = rank
        self.option = option
        self.file_path = Path(option["path"]).resolve()
        self._drag_offset = None
        self.accent = POPUP_ACCENTS[(rank - 1) % len(POPUP_ACCENTS)]
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
        layout.setContentsMargins(12, 8, 12, 9)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)
        badge = QLabel(str(self.rank))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(24, 24)
        badge.setStyleSheet(
            f"color: white; background: {self.accent}; border-radius: 12px;"
            "font-size: 12px; font-weight: 700;"
        )
        _set_drag_surface(badge)
        header.addWidget(badge)
        title = QLabel(
            "Not sure: is this it?"
            if len(self.group.options) == 1
            else "Not sure: pick a file"
        )
        self.header_title = title
        title.setStyleSheet(
            f"color: {self.accent}; font-size: 12px; font-weight: 700;"
        )
        _set_drag_surface(title)
        header.addWidget(title)
        header.addStretch(1)
        dots = QLabel("•••")
        dots.setStyleSheet(f"color: {self.accent}; font-size: 12px;")
        _set_drag_surface(dots)
        header.addWidget(dots)
        self.close_button = QPushButton()
        self.close_button.setIcon(qta.icon("fa5s.times", color="#dce8ff"))
        self.close_button.setFixedSize(23, 23)
        self.close_button.setStyleSheet("QPushButton { border: none; }")
        self.close_button.clicked.connect(
            lambda checked=False: self.group.close_card(self.rank)
        )
        header.addWidget(self.close_button)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        thumbnail = QLabel()
        thumbnail.setFixedSize(82, 82)
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbnail.setStyleSheet(
            f"background: {self.accent}; border: 1px solid {self.accent};"
            "border-radius: 15px;"
        )
        if self.file_path.suffix.lower() in IMAGE_EXTENSIONS:
            pixmap = QPixmap(str(self.file_path))
            if not pixmap.isNull():
                thumbnail.setPixmap(pixmap.scaled(
                    82, 82, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        if thumbnail.pixmap() is None or thumbnail.pixmap().isNull():
            icon_name = (
                "fa5s.file-pdf"
                if self.file_path.suffix.lower() == ".pdf"
                else "fa5s.file-alt"
            )
            thumbnail.setPixmap(
                qta.icon(icon_name, color="white").pixmap(42, 48)
            )
        _set_drag_surface(thumbnail)
        body.addWidget(thumbnail)

        details = QVBoxLayout()
        details.setSpacing(4)
        filename = QLabel()
        self.filename_label = filename
        filename.setToolTip(self.file_path.name)
        font = QFont(filename.font())
        font.setPixelSize(13)
        font.setBold(True)
        filename.setFont(font)
        filename.setStyleSheet("color: #f4f7ff;")
        filename.setFixedWidth(width - 132)
        filename.setText(_elide_filename(
            self.file_path.name, font, width - 132
        ))
        _set_drag_surface(filename)
        details.addWidget(filename)

        score_row = QHBoxLayout()
        score_row.setSpacing(8)
        score = QLabel(f"{float(self.option['score']):.0%} match")
        score.setStyleSheet(
            f"color: white; background: {self.accent}; border-radius: 9px;"
            "padding: 3px 8px; font-size: 10px; font-weight: 700;"
        )
        _set_drag_surface(score)
        score_row.addWidget(score)
        score_bar = QFrame()
        score_bar.setFixedSize(136, 7)
        score_bar.setStyleSheet("background: #25365e; border-radius: 3px;")
        score_fill = QFrame(score_bar)
        score_fill.setGeometry(
            0, 0,
            int(136 * max(0, min(1, float(self.option["score"])))),
            7,
        )
        score_fill.setStyleSheet(
            f"background: {self.accent}; border-radius: 3px;"
        )
        _set_drag_surface(score_bar)
        score_row.addWidget(score_bar)
        score_row.addStretch(1)
        details.addLayout(score_row)

        explanation_text = _strip_confidence_from_explanation(
            self.option.get("explanation", "")
        )
        explanation = QLabel(_elide_explanation(
            explanation_text or "Content is similar to your query",
            self.font(),
            width - 132,
            max_lines=1,
        ))
        explanation.setToolTip(explanation_text)
        explanation.setStyleSheet("color: #b7c9e8; font-size: 11px;")
        explanation.setFixedHeight(16)
        _set_drag_surface(explanation)
        details.addWidget(explanation)

        folder = QLabel(f"…\\{self.file_path.parent.name}")
        folder.setToolTip(str(self.file_path.parent))
        folder.setStyleSheet("color: #b5c7e4; font-size: 10px;")
        _set_drag_surface(folder)
        details.addWidget(folder)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.open_button = QPushButton("Open")
        self.open_button.setStyleSheet(COMPACT_SECONDARY_BUTTON_STYLE)
        self.open_button.setFixedSize(90, 28)
        self.open_button.clicked.connect(
            lambda checked=False: self.group.choose(self.rank, "open")
        )
        self.show_button = QPushButton("Show")
        self.show_button.setStyleSheet(
            COMPACT_PRIMARY_BUTTON_STYLE.replace("#0877ff", self.accent)
        )
        self.show_button.setFixedSize(90, 28)
        self.show_button.clicked.connect(
            lambda checked=False: self.group.choose(self.rank, "show")
        )
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.show_button)
        details.addLayout(buttons)
        body.addLayout(details, stretch=1)
        layout.addLayout(body)

    def paintEvent(self, event):
        _paint_card_background(self, self.accent)

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
    CARD_WIDTH = 400
    MAX_CARD_HEIGHT = 168
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