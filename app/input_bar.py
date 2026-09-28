"""
app/input_bar.py

The search box UI -- a small, rounded, draggable, always-on-top floating
window with a single text input. Typing a query and pressing Enter
triggers on_submit with the raw text.

Drag behavior: a thin handle strip above the text field is the only
draggable zone -- the QLineEdit itself must own mouse clicks for cursor
placement/text selection, so dragging can't be implemented on the text
field itself without conflicting with normal typing/editing.

Default position: centered horizontally, positioned in the lower-middle
of the screen (not dead center) on first launch. Once the user drags it
elsewhere, that position is remembered for next time.
"""

from PyQt6.QtWidgets import QWidget, QLineEdit, QApplication
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtGui import QIcon

HANDLE_HEIGHT = 20


class DragHandle(QWidget):
    """Thin strip at the top of the bar -- the only draggable zone."""
    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self._drag_offset = None
        self.setFixedHeight(HANDLE_HEIGHT)
        self.setStyleSheet("""
            background-color: rgba(60, 60, 68, 235);
            border-top-left-radius: 20px;
            border-top-right-radius: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 30);
        """)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.parent_window.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.parent_window.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag_offset is not None:
            self.parent_window.remember_position()
        self._drag_offset = None


class InputBar(QWidget):
    def __init__(self, on_submit):
        super().__init__()
        self.on_submit = on_submit
        self._remembered_pos = None
        self._setup_ui()

    def _setup_ui(self):
        from app_paths import get_resource_path
        self.setWindowTitle("Smart File Finder")
        self.setWindowIcon(QIcon(str(get_resource_path("app_icon.ico"))))
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(520, 64 + HANDLE_HEIGHT)

        self.drag_handle = DragHandle(self)
        self.drag_handle.setGeometry(0, 0, 520, HANDLE_HEIGHT)

        self.text_input = QLineEdit(self)
        self.text_input.setGeometry(16, HANDLE_HEIGHT + 12, 488, 40)
        self.text_input.setPlaceholderText("Type a query, or /open, /show, /find ...")
        self.text_input.returnPressed.connect(self._handle_submit)
        self.text_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(40, 40, 45, 235);
                color: #f0f0f0;
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 20px;
                padding: 0px 18px;
                font-size: 15px;
            }
        """)

    def _default_position(self):
        """Centered horizontally, just above the taskbar."""
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + screen.height() - self.height() - 20
        return QPoint(x, y)

    def remember_position(self):
        self._remembered_pos = self.pos()

    def show_and_focus(self):
        """Bring the bar to front, at its remembered position or the
        default lower-middle position on first show."""
        target_pos = self._remembered_pos if self._remembered_pos is not None else self._default_position()
        self.move(target_pos)
        self.show()
        self.raise_()
        self.activateWindow()
        self.text_input.setFocus()

    def _handle_submit(self):
        text = self.text_input.text()
        self.on_submit(text)
        self.text_input.clear()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        """Clicking outside the window closes it, like Spotlight."""
        self.hide()
        super().focusOutEvent(event)