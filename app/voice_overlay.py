"""Click-through, non-focusing status overlay for voice search."""

import math

from PyQt6.QtCore import (
    QPropertyAnimation,
    QPoint,
    QRectF,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from app.input_bar import BAR_HEIGHT, BAR_WIDTH, input_bar_position
from observability.logger import log_voice


def voice_overlay_position(available_geometry, bar_position, bar_size,
                           overlay_size):
    """Align the overlay's center and bottom edge with the input bar."""
    x = bar_position.x() + (bar_size.width() - overlay_size.width()) // 2
    y = bar_position.y() + bar_size.height() - overlay_size.height()
    max_x = max(
        available_geometry.left(),
        available_geometry.right() - overlay_size.width() + 1,
    )
    max_y = max(
        available_geometry.top(),
        available_geometry.bottom() - overlay_size.height() + 1,
    )
    return QPoint(
        max(available_geometry.left(), min(x, max_x)),
        max(available_geometry.top(), min(y, max_y)),
    )


class _MicrophoneIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.listening = False
        self.phase = 0.0
        self.setFixedSize(72, 72)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center_x = self.width() / 2
        center_y = self.height() / 2
        scale = 1.0 + (0.045 * math.sin(self.phase) if self.listening else 0.0)

        if self.listening:
            ring_pen = QPen(QColor(246, 249, 255, 65), 2)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QRectF(center_x - 27, center_y - 27, 54, 54)
            )

        painter.save()
        painter.translate(center_x, center_y)
        painter.scale(scale, scale)
        painter.translate(-center_x, -center_y)
        painter.setPen(QPen(QColor(12, 16, 22, 185), 2))
        painter.setBrush(QColor(247, 249, 255, 225))
        painter.drawRoundedRect(
            QRectF(center_x - 8, center_y - 20, 16, 31), 8, 8
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            QRectF(center_x - 17, center_y - 10, 34, 34),
            180 * 16,
            180 * 16,
        )
        painter.drawLine(
            int(center_x), int(center_y + 24),
            int(center_x), int(center_y + 30),
        )
        painter.drawLine(
            int(center_x - 9), int(center_y + 30),
            int(center_x + 9), int(center_y + 30),
        )
        painter.restore()


class VoiceOverlay(QWidget):
    """Floating voice status that never intercepts input or takes focus."""

    dismissed = pyqtSignal()

    def __init__(
        self,
        transcript_duration_ms=800,
        fade_duration_ms=180,
        safety_timeout_ms=10000,
        input_bar=None,
    ):
        super().__init__()
        self.transcript_duration_ms = transcript_duration_ms
        self.fade_duration_ms = fade_duration_ms
        self.safety_timeout_ms = safety_timeout_ms
        self.input_bar = input_bar
        self._state = "idle"
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(420, 132)

        self.icon = _MicrophoneIcon(self)
        self.label = QLabel("", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self._set_label_style(feedback=False)
        shadow = QGraphicsDropShadowEffect(self.label)
        shadow.setBlurRadius(5)
        shadow.setOffset(0, 1)
        shadow.setColor(QColor(0, 0, 0, 220))
        self.label.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 3, 14, 3)
        layout.setSpacing(0)
        layout.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.label)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(70)
        self._pulse_timer.timeout.connect(self._pulse)
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self.fade_out)
        self._safety_timer = QTimer(self)
        self._safety_timer.setSingleShot(True)
        self._safety_timer.timeout.connect(self._safety_timeout)
        self._fade_animation = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_animation.finished.connect(self._finish_fade)
        self._log_overlay("created")

    @property
    def state(self):
        return self._state

    def show_listening(self):
        self._state = "listening"
        self._set_label_style(feedback=False)
        self.label.setText("Listening...")
        self.icon.listening = True
        self._pulse_timer.start()
        self._show_at_cursor_screen()
        self._fade_in()
        self._start_safety_timer()

    def show_transcribing(self):
        self._state = "transcribing"
        self._message_timer.stop()
        self._pulse_timer.stop()
        self._fade_animation.stop()
        self._set_label_style(feedback=False)
        self.icon.listening = False
        self.label.setText("Transcribing...")
        self._show_at_cursor_screen()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._log_overlay("show")
        self._start_safety_timer()

    def show_transcript(self, text):
        self._state = "transcript"
        self._pulse_timer.stop()
        self._fade_animation.stop()
        self._set_label_style(feedback=False)
        self.icon.listening = False
        self.label.setText(text)
        self._show_at_cursor_screen()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._log_overlay("show")
        self._message_timer.start(self.transcript_duration_ms)
        self._start_safety_timer()

    def show_message(self, text, duration_ms=1800):
        self._state = "message"
        self._pulse_timer.stop()
        self._fade_animation.stop()
        self._set_label_style(feedback=False)
        self.icon.listening = False
        self.label.setText(text)
        self._show_at_cursor_screen()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._log_overlay("show")
        self._message_timer.start(duration_ms)
        self._start_safety_timer()

    def show_feedback(self, text, duration_ms=1000):
        self._state = "feedback"
        self._pulse_timer.stop()
        self._fade_animation.stop()
        self._set_label_style(feedback=True)
        self.icon.listening = False
        self.label.setText(text)
        self._show_at_cursor_screen()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._log_overlay("show")
        self._message_timer.start(duration_ms)
        self._start_safety_timer()

    def _set_label_style(self, feedback):
        font_size = 11 if feedback else 13
        opacity = 175 if feedback else 224
        self.label.setStyleSheet(
            "QLabel { color: rgba(247, 249, 255, "
            f"{opacity}); background: transparent; font-family: Segoe UI; "
            f"font-size: {font_size}px; font-weight: 500; }}"
        )

    def _start_safety_timer(self):
        self._safety_timer.start(self.safety_timeout_ms)

    def _safety_timeout(self):
        if not self.isVisible():
            return
        log_voice(f"overlay safety timeout fired state={self._state}")
        self._message_timer.stop()
        self._pulse_timer.stop()
        self._fade_animation.stop()
        self._safety_timer.stop()
        self.hide()
        self._state = "idle"
        self.dismissed.emit()

    def fade_out(self):
        if not self.isVisible():
            return
        self._message_timer.stop()
        self._pulse_timer.stop()
        self._state = "fading"
        self._fade_animation.stop()
        self._fade_animation.setDuration(self.fade_duration_ms)
        self._fade_animation.setStartValue(self.windowOpacity())
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()
        self._log_overlay("fade_started")

    def _fade_in(self):
        self._message_timer.stop()
        self._fade_animation.stop()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self._log_overlay("show")
        self._fade_animation.setDuration(self.fade_duration_ms)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()

    def _finish_fade(self):
        if self._state != "fading":
            return
        self._safety_timer.stop()
        self.hide()
        self._state = "idle"
        self.dismissed.emit()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._log_overlay("hide")

    def _pulse(self):
        self.icon.phase += 0.28
        self.icon.update()

    def _log_overlay(self, event):
        app = QApplication.instance()
        cursor = QCursor.pos()
        screen = (
            app.screenAt(cursor) or app.primaryScreen()
            if app is not None
            else None
        )
        screen_name = screen.name() if screen is not None else None
        screen_geometry = (
            screen.availableGeometry().getRect()
            if screen is not None
            else None
        )
        log_voice(
            f"overlay {event} state={self._state} "
            f"geometry={self.geometry().getRect()} screen={screen_name!r} "
            f"screen_geometry={screen_geometry} "
            f"opacity={self.windowOpacity():.3f} "
            f"isVisible={self.isVisible()}"
        )

    def _show_at_cursor_screen(self):
        position = QCursor.pos()
        screen = QApplication.screenAt(position) or QApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        if self.input_bar is None:
            bar_size = QSize(BAR_WIDTH, BAR_HEIGHT)
            bar_position = input_bar_position(
                geometry, bar_size, clamp=True
            )
        else:
            bar_size = self.input_bar.size()
            bar_position = self.input_bar.position_for_screen(
                screen, clamp=True
            )
        overlay_position = voice_overlay_position(
            geometry, bar_position, bar_size, self.size()
        )
        self.move(overlay_position)
