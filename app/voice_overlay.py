"""Click-through, non-focusing status overlay for voice search."""

import math

from PyQt6.QtCore import QPropertyAnimation, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from observability.logger import log_voice


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

    def __init__(self, transcript_duration_ms=800, fade_duration_ms=180):
        super().__init__()
        self.transcript_duration_ms = transcript_duration_ms
        self.fade_duration_ms = fade_duration_ms
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
        self.label.setStyleSheet(
            "QLabel { color: rgba(247, 249, 255, 224); "
            "background: transparent; font-family: Segoe UI; "
            "font-size: 13px; font-weight: 500; }"
        )
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
        self._fade_animation = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_animation.finished.connect(self._finish_fade)
        self._log_overlay("created")

    @property
    def state(self):
        return self._state

    def show_listening(self):
        self._state = "listening"
        self.label.setText("Listening...")
        self.icon.listening = True
        self._pulse_timer.start()
        self._show_at_cursor_screen()
        self._fade_in()

    def show_transcribing(self):
        self._state = "transcribing"
        self._message_timer.stop()
        self._pulse_timer.stop()
        self._fade_animation.stop()
        self.icon.listening = False
        self.label.setText("Transcribing...")
        self._show_at_cursor_screen()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._log_overlay("show")

    def show_transcript(self, text):
        self._state = "transcript"
        self._pulse_timer.stop()
        self._fade_animation.stop()
        self.icon.listening = False
        self.label.setText(text)
        self._show_at_cursor_screen()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._log_overlay("show")
        self._message_timer.start(self.transcript_duration_ms)

    def show_message(self, text, duration_ms=1800):
        self._state = "message"
        self._pulse_timer.stop()
        self._fade_animation.stop()
        self.icon.listening = False
        self.label.setText(text)
        self._show_at_cursor_screen()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._log_overlay("show")
        self._message_timer.start(duration_ms)

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
        x = geometry.x() + (geometry.width() - self.width()) // 2
        y = geometry.y() + int(geometry.height() * 0.39) - self.height() // 2
        self.move(x, y)
