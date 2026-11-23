import time

import pytest
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app import voice_overlay as voice_overlay_module
from app.input_bar import BAR_HEIGHT, BAR_WIDTH, input_bar_position
from app.voice_overlay import VoiceOverlay, voice_overlay_position


def _app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def _wait_for(app, predicate, timeout_ms=2000):
    deadline = time.monotonic() + timeout_ms / 1000
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(20)
    return predicate()


def test_overlay_is_click_through_and_does_not_accept_focus(monkeypatch):
    app = _app(monkeypatch)
    overlay = VoiceOverlay()
    assert overlay.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )
    assert overlay.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert overlay.windowFlags() & Qt.WindowType.Tool
    overlay.close()
    app.processEvents()


def test_overlay_shows_listening_transcribing_transcript_and_fades(
    monkeypatch,
):
    app = _app(monkeypatch)
    overlay = VoiceOverlay(transcript_duration_ms=60, fade_duration_ms=20)
    dismissed = []
    overlay.dismissed.connect(lambda: dismissed.append(True))

    overlay.show_listening()
    app.processEvents()
    assert overlay.isVisible()
    assert overlay.label.text() == "Listening..."

    overlay.show_transcribing()
    assert overlay.label.text() == "Transcribing..."

    overlay.show_transcript("open corrective rag")
    assert overlay.label.text() == "open corrective rag"
    assert _wait_for(app, lambda: dismissed == [True])
    assert not overlay.isVisible()
    overlay.close()


def test_overlay_displays_empty_and_microphone_error_messages(monkeypatch):
    app = _app(monkeypatch)
    overlay = VoiceOverlay(fade_duration_ms=10)
    overlay.show_message("Didn't catch that", duration_ms=1000)
    app.processEvents()
    assert overlay.label.text() == "Didn't catch that"

    overlay.show_message(
        "Could not open the microphone. Check your audio device.",
        duration_ms=1000,
    )
    assert "microphone" in overlay.label.text().lower()
    overlay.fade_out()
    assert _wait_for(app, lambda: not overlay.isVisible())
    overlay.close()


def test_overlay_feedback_is_small_faded_and_dismisses(monkeypatch):
    app = _app(monkeypatch)
    overlay = VoiceOverlay(fade_duration_ms=10)
    overlay.show_feedback("Opening candidate.txt", duration_ms=30)
    app.processEvents()

    assert overlay.label.text() == "Opening candidate.txt"
    assert "font-size: 11px" in overlay.label.styleSheet()
    assert "rgba(247, 249, 255, 175)" in overlay.label.styleSheet()

    assert _wait_for(app, lambda: not overlay.isVisible())
    overlay.close()


@pytest.mark.parametrize(
    "show_state",
    [
        lambda overlay: overlay.show_listening(),
        lambda overlay: overlay.show_transcribing(),
        lambda overlay: overlay.show_transcript("open candidate"),
        lambda overlay: overlay.show_message("No match found", 5000),
        lambda overlay: overlay.show_feedback("Opening candidate.txt", 5000),
        lambda overlay: (
            overlay.show_message("Fading", 5000),
            overlay.fade_out(),
        ),
    ],
)
def test_overlay_safety_timeout_hides_every_state(monkeypatch, show_state):
    app = _app(monkeypatch)
    events = []
    monkeypatch.setattr(voice_overlay_module, "log_voice", events.append)
    overlay = VoiceOverlay(
        transcript_duration_ms=5000,
        fade_duration_ms=5000,
        safety_timeout_ms=20,
    )
    show_state(overlay)

    QTest.qWait(100)

    assert not overlay.isVisible()
    assert any("safety timeout fired" in event for event in events)
    overlay.close()
    app.processEvents()


def test_overlay_uses_taskbar_aware_input_bar_position(monkeypatch):
    screen_geometry = QRect(0, 0, 1920, 1080)
    available = QRect(0, 0, 1920, 1040)
    bar_size = QSize(BAR_WIDTH, BAR_HEIGHT)
    overlay_size = QSize(420, 132)

    bar_position = input_bar_position(available, bar_size, clamp=True)
    overlay_position = voice_overlay_position(
        available, bar_position, bar_size, overlay_size
    )

    assert bar_position == QPoint(700, 936)
    assert overlay_position == QPoint(750, 888)
    assert overlay_position.y() + overlay_size.height() == (
        bar_position.y() + bar_size.height()
    )
    assert overlay_position.y() + overlay_size.height() <= available.bottom() + 1
    assert available.height() < screen_geometry.height()


def test_overlay_uses_and_clamps_remembered_input_bar_position(monkeypatch):
    available = QRect(0, 0, 1920, 1040)
    bar_size = QSize(BAR_WIDTH, BAR_HEIGHT)
    overlay_size = QSize(420, 132)

    bar_position = input_bar_position(
        available, bar_size, QPoint(240, 120), clamp=True
    )
    overlay_position = voice_overlay_position(
        available, bar_position, bar_size, overlay_size
    )
    assert bar_position == QPoint(240, 120)
    assert overlay_position == QPoint(290, 72)

    clamped_bar_position = input_bar_position(
        available, bar_size, QPoint(-100, 1100), clamp=True
    )
    clamped_overlay_position = voice_overlay_position(
        available, clamped_bar_position, bar_size, overlay_size
    )
    assert clamped_bar_position == QPoint(0, 956)
    assert clamped_overlay_position == QPoint(50, 908)


def test_overlay_uses_cursor_screen_with_multiple_monitors():
    screens = [
        (QRect(0, 0, 1920, 1080), QRect(0, 0, 1920, 1040)),
        (QRect(1920, 0, 1280, 720), QRect(1920, 0, 1280, 680)),
    ]
    cursor_position = QPoint(2200, 120)
    screen_geometry, available = next(
        pair for pair in screens if pair[0].contains(cursor_position)
    )
    bar_size = QSize(BAR_WIDTH, BAR_HEIGHT)
    overlay_size = QSize(420, 132)
    bar_position = input_bar_position(available, bar_size, clamp=True)
    overlay_position = voice_overlay_position(
        available, bar_position, bar_size, overlay_size
    )

    assert screen_geometry == screens[1][0]
    assert available.height() < screen_geometry.height()
    assert overlay_position == QPoint(2350, 528)
    assert overlay_position != QPoint(750, 888)
