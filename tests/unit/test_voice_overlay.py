from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.voice_overlay import VoiceOverlay


def _app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


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
    overlay = VoiceOverlay(transcript_duration_ms=30, fade_duration_ms=20)
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
    QTest.qWait(100)
    assert dismissed == [True]
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
    QTest.qWait(30)
    assert not overlay.isVisible()
    overlay.close()
