import os
import threading
import time
from types import SimpleNamespace

import keyboard
import numpy as np
import pytest
from PyQt6.QtCore import QObject
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app import main
from app.voice_overlay import VoiceOverlay


class _FakeRecorder:
    SAMPLE_RATE = 16000

    def __init__(self):
        self.started = threading.Event()
        self.start_thread = None

    def start(self):
        self.start_thread = threading.get_ident()
        self.started.set()

    def stop(self):
        return np.zeros(16000, dtype=np.float32)

    def cancel(self):
        pass


class _FakeTranscriber:
    def __init__(self, transcript):
        self.transcript = transcript

    def transcribe(self, _audio):
        return self.transcript


class _FakeInputBar:
    def __init__(self):
        self.text_input = SimpleNamespace(setText=lambda _text: None)

    def hide(self):
        pass

    def show_and_focus(self):
        pass


class _FakeResultCard:
    def __init__(self, actions, **_kwargs):
        self.actions = actions

    def _handle_open(self):
        self.actions.append("open")

    def _handle_show(self):
        self.actions.append("show")

    def show(self):
        self.actions.append("find")


@pytest.mark.parametrize(
    ("transcript", "expected_action", "expected_query"),
    [
        ("open candidate", "open", "candidate"),
        ("show candidate", "show", "candidate"),
        ("find candidate", "find", "candidate"),
    ],
)
def test_alt_voice_flow_uses_main_thread_and_dispatches_spoken_command(
    monkeypatch, transcript, expected_action, expected_query
):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    qt_app = QApplication.instance() or QApplication([])
    gui_thread = threading.get_ident()
    controller = main.SmartFileFinderApp.__new__(main.SmartFileFinderApp)
    if isinstance(controller, QObject):
        QObject.__init__(controller)

    recorder = _FakeRecorder()
    overlay = VoiceOverlay(transcript_duration_ms=5000, fade_duration_ms=10)
    transcript_show_threads = []
    show_transcript = overlay.show_transcript

    def tracked_show_transcript(text):
        transcript_show_threads.append(threading.get_ident())
        show_transcript(text)

    overlay.show_transcript = tracked_show_transcript
    actions = []
    queries = []
    controller.voice_enabled = True
    controller._voice_tier = "mid"
    controller._recording = False
    controller._voice_setup_window = None
    controller._voice_overlay_active = threading.Event()
    controller._pending_voice_transcript = None
    controller._pending_voice_auto_search = True
    controller._hotkey_unregister = None
    controller._search_hotkeys_registered = False
    controller.voice_overlay = overlay
    controller.voice_recorder = recorder
    controller.voice_transcriber = _FakeTranscriber(transcript)
    controller.input_bar = _FakeInputBar()
    controller._load_voice_setting = lambda _key, default: {
        "voice_hold_ms": 30,
        "voice_key": "left alt",
        "voice_max_seconds": 15,
        "voice_auto_search": True,
    }.get(_key, default)
    controller.voice_bridge = main.VoiceBridge()
    controller.voice_bridge.state.connect(controller._on_voice_state)
    controller.voice_bridge.text.connect(controller._on_voice_text)
    controller.voice_bridge.error.connect(controller._on_voice_error)
    overlay.dismissed.connect(controller._on_voice_overlay_dismissed)

    def fake_handle_query(query):
        queries.append(query)
        return {
            "action": "answer",
            "found": True,
            "path": "candidate.txt",
            "score": 0.9,
            "explanation": "test match",
        }

    monkeypatch.setattr(main, "handle_query", fake_handle_query)
    monkeypatch.setattr(
        main, "ResultCard",
        lambda **kwargs: _FakeResultCard(actions, **kwargs),
    )
    monkeypatch.setattr(main, "missing_voice_model_for_tier", lambda _tier: None)
    monkeypatch.setattr(
        main, "get_voice_model_for_tier",
        lambda _tier: {"repo_id": "test/local-voice-model"},
    )
    monkeypatch.setattr(
        "app_paths.find_model", lambda _repo_id: "test/local-voice-model"
    )

    hook_callbacks = []
    monkeypatch.setattr(
        keyboard, "hook", lambda callback: hook_callbacks.append(callback) or callback
    )
    monkeypatch.setattr(keyboard, "unhook", lambda _hook: None)

    def send_key(name, scan_code, event_type):
        hook_callbacks[0](
            SimpleNamespace(
                name=name,
                scan_code=scan_code,
                event_type=event_type,
                is_repeat=False,
            )
        )

    def fake_inject_unassigned_key():
        # keyboard's Windows backend represents VK 0xE8 as scan_code -232
        # and a missing key name when MapVirtualKey returns no scan code.
        send_key(None, -0xE8, keyboard.KEY_DOWN)
        send_key(None, -0xE8, keyboard.KEY_UP)

    monkeypatch.setattr(
        "app.hotkey_listener.inject_unassigned_key",
        fake_inject_unassigned_key,
    )

    release_alt = threading.Event()

    def fake_key_source():
        send_key("left alt", 56, keyboard.KEY_DOWN)
        release_alt.wait(2)
        send_key("left alt", 56, keyboard.KEY_UP)

    controller._register_search_hotkeys()
    event_thread = threading.Thread(target=fake_key_source, daemon=True)
    try:
        event_thread.start()
        deadline = time.monotonic() + 2
        while not recorder.started.is_set() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.005)

        assert recorder.started.is_set(), "Alt hold did not start the recorder"
        assert recorder.start_thread == gui_thread
        qt_app.processEvents()
        assert overlay.isVisible()
        assert overlay.label.text() == "Listening..."

        release_alt.set()
        event_thread.join(timeout=1)
        deadline = time.monotonic() + 2
        while (
            overlay.label.text() != transcript
            and time.monotonic() < deadline
        ):
            qt_app.processEvents()
            time.sleep(0.005)

        assert overlay.isVisible()
        assert overlay.label.text() == transcript
        assert transcript_show_threads == [gui_thread]
        overlay.fade_out()
        QTest.qWait(100)

        assert queries == [expected_query]
        assert actions == [expected_action]
    finally:
        release_alt.set()
        event_thread.join(timeout=1)
        if controller._hotkey_unregister is not None:
            controller._hotkey_unregister()
        overlay.close()
        qt_app.processEvents()
