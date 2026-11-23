import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import keyboard
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from agent.planner import Candidate
from app import main, query_handler, result_card
from app.input_bar import InputBar
from app.voice_overlay import VoiceOverlay


class _FakeRecorder:
    SAMPLE_RATE = 16000

    def __init__(self):
        self.started = threading.Event()
        self.start_thread = None
        self.cancel_calls = 0

    def start(self):
        self.start_thread = threading.get_ident()
        self.started.set()

    def stop(self):
        return np.zeros(16000, dtype=np.float32)

    def cancel(self):
        self.cancel_calls += 1


class _FakeTranscriber:
    def __init__(self, transcript):
        self.transcript = transcript

    def transcribe(self, _audio):
        return self.transcript


class _FakeSignal:
    def connect(self, callback):
        self.callback = callback


class _FakeOptionsGroup:
    def __init__(self, options, anchor, actions):
        self.options = options
        self.anchor = anchor
        self.choice_made = _FakeSignal()
        self.actions = actions

    def show(self):
        self.actions.append(("options", self.options))


@pytest.mark.parametrize(
    ("transcript", "expected_query", "expected_overlay_text", "expected_action"),
    [
        ("open candidate", "candidate", "Opening candidate.txt", "open"),
        (
            "show candidate",
            "candidate",
            "Showing candidate.txt in Explorer",
            "show",
        ),
        ("find candidate", "candidate", "find candidate", "find"),
        (
            "open ambiguous",
            "ambiguous",
            "Not sure which file - pick one",
            "options",
        ),
        ("open missing", "missing", "No match found", "no_match"),
    ],
)
def test_alt_voice_flow_dispatches_through_real_query_handler(
    monkeypatch,
    tmp_path,
    transcript,
    expected_query,
    expected_overlay_text,
    expected_action,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qt_app = QApplication.instance() or QApplication([])
    gui_thread = threading.get_ident()
    controller = main.SmartFileFinderApp.__new__(main.SmartFileFinderApp)

    candidate_path = tmp_path / "candidate.txt"
    other_path = tmp_path / "candidate-notes.txt"
    candidate_path.write_text("candidate result", encoding="utf-8")
    other_path.write_text("another candidate", encoding="utf-8")

    retrieval_queries = []

    def stub_index(query):
        retrieval_queries.append(query)
        if query == "ambiguous":
            return [
                Candidate(str(candidate_path), 0.70, {"chunk_text": "candidate"}),
                Candidate(str(other_path), 0.68, {"chunk_text": "candidate"}),
            ]
        if query == "missing":
            return [
                Candidate(str(candidate_path), 0.20, {"chunk_text": "candidate"})
            ]
        return [
            Candidate(str(candidate_path), 0.95, {"chunk_text": "candidate"})
        ]

    monkeypatch.setattr(query_handler, "retrieve", stub_index)

    recorder = _FakeRecorder()
    input_bar = InputBar(lambda _text: None)
    overlay = VoiceOverlay(
        transcript_duration_ms=5000,
        fade_duration_ms=10,
        input_bar=input_bar,
    )
    transcript_show_threads = []
    show_transcript = overlay.show_transcript

    def tracked_show_transcript(text):
        transcript_show_threads.append(threading.get_ident())
        show_transcript(text)

    overlay.show_transcript = tracked_show_transcript
    actions = []
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
    controller.input_bar = input_bar
    controller.result_card = None
    controller.options_group = None
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

    monkeypatch.setattr(
        main,
        "OptionsPopupGroup",
        lambda options, anchor: _FakeOptionsGroup(options, anchor, actions),
    )
    monkeypatch.setattr(
        main.ResultCard,
        "show",
        lambda card: actions.append(("find", card.file_path)),
    )
    monkeypatch.setattr(
        result_card.os,
        "startfile",
        lambda path: actions.append(("open", str(path))),
        raising=False,
    )
    monkeypatch.setattr(
        result_card.subprocess,
        "Popen",
        lambda args: actions.append(("show", args)),
    )
    monkeypatch.setattr(
        main.QMessageBox,
        "information",
        lambda *_args: actions.append(("no_match_message", None)),
    )
    monkeypatch.setattr(main, "missing_voice_model_for_tier", lambda _tier: None)
    monkeypatch.setattr(
        main,
        "get_voice_model_for_tier",
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
        send_key(None, -0xE8, keyboard.KEY_DOWN)
        send_key(None, -0xE8, keyboard.KEY_UP)

    monkeypatch.setattr(
        "app.hotkey_listener.inject_unassigned_key",
        fake_inject_unassigned_key,
    )

    release_alt = threading.Event()
    repeated_alt_sent = threading.Event()

    def fake_key_source():
        send_key("left alt", 56, keyboard.KEY_DOWN)
        if controller._voice_overlay_active.wait(2):
            send_key("left alt", 56, keyboard.KEY_DOWN)
            repeated_alt_sent.set()
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
        deadline = time.monotonic() + 2
        while not repeated_alt_sent.is_set() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.005)
        assert repeated_alt_sent.is_set()
        qt_app.processEvents()
        assert recorder.cancel_calls == 0
        assert recorder.start_thread == gui_thread
        assert overlay.isVisible()
        assert overlay.label.text() == "Listening..."

        release_alt.set()
        event_thread.join(timeout=1)
        deadline = time.monotonic() + 2
        while overlay.label.text() != transcript and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.005)

        assert overlay.isVisible()
        assert overlay.label.text() == transcript
        assert transcript_show_threads == [gui_thread]
        overlay._state = "fading"
        overlay._finish_fade()

        expected_path = str(candidate_path.resolve())
        assert retrieval_queries[0] == expected_query

        if expected_action == "open":
            assert actions == [("open", expected_path)]
            assert overlay.label.text() == expected_overlay_text
            assert overlay.isVisible()
        elif expected_action == "show":
            assert actions == [
                ("show", ["explorer.exe", "/select,", expected_path])
            ]
            assert overlay.label.text() == expected_overlay_text
            assert overlay.isVisible()
        elif expected_action == "find":
            assert actions == [("find", expected_path)]
            assert overlay.label.text() == expected_overlay_text
            assert not overlay.isVisible()
        elif expected_action == "options":
            assert actions[0][0] == "options"
            assert len(actions[0][1]) == 2
            assert not any(action[0] in {"open", "show"} for action in actions)
            assert overlay.label.text() == expected_overlay_text
            assert overlay.isVisible()
        else:
            assert ("no_match_message", None) in actions
            assert not any(action[0] in {"open", "show"} for action in actions)
            assert overlay.label.text() == expected_overlay_text
            assert overlay.isVisible()
    finally:
        release_alt.set()
        event_thread.join(timeout=1)
        if controller._hotkey_unregister is not None:
            controller._hotkey_unregister()
        if controller.result_card is not None:
            controller.result_card.close()
        overlay.close()
        input_bar.close()
        qt_app.processEvents()
