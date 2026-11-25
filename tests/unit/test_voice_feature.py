import json
import os
import sys

import app_paths


def test_tier_to_voice_model_mapping():
    config = app_paths.load_model_registry()
    assert app_paths.get_voice_model_for_tier(
        "small", config
    )["repo_id"].endswith("small.en")
    assert app_paths.get_voice_model_for_tier(
        "mid", config
    )["repo_id"] == "dropbox-dash/faster-whisper-large-v3-turbo"
    assert app_paths.get_voice_model_for_tier(
        "big", config
    )["repo_id"] == "dropbox-dash/faster-whisper-large-v3-turbo"
    assert app_paths.get_voice_model_for_tier(
        "mid", config, model_name="Whisper small.en"
    )["repo_id"].endswith("small.en")


def test_missing_voice_model_triggers_consent(monkeypatch):
    from indexing import model_downloader

    model = app_paths.get_voice_model_for_tier("mid")
    monkeypatch.setattr(model_downloader, "find_model", lambda repo_id: None)
    assert model_downloader.missing_voice_model_for_tier("mid")["repo_id"] == model["repo_id"]


def test_slow_voice_hint_is_one_time_and_uses_registry_download_size(
    monkeypatch,
):
    from types import SimpleNamespace
    from app import main

    class FakeStatus:
        def __init__(self):
            self.messages = []

        def show_notice(self, message):
            self.messages.append(message)

    class FakeTray:
        def __init__(self):
            self.messages = []

        def showMessage(self, *args):
            self.messages.append(args)

    class FakeAction:
        def __init__(self):
            self.enabled = False

        def setEnabled(self, value):
            self.enabled = value

    model = {
        "name": "Whisper small.en",
        "repo_id": "Systran/faster-whisper-small.en",
        "download_size_bytes": 486098798,
        "files": {},
    }
    monkeypatch.setattr(
        main,
        "get_voice_model_for_tier",
        lambda _tier, model_name=None: model,
    )
    monkeypatch.setattr(
        main,
        "QSystemTrayIcon",
        SimpleNamespace(
            MessageIcon=SimpleNamespace(Information=1)
        ),
    )

    instance = object.__new__(main.SmartFileFinderApp)
    instance._voice_tier = "mid"
    instance._slow_voice_hint_shown = False
    instance._lighter_voice_offer_pending = False
    instance.status_window = FakeStatus()
    instance.tray_icon = FakeTray()
    instance.lighter_voice_action = FakeAction()
    saved = []
    instance._save_voice_setting = lambda key, value: saved.append(
        (key, value)
    )

    instance._on_slow_voice_detected(8.1, 5.0)
    instance._on_slow_voice_detected(9.0, 5.0)

    assert len(instance.tray_icon.messages) == 1
    assert "Voice is slow on this PC" in instance.tray_icon.messages[0][1]
    assert "486 MB" in instance.tray_icon.messages[0][1]
    assert instance._lighter_voice_offer_pending
    assert instance.lighter_voice_action.enabled
    assert saved == [("voice_slow_hint_shown", True)]


def test_import_failure_disables_voice(monkeypatch):
    from app import main

    instance = object.__new__(main.SmartFileFinderApp)
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name in {"faster_whisper", "ctranslate2", "sounddevice", "numpy"}:
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    assert instance._check_voice_imports() is False


def test_offscreen_voice_pill_state_sequence(monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from app.voice_pill import VoicePill

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    pill = VoicePill()
    pill.show_listening()
    assert pill.label.text().startswith("Listening...")
    pill.show_transcribing()
    assert pill.label.text() == "Transcribing..."
    pill.show_error("Didn't catch that, try again")
    assert "Didn't catch" in pill.label.text()
    pill.close()


def test_open_alone_shows_instruction_without_search(monkeypatch):
    from app.main import SmartFileFinderApp

    class FakeOverlay:
        def __init__(self):
            self.messages = []

        def show_message(self, message, duration_ms):
            self.messages.append((message, duration_ms))

    instance = object.__new__(SmartFileFinderApp)
    instance._pending_voice_transcript = "open"
    instance._pending_voice_auto_search = True
    instance._voice_overlay_active = __import__("threading").Event()
    instance._hotkey_unregister = None
    instance.voice_overlay = FakeOverlay()
    searched = []
    instance._dispatch_command = searched.append

    instance._on_voice_overlay_dismissed()

    assert searched == []
    assert instance.voice_overlay.messages == [
        ("Say what to open, for example: open corrective rag", 2000)
    ]


def test_voice_auto_search_false_opens_editable_bar_and_parses_unchanged_text():
    from app.main import SmartFileFinderApp

    class FakeTextInput:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    class FakeInputBar:
        def __init__(self):
            self.text_input = FakeTextInput()
            self.shown = False

        def show_and_focus(self):
            self.shown = True

    instance = object.__new__(SmartFileFinderApp)
    instance._pending_voice_transcript = "open the corrective rag"
    instance._pending_voice_auto_search = False
    instance._voice_overlay_active = __import__("threading").Event()
    instance._hotkey_unregister = None
    instance.input_bar = FakeInputBar()
    submitted = []
    instance._dispatch_command = (
        lambda parsed, voice_feedback=False: submitted.append(parsed)
    )

    instance._on_voice_overlay_dismissed()
    assert instance.input_bar.text_input.text == "open the corrective rag"
    assert instance.input_bar.shown
    assert submitted == []

    instance._on_query_submitted("open the corrective rag")
    assert len(submitted) == 1
    assert submitted[0].command.value == "open"
    assert submitted[0].query == "corrective rag"


def test_typed_and_spoken_commands_share_dispatch():
    from app.main import SmartFileFinderApp

    instance = object.__new__(SmartFileFinderApp)
    dispatched = []
    instance._pending_voice_transcript = None
    instance._dispatch_command = (
        lambda parsed, voice_feedback=False: dispatched.append(parsed)
    )

    instance._on_query_submitted("/open corrective rag")

    instance._pending_voice_transcript = " Open corrective rag. "
    instance._pending_voice_auto_search = True
    instance._voice_overlay_active = __import__("threading").Event()
    instance._hotkey_unregister = None
    instance._on_voice_overlay_dismissed()

    assert len(dispatched) == 2
    assert dispatched[0] == dispatched[1]
    assert dispatched[0].command.value == "open"
    assert dispatched[0].query == "corrective rag"


def test_voice_open_options_are_shown_without_automatic_open(monkeypatch):
    from app import main
    from app.command_parser import parse_voice_command

    class FakeSignal:
        def connect(self, _callback):
            pass

    class FakeOptions:
        def __init__(self, options, anchor):
            self.options = options
            self.anchor = anchor
            self.choice_made = FakeSignal()
            self.shown = False

        def show(self):
            self.shown = True

    instance = object.__new__(main.SmartFileFinderApp)
    instance.input_bar = object()
    instance.result_card = None
    instance.options_group = None
    monkeypatch.setattr(
        main,
        "handle_query",
        lambda _query: {
            "action": "options",
            "options": [{"path": "candidate.txt"}],
        },
    )
    monkeypatch.setattr(main, "OptionsPopupGroup", FakeOptions)

    instance._dispatch_command(parse_voice_command("open candidate"))

    assert instance.options_group.shown
    assert instance.result_card is None


def test_cancel_gesture_discards_audio_and_fades_overlay():
    from app.main import SmartFileFinderApp

    class FakeRecorder:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class FakeOverlay:
        def __init__(self):
            self.faded = False

        def fade_out(self):
            self.faded = True

    instance = object.__new__(SmartFileFinderApp)
    instance._recording = True
    instance._pending_voice_transcript = "open a file"
    instance._voice_overlay_active = __import__("threading").Event()
    instance._voice_overlay_active.set()
    instance._hotkey_unregister = None
    instance.voice_recorder = FakeRecorder()
    instance.voice_overlay = FakeOverlay()

    instance._on_voice_state("cancel")

    assert instance.voice_recorder.cancelled
    assert not instance._recording
    assert instance._pending_voice_transcript is None
    assert not instance._voice_overlay_active.is_set()
    assert instance.voice_overlay.faded


def test_early_preroll_release_discards_audio_without_feedback():
    from app.main import SmartFileFinderApp

    class FakeRecorder:
        def __init__(self):
            self.started = 0
            self.cancelled = 0

        def start(self):
            self.started += 1

        def cancel(self):
            self.cancelled += 1

    instance = object.__new__(SmartFileFinderApp)
    instance.voice_enabled = True
    instance._recording = False
    instance._voice_setup_window = None
    instance._voice_tier = "mid"
    instance._voice_preroll_active = False
    instance.voice_recorder = FakeRecorder()
    instance._load_voice_setting = lambda _key, default: default

    instance._on_voice_state("pre_roll_start")
    instance._on_voice_state("pre_roll_cancel")

    assert instance.voice_recorder.started == 1
    assert instance.voice_recorder.cancelled == 1
    assert not instance._voice_preroll_active


def test_hold_release_keeps_configured_tail_delay():
    from app.main import SmartFileFinderApp

    class FakeTimer:
        def __init__(self):
            self.delay = None

        def start(self, delay):
            self.delay = delay

    instance = object.__new__(SmartFileFinderApp)
    instance._recording = True
    instance._voice_tail_timer = FakeTimer()
    instance._load_voice_setting = lambda _key, default: 300

    instance._on_voice_state("hold_end")

    assert instance._voice_tail_timer.delay == 300


def test_ctrl_shift_space_tap_still_emits_typing_bar_signal():
    from app.main import SmartFileFinderApp
    from app.voice_hotkey import VoiceGesture

    class FakeSignal:
        def __init__(self):
            self.emitted = 0

        def emit(self):
            self.emitted += 1

    instance = object.__new__(SmartFileFinderApp)
    instance.hotkey_bridge = type(
        "FakeBridge", (), {"triggered": FakeSignal()}
    )()

    instance._on_voice_hotkey_event(VoiceGesture.TAP)

    assert instance.hotkey_bridge.triggered.emitted == 1
