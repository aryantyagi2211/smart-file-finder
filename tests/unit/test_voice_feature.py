import json
import os
import sys

import app_paths


def test_tier_to_voice_model_mapping():
    config = app_paths.load_model_registry()
    assert app_paths.get_voice_model_for_tier("small", config)["repo_id"].endswith("tiny.en")
    assert app_paths.get_voice_model_for_tier("mid", config)["repo_id"].endswith("base.en")
    assert app_paths.get_voice_model_for_tier("big", config)["repo_id"].endswith("small.en")


def test_missing_voice_model_triggers_consent(monkeypatch):
    from indexing import model_downloader

    model = app_paths.get_voice_model_for_tier("mid")
    monkeypatch.setattr(model_downloader, "find_model", lambda repo_id: None)
    assert model_downloader.missing_voice_model_for_tier("mid")["repo_id"] == model["repo_id"]


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
    pill = VoicePill()
    pill.show_listening()
    assert pill.label.text().startswith("Listening...")
    pill.show_transcribing()
    assert pill.label.text() == "Transcribing..."
    pill.show_error("Didn't catch that, try again")
    assert "Didn't catch" in pill.label.text()
    pill.close()
