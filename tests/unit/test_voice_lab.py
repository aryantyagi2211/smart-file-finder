import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app import voice_lab


def test_voice_lab_has_eight_fixed_and_six_data_driven_phrases():
    phrases = voice_lab.recording_phrases()

    assert len(phrases) == 14
    assert phrases[:8] == voice_lab.FIXED_PHRASES
    assert all(phrase.split(maxsplit=1)[0] in {"open", "show", "find"}
               for phrase in phrases)
    assert any("meeting notes" in phrase for phrase in phrases)


def test_recording_layout_stays_under_private_voice_lab_directory(tmp_path):
    audio = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)
    path = voice_lab.write_recording(
        tmp_path, "session-1", 1, audio, "show corrective rag"
    )
    recordings = voice_lab.load_recordings(tmp_path)

    assert path == (
        tmp_path / "recordings" / "session-1" / "phrase-01.wav"
    )
    assert path.is_file()
    assert len(recordings) == 1
    assert recordings[0].phrase == "show corrective rag"
    decoded = voice_lab.read_recording(recordings[0])
    assert len(decoded) == len(audio)
    assert (path.parent / "manifest.json").is_file()
    assert not list(tmp_path.glob("*.wav"))


def test_record_mode_writes_wav_after_the_second_enter(tmp_path):
    prompts = []

    class FakeRecorder:
        def __init__(self, **_kwargs):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            assert self.started
            return np.zeros(16000, dtype=np.float32)

        def cancel(self):
            raise AssertionError("successful recording must not be cancelled")

    session = voice_lab.record_phrases(
        recorder_factory=FakeRecorder,
        input_fn=lambda prompt: prompts.append(prompt) or "",
        voice_lab_root=tmp_path,
        phrases=["show corrective rag"],
    )

    recordings = voice_lab.load_recordings(tmp_path)
    assert session
    assert len(prompts) == 2
    assert len(recordings) == 1
    assert recordings[0].phrase == "show corrective rag"


def test_hugging_face_candidate_metadata_pins_size_and_license():
    siblings = [
        SimpleNamespace(rfilename="config.json", size=10),
        SimpleNamespace(rfilename="model.bin", size=100),
        SimpleNamespace(rfilename="tokenizer.json", size=20),
        SimpleNamespace(rfilename="vocabulary.txt", size=30),
        SimpleNamespace(rfilename="README.md", size=50),
    ]
    info = SimpleNamespace(
        sha="abc123",
        gated=False,
        private=False,
        card_data=SimpleNamespace(license="mit"),
        tags=["license:mit"],
        siblings=siblings,
    )

    class FakeApi:
        def model_info(self, repo_id, **kwargs):
            assert repo_id == "Systran/faster-whisper-tiny.en"
            assert kwargs["files_metadata"] is True
            return info

    model = voice_lab.fetch_candidate_metadata(
        voice_lab.VOICE_LAB_CANDIDATES[0], api=FakeApi()
    )

    assert model["revision"] == "abc123"
    assert model["license"] == "mit"
    assert model["gated"] is False
    assert model["download_size_bytes"] == 160
    assert model["repository_size_bytes"] == 210
    assert model["files"] == {
        "config.json": 10,
        "model.bin": 100,
        "tokenizer.json": 20,
        "vocabulary.txt": 30,
    }


def test_download_consent_supports_decline_and_cancel():
    model = {"name": "Whisper test", "download_size_bytes": 1_000_000}
    assert not voice_lab.consent_to_download(
        model, input_fn=lambda _prompt: "n"
    )
    try:
        voice_lab.consent_to_download(
            model, input_fn=lambda _prompt: "cancel"
        )
    except InterruptedError as error:
        assert "cancelled" in str(error)
    else:
        raise AssertionError("cancel must stop evaluation")


def test_voice_lab_model_download_is_marked_as_candidate(monkeypatch):
    from indexing import model_downloader

    calls = []
    monkeypatch.setattr(
        model_downloader,
        "download_model",
        lambda model, download_source: calls.append(
            (model, download_source)
        ) or "snapshot",
    )
    candidate = {"repo_id": "test/voice-candidate"}

    assert voice_lab._download_voice_lab_model(candidate) == "snapshot"
    assert calls == [(candidate, "voice_lab")]


def test_fake_model_evaluation_reports_transcript_metrics_and_no_audio(
    tmp_path,
):
    audio = np.zeros(16000, dtype=np.float32)
    path = voice_lab.write_recording(
        tmp_path, "session-2", 1, audio, "show corrective rag"
    )
    recording = voice_lab.load_recordings(tmp_path)[0]
    model = {
        "name": "Whisper fake",
        "repo_id": "test/fake",
        "license": "mit",
        "gated": False,
        "download_size_bytes": 100,
        "repository_size_bytes": 120,
        "revision": "test-revision",
    }
    settings = {
        "initial_prompt": "Open, show, find.",
        "hotwords": "open show find",
        "command_aliases": {
            "show": ["show", "sho", "so", "shoe", "shaw"],
            "open": ["open", "opan"],
            "find": ["find", "fine", "fined"],
        },
    }

    class FakeTranscriber:
        def __init__(self, **kwargs):
            assert kwargs["initial_prompt"] == settings["initial_prompt"]
            assert kwargs["hotwords"] == settings["hotwords"]

        def get_model(self):
            return object()

        def transcribe(self, _audio):
            return "So corrective rag"

        def unload(self):
            pass

    result = voice_lab.evaluate_faster_candidate(
        model,
        path.parent,
        [recording],
        settings,
        transcriber_factory=FakeTranscriber,
    )
    report_path = voice_lab.save_report(
        [result],
        report_path=tmp_path / "eval" / "voice_report.md",
        report_date="2026-11-25",
    )
    report = report_path.read_text(encoding="utf-8")

    assert result.status == "complete"
    assert result.wer == 1 / 3
    assert result.command_accuracy == 1.0
    assert result.transcripts == [
        ("show corrective rag", "So corrective rag")
    ]
    assert "# Voice Lab report — 2026-11-25" in report
    assert "So corrective rag" in report
    assert "33.33%" in report
    assert ".wav" not in report
    assert path.exists()
