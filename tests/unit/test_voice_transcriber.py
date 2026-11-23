from app.voice_transcriber import VoiceTranscriber


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return [FakeSegment(" adaptive rag ")], None


def test_transcriber_lazy_load_language_beam_and_vad():
    fake = FakeWhisper()
    transcriber = VoiceTranscriber(
        model_path="/tmp/model",
        whisper_model=fake,
    )
    assert transcriber._model is fake
    text = transcriber.transcribe([0.0] * 16000)

    assert text == "adaptive rag"
    assert fake.calls == [{
        "language": "en",
        "beam_size": 5,
        "vad_filter": True,
    }]


def test_partial_transcription_uses_fast_non_vad_decode():
    fake = FakeWhisper()
    transcriber = VoiceTranscriber(
        model_path="/tmp/model",
        whisper_model=fake,
    )

    assert transcriber.transcribe_partial([0.0] * 16000) == "adaptive rag"
    assert fake.calls == [{
        "language": "en",
        "beam_size": 1,
        "vad_filter": False,
        "condition_on_previous_text": False,
    }]


def test_idle_unload():
    fake = FakeWhisper()
    transcriber = VoiceTranscriber(
        model_path="/tmp/model",
        idle_seconds=0,
        whisper_model=fake,
    )
    transcriber.transcribe([0.0])
    transcriber._unload_if_idle()
    assert transcriber._model is None


def test_lazy_voice_model_load_reports_loading_stage(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    statuses = []
    fake = FakeWhisper()
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=lambda *args, **kwargs: fake),
    )
    transcriber = VoiceTranscriber(
        model_path=tmp_path,
        model_name="Whisper base.en",
        model_status_callback=statuses.append,
    )

    assert transcriber.get_model() is fake
    assert statuses == [
        "Loading voice model: Whisper base.en...",
        "Voice model loaded",
    ]
