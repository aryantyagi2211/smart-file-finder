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
