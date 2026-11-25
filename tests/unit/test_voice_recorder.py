import numpy as np

from app.voice_recorder import VoiceRecorder, VoiceRecorderError


class FakeStream:
    def __init__(self, callback, frames):
        self.callback = callback
        self.frames = frames
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True
        self.callback(
            np.zeros(self.frames, dtype=np.float32).reshape(-1, 1),
            self.frames,
            None,
            None,
        )

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_fake_stream_caps_at_max_and_too_short_returns_none(tmp_path):
    stream = None

    def factory(callback):
        nonlocal stream
        stream = FakeStream(callback, 100)
        return stream

    recorder = VoiceRecorder(
        stream_factory=factory,
        max_seconds=1,
        min_seconds=0.6,
    )
    recorder.start()
    assert recorder.stop() is None
    assert stream.closed

    recorder = VoiceRecorder(
        stream_factory=lambda callback: FakeStream(
            callback, recorder.SAMPLE_RATE
        ),
        max_seconds=2,
        min_seconds=0.6,
    )
    recorder.start()
    snapshot = recorder.snapshot()
    assert snapshot is not None
    assert len(snapshot) == recorder.SAMPLE_RATE
    audio = recorder.stop()
    assert len(audio) == recorder.SAMPLE_RATE
    recorder.clear()
    assert recorder._buffer == []


def test_no_microphone_error_is_clear(monkeypatch):
    import sounddevice
    monkeypatch.setattr(
        sounddevice,
        "InputStream",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("busy")),
    )
    recorder = VoiceRecorder()
    try:
        recorder.start()
    except VoiceRecorderError as error:
        assert "microphone" in str(error).lower()
    else:
        raise AssertionError("missing microphone should raise VoiceRecorderError")


def test_recording_never_creates_audio_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stream = FakeStream

    def factory(callback):
        return stream(callback, 16000)

    recorder = VoiceRecorder(stream_factory=factory)
    recorder.start()
    audio = recorder.stop()
    assert audio is not None
    assert list(tmp_path.iterdir()) == []


def test_cancel_discards_preroll_and_closes_stream():
    stream = None

    def factory(callback):
        nonlocal stream
        stream = FakeStream(callback, 16000)
        return stream

    recorder = VoiceRecorder(stream_factory=factory)
    recorder.start()
    assert recorder.snapshot() is not None

    recorder.cancel()

    assert stream.stopped
    assert stream.closed
    assert recorder.snapshot() is None
