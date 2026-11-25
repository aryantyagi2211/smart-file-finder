from app.voice_transcriber import (
    VoiceTranscriber,
    should_offer_lighter_voice_model,
)


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return [FakeSegment(" adaptive rag ")], None


def test_transcriber_uses_production_decode_options_and_vocabulary():
    fake = FakeWhisper()
    transcriber = VoiceTranscriber(
        model_path="/tmp/model",
        whisper_model=fake,
    )
    assert transcriber._model is fake
    text = transcriber.transcribe([0.1] * 16000)

    assert text == "adaptive rag"
    assert fake.calls == [{
        "language": "en",
        "beam_size": 5,
        "vad_filter": False,
        "temperature": 0,
        "condition_on_previous_text": False,
        "initial_prompt": "Open, show, find. open corrective rag. show the roadmap.",
        "hotwords": "open show find",
    }]


def test_partial_transcription_uses_fast_non_vad_decode():
    fake = FakeWhisper()
    transcriber = VoiceTranscriber(
        model_path="/tmp/model",
        whisper_model=fake,
    )

    assert transcriber.transcribe_partial([0.1] * 16000) == "adaptive rag"
    assert fake.calls == [{
        "language": "en",
        "beam_size": 1,
        "vad_filter": False,
        "temperature": 0,
        "condition_on_previous_text": False,
        "initial_prompt": "Open, show, find. open corrective rag. show the roadmap.",
        "hotwords": "open show find",
    }]


def test_all_silence_returns_empty_without_calling_model():
    fake = FakeWhisper()
    transcriber = VoiceTranscriber(whisper_model=fake)

    assert transcriber.transcribe([0.0] * 16000) == ""
    assert fake.calls == []


def test_transcriber_normalizes_audio_and_skips_unsupported_hotwords():
    class NoHotwordsWhisper:
        def __init__(self):
            self.audio = None
            self.kwargs = None

        def transcribe(
            self,
            audio,
            language=None,
            beam_size=5,
            vad_filter=False,
            temperature=0,
            condition_on_previous_text=False,
            initial_prompt=None,
        ):
            self.audio = audio
            self.kwargs = {
                "language": language,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "temperature": temperature,
                "condition_on_previous_text": condition_on_previous_text,
                "initial_prompt": initial_prompt,
            }
            return [FakeSegment(" normalized ")], None

    fake = NoHotwordsWhisper()
    transcriber = VoiceTranscriber(whisper_model=fake)
    audio = [0.0, 0.1, 0.2, 0.1, 0.0]

    assert transcriber.transcribe(audio) == "normalized"
    assert len(fake.audio) == 3
    assert abs(float(max(fake.audio)) - 0.7079458) < 1e-5
    assert "hotwords" not in fake.kwargs


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
        "Loading voice model: Whisper base.en on CPU...",
        "Voice model loaded on CPU (int8)",
    ]


def _install_voice_runtime(monkeypatch, cuda_devices, factory):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=factory),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: cuda_devices),
    )


def _watch_model_downloads(monkeypatch):
    from indexing import model_downloader

    calls = []
    monkeypatch.setattr(
        model_downloader,
        "download_model",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def test_mid_high_cuda_model_uses_int8_float16_and_warms_up(
    monkeypatch, tmp_path
):
    constructors = []

    def factory(path, **kwargs):
        constructors.append((path, kwargs))
        return FakeWhisper()

    _install_voice_runtime(monkeypatch, 1, factory)
    transcriber = VoiceTranscriber(
        model_path=tmp_path,
        model_name="Whisper large-v3-turbo",
        prefer_cuda=True,
    )

    assert transcriber.transcribe([0.1]) == "adaptive rag"
    assert transcriber.active_device == "cuda"
    assert constructors == [(
        str(tmp_path),
        {
            "device": "cuda",
            "compute_type": "int8_float16",
            "local_files_only": True,
        },
    )]


def test_cuda_warmup_failure_falls_back_to_cpu_same_local_model(
    monkeypatch, tmp_path
):
    constructors = []
    statuses = []
    download_calls = _watch_model_downloads(monkeypatch)

    class WarmupFailure(FakeWhisper):
        def transcribe(self, audio, **kwargs):
            if constructors[-1][1]["device"] == "cuda":
                raise RuntimeError("CUBLAS_STATUS_NOT_INITIALIZED")
            return super().transcribe(audio, **kwargs)

    def factory(path, **kwargs):
        constructors.append((path, kwargs))
        return WarmupFailure()

    _install_voice_runtime(monkeypatch, 1, factory)
    transcriber = VoiceTranscriber(
        model_path=tmp_path,
        model_name="Whisper large-v3-turbo",
        prefer_cuda=True,
        model_status_callback=statuses.append,
    )

    assert transcriber.transcribe([0.1]) == "adaptive rag"
    assert transcriber.active_device == "cpu"
    assert [entry[0] for entry in constructors] == [
        str(tmp_path),
        str(tmp_path),
    ]
    assert [entry[1]["device"] for entry in constructors] == [
        "cuda",
        "cpu",
    ]
    assert all(entry[1]["local_files_only"] for entry in constructors)
    assert any("CUBLAS_STATUS_NOT_INITIALIZED" in message for message in statuses)
    assert download_calls == []


def test_cuda_decode_out_of_memory_retries_clip_once_on_cpu(
    monkeypatch, tmp_path
):
    constructors = []
    cuda_decode_calls = []
    download_calls = _watch_model_downloads(monkeypatch)

    class OomOnClip(FakeWhisper):
        def transcribe(self, audio, **kwargs):
            if constructors[-1][1]["device"] == "cuda":
                cuda_decode_calls.append(True)
                if len(cuda_decode_calls) > 1:
                    raise RuntimeError("CUDA out of memory")
            return super().transcribe(audio, **kwargs)

    def factory(path, **kwargs):
        constructors.append((path, kwargs))
        return OomOnClip()

    _install_voice_runtime(monkeypatch, 1, factory)
    transcriber = VoiceTranscriber(
        model_path=tmp_path,
        model_name="Whisper large-v3-turbo",
        prefer_cuda=True,
    )

    assert transcriber.transcribe([0.1]) == "adaptive rag"
    assert transcriber.active_device == "cpu"
    assert [entry[1]["device"] for entry in constructors] == [
        "cuda",
        "cpu",
    ]
    assert all(entry[0] == str(tmp_path) for entry in constructors)
    assert all(entry[1]["local_files_only"] for entry in constructors)
    assert download_calls == []


def test_no_cuda_device_uses_cpu_without_attempting_download(
    monkeypatch, tmp_path
):
    constructors = []
    download_calls = _watch_model_downloads(monkeypatch)

    def factory(path, **kwargs):
        constructors.append((path, kwargs))
        return FakeWhisper()

    _install_voice_runtime(monkeypatch, 0, factory)
    transcriber = VoiceTranscriber(
        model_path=tmp_path,
        model_name="Whisper large-v3-turbo",
        prefer_cuda=True,
    )

    assert transcriber.transcribe([0.1]) == "adaptive rag"
    assert transcriber.active_device == "cpu"
    assert len(constructors) == 1
    assert constructors[0][1]["device"] == "cpu"
    assert constructors[0][1]["local_files_only"] is True
    assert download_calls == []


def test_lighter_model_hint_only_triggers_for_slow_cpu_turbo():
    assert not should_offer_lighter_voice_model(
        8.0, 5.0, "mid", "Whisper large-v3-turbo", "cpu"
    )
    assert should_offer_lighter_voice_model(
        8.01, 5.0, "mid", "Whisper large-v3-turbo", "cpu"
    )
    assert should_offer_lighter_voice_model(
        9.0, 5.0, "big", "Whisper large-v3-turbo", "cpu"
    )
    assert not should_offer_lighter_voice_model(
        9.0, 5.0, "mid", "Whisper large-v3-turbo", "cuda"
    )
    assert not should_offer_lighter_voice_model(
        9.0, 5.0, "small", "Whisper small.en", "cpu"
    )
