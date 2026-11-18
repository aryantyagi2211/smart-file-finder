"""Lazy, English-only faster-whisper transcription and idle unloading."""

import threading
import time


class VoiceTranscriber:
    def __init__(self, model_path=None, idle_seconds=300, whisper_model=None):
        self.model_path = model_path
        self.idle_seconds = idle_seconds
        self._model = whisper_model
        self._lock = threading.Lock()
        self._last_used = None
        self._timer = None

    def _schedule_unload(self):
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(
            self.idle_seconds,
            self._unload_if_idle,
        )
        self._timer.daemon = True
        self._timer.start()

    def _unload_if_idle(self):
        with self._lock:
            if self._last_used is None:
                return
            if time.monotonic() - self._last_used >= self.idle_seconds:
                self._model = None
                self._last_used = None

    def unload(self):
        with self._lock:
            self._model = None
            self._last_used = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def get_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except Exception as error:
                raise RuntimeError(
                    "Voice search is not available: faster-whisper "
                    "could not be loaded."
                ) from error
            if self.model_path is None:
                raise RuntimeError("A local voice model has not been selected.")
            self._model = WhisperModel(
                str(self.model_path),
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
            return self._model

    def transcribe(self, audio):
        model = self.get_model()
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        with self._lock:
            self._last_used = time.monotonic()
        self._schedule_unload()
        return text
