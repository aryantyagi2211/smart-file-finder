"""Lazy, English-only faster-whisper transcription and idle unloading."""

import threading
import time

from observability.logger import log_voice_exception


class VoiceTranscriber:
    def __init__(
        self,
        model_path=None,
        idle_seconds=300,
        whisper_model=None,
        model_status_callback=None,
        model_name="Whisper",
    ):
        self.model_path = model_path
        self.idle_seconds = idle_seconds
        self._model = whisper_model
        self.model_status_callback = model_status_callback
        self.model_name = model_name
        self._lock = threading.Lock()
        self._transcription_lock = threading.Lock()
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
        try:
            with self._lock:
                if self._last_used is None:
                    return
                if time.monotonic() - self._last_used >= self.idle_seconds:
                    self._model = None
                    self._last_used = None
        except Exception:
            log_voice_exception("voice model idle-unload timer failed")

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
            if self.model_status_callback is not None:
                self.model_status_callback(
                    f"Loading voice model: {self.model_name}..."
                )
            self._model = WhisperModel(
                str(self.model_path),
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
            if self.model_status_callback is not None:
                self.model_status_callback("Voice model loaded")
            return self._model

    def transcribe(self, audio):
        return self._transcribe(audio, beam_size=5, vad_filter=True)

    def transcribe_partial(self, audio):
        """Decode a live snapshot quickly; final transcription remains authoritative."""
        return self._transcribe(
            audio,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )

    def _transcribe(self, audio, **options):
        with self._transcription_lock:
            model = self.get_model()
            segments, _ = model.transcribe(
                audio,
                language="en",
                **options,
            )
            text = " ".join(
                segment.text.strip() for segment in segments
            ).strip()
        with self._lock:
            self._last_used = time.monotonic()
        self._schedule_unload()
        return text
