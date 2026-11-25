"""Lazy faster-whisper transcription with local CUDA-to-CPU fallback."""

import gc
import threading
import time

from observability.logger import log_voice, log_voice_exception


VOICE_SLOW_SECONDS_PER_AUDIO_SECOND = 1.6


def should_offer_lighter_voice_model(
    elapsed_seconds, clip_seconds, tier, model_name, device
):
    return (
        tier in {"mid", "big"}
        and model_name == "Whisper large-v3-turbo"
        and device == "cpu"
        and clip_seconds > 0
        and elapsed_seconds
        > clip_seconds * VOICE_SLOW_SECONDS_PER_AUDIO_SECOND
    )


def _is_cuda_error(error):
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "cuda",
            "cudnn",
            "cublas",
            "out of memory",
            "device-side",
        )
    )


class VoiceTranscriber:
    def __init__(
        self,
        model_path=None,
        idle_seconds=300,
        whisper_model=None,
        model_status_callback=None,
        model_name="Whisper",
        prefer_cuda=False,
    ):
        self.model_path = model_path
        self.idle_seconds = idle_seconds
        self._model = whisper_model
        self.model_status_callback = model_status_callback
        self.model_name = model_name
        self.prefer_cuda = bool(prefer_cuda)
        self._cuda_disabled = False
        self._cuda_verified = False
        self._cuda_fallback_reported = False
        self._active_device = "cpu" if whisper_model is not None else None
        self._lock = threading.Lock()
        self._transcription_lock = threading.Lock()
        self._last_used = None
        self._timer = None

    @property
    def active_device(self):
        return self._active_device

    def _report_status(self, message):
        if self.model_status_callback is not None:
            self.model_status_callback(message)

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
                    self._active_device = None
                    self._last_used = None
            gc.collect()
        except Exception:
            log_voice_exception("voice model idle-unload timer failed")

    def unload(self):
        with self._lock:
            self._model = None
            self._active_device = None
            self._last_used = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        gc.collect()

    def _new_model(self, device, compute_type):
        try:
            from faster_whisper import WhisperModel
        except Exception as error:
            raise RuntimeError(
                "Voice search is not available: faster-whisper "
                "could not be loaded."
            ) from error
        if self.model_path is None:
            raise RuntimeError("A local voice model has not been selected.")
        return WhisperModel(
            str(self.model_path),
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )

    @staticmethod
    def _warm_up_cuda(model):
        import numpy as np

        segments, _ = model.transcribe(
            np.zeros(8000, dtype=np.float32),
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        list(segments)

    def _report_cuda_fallback(self, reason):
        self._cuda_disabled = True
        if self._cuda_fallback_reported:
            return
        self._cuda_fallback_reported = True
        log_voice(
            "CUDA Whisper unavailable; using CPU int8 with the same local "
            f"model files. reason={reason}"
        )
        self._report_status(
            f"Voice using CPU int8 (CUDA unavailable): {reason}"
        )

    def _load_cpu_model(self):
        self._report_status(f"Loading voice model: {self.model_name} on CPU...")
        model = self._new_model("cpu", "int8")
        self._active_device = "cpu"
        self._report_status("Voice model loaded on CPU (int8)")
        return model

    def _load_model(self):
        if self.prefer_cuda and not self._cuda_disabled:
            model = None
            try:
                import ctranslate2

                device_count = ctranslate2.get_cuda_device_count()
                if device_count < 1:
                    raise RuntimeError(
                        "CTranslate2 reports no CUDA devices"
                    )
                self._report_status(
                    f"Loading voice model: {self.model_name} on CUDA..."
                )
                model = self._new_model("cuda", "int8_float16")
                if not self._cuda_verified:
                    self._warm_up_cuda(model)
                    self._cuda_verified = True
                self._active_device = "cuda"
                self._report_status(
                    "Voice model loaded on CUDA (int8_float16)"
                )
                return model
            except Exception as error:
                reason = str(error)
                error.__traceback__ = None
                model = None
                gc.collect()
                self._report_cuda_fallback(reason)
        return self._load_cpu_model()

    def get_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            self._model = self._load_model()
            return self._model

    def _retry_on_cpu(self, reason):
        with self._lock:
            old_model = self._model
            self._model = None
            self._active_device = None
        del old_model
        gc.collect()
        self._report_cuda_fallback(
            f"CUDA decode failed; retrying this clip on CPU int8: {reason}"
        )
        model = self._load_cpu_model()
        with self._lock:
            self._model = model
        return model

    @staticmethod
    def _decode(model, audio, **options):
        segments, _ = model.transcribe(audio, language="en", **options)
        return " ".join(
            segment.text.strip() for segment in segments
        ).strip()

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
            try:
                text = self._decode(model, audio, **options)
            except Exception as error:
                if self.active_device != "cuda" or not _is_cuda_error(error):
                    raise
                reason = str(error)
                error.__traceback__ = None
                model = None
                gc.collect()
                model = self._retry_on_cpu(reason)
                text = self._decode(model, audio, **options)
        with self._lock:
            self._last_used = time.monotonic()
        self._schedule_unload()
        return text
