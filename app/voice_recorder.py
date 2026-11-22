"""In-memory microphone recording for hold-to-talk voice search."""

import threading

from observability.logger import log_voice, log_voice_exception


class VoiceRecorderError(RuntimeError):
    pass


class VoiceRecorder:
    SAMPLE_RATE = 16000

    def __init__(self, stream_factory=None, max_seconds=15, min_seconds=0.6):
        self._stream_factory = stream_factory
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds
        self._stream = None
        self._buffer = []
        self._lock = threading.Lock()
        self._cancelled = False
        self._closed = False
        self._frame_count = 0

    def _new_stream(self, callback):
        if self._stream_factory is not None:
            return self._stream_factory(callback)
        try:
            import sounddevice as sd
        except Exception as error:
            raise VoiceRecorderError(
                "Voice search is not available: sounddevice could not be loaded."
            ) from error
        try:
            return sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=callback,
                blocksize=0,
            )
        except Exception as error:
            raise VoiceRecorderError(
                f"Could not open the microphone: {error}"
            ) from error

    def start(self):
        with self._lock:
            self._buffer = []
            self._cancelled = False
            self._closed = False
            self._frame_count = 0

        def callback(indata, frames, time_info, status):
            try:
                if status:
                    log_voice(f"microphone callback status={status}")
                    return
                with self._lock:
                    if self._cancelled:
                        return
                    frames_copy = indata.copy()
                    remaining = (
                        self.max_seconds * self.SAMPLE_RATE - self._frame_count
                    )
                    if remaining > 0:
                        frames_copy = frames_copy[:max(0, remaining)]
                        self._buffer.append(frames_copy)
                        self._frame_count += len(frames_copy)
            except Exception:
                log_voice_exception("microphone callback failed")

        self._stream = self._new_stream(callback)
        try:
            self._stream.start()
        except Exception as error:
            raise VoiceRecorderError(
                f"Could not start the microphone: {error}"
            ) from error

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                try:
                    self._stream.close()
                except Exception:
                    log_voice_exception("microphone stream close failed")
        with self._lock:
            self._closed = True
            if not self._buffer:
                return None
            try:
                import numpy as np
                audio = np.concatenate(self._buffer, axis=0).reshape(-1)
            except Exception as error:
                raise VoiceRecorderError(
                    f"Could not prepare in-memory audio: {error}"
                ) from error
            duration = len(audio) / self.SAMPLE_RATE
            if duration < self.min_seconds:
                self._buffer = []
                return None
            result = audio
            self._buffer = []
            return result

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self._buffer = []
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                log_voice_exception("cancel could not stop microphone stream")
            try:
                self._stream.close()
            except Exception:
                log_voice_exception("cancel could not close microphone stream")
        self._closed = True

    def clear(self):
        with self._lock:
            self._buffer = []
