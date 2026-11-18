"""Pure tap-versus-hold detector for the Ctrl+Shift+Space gesture."""

from enum import Enum


class VoiceGesture(str, Enum):
    TAP = "TAP"
    HOLD_START = "HOLD_START"
    HOLD_END = "HOLD_END"


class TapHoldDetector:
    def __init__(self, hold_threshold_ms=400, max_hold_ms=15000, clock=None):
        self.hold_threshold_ms = hold_threshold_ms
        self.max_hold_ms = max_hold_ms
        self._clock = clock or (lambda: __import__("time").monotonic() * 1000)
        self._pressed = set()
        self._hotkey_down = False
        self._hold_started = False
        self._started_at = None
        self._tap_emitted = False

    @staticmethod
    def _normalize(name):
        name = str(name).casefold().replace("_", " ")
        return name

    def key_down(self, key_name, repeated=False):
        name = self._normalize(key_name)
        if repeated:
            return None
        if name in {"ctrl", "left ctrl", "right ctrl"}:
            self._pressed.add("ctrl")
        elif name in {"shift", "left shift", "right shift"}:
            self._pressed.add("shift")
        elif name in {"space", "spacebar"}:
            self._pressed.add("space")
        if not self._hotkey_down and {"ctrl", "shift", "space"} <= self._pressed:
            self._hotkey_down = True
            self._hold_started = False
            self._tap_emitted = False
            self._started_at = self._clock()
        return None

    def poll(self):
        if not self._hotkey_down or self._hold_started or self._started_at is None:
            return None
        elapsed = self._clock() - self._started_at
        if elapsed >= self.hold_threshold_ms:
            self._hold_started = True
            return VoiceGesture.HOLD_START
        return None

    def key_up(self, key_name):
        name = self._normalize(key_name)
        if name in {"ctrl", "left ctrl", "right ctrl"}:
            self._pressed.discard("ctrl")
        elif name in {"shift", "left shift", "right shift"}:
            self._pressed.discard("shift")
        elif name in {"space", "spacebar"}:
            self._pressed.discard("space")

        if not self._hotkey_down:
            return None

        ended = self._clock()
        elapsed = ended - self._started_at
        if self._hold_started:
            self._reset()
            return VoiceGesture.HOLD_END
        if not {"ctrl", "shift", "space"} & self._pressed:
            return None
        if name in {"ctrl", "left ctrl", "right ctrl",
                    "shift", "left shift", "right shift", "space", "spacebar"}:
            if elapsed < self.hold_threshold_ms:
                self._reset()
                return VoiceGesture.TAP
        return None

    def _reset(self):
        self._pressed.clear()
        self._hotkey_down = False
        self._hold_started = False
        self._started_at = None
        self._tap_emitted = False
