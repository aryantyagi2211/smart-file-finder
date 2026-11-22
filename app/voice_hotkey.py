"""Pure detector for the search-bar tap and voice-key hold gestures."""

from enum import Enum


class VoiceGesture(str, Enum):
    TAP = "TAP"
    HOLD_START = "HOLD_START"
    HOLD_END = "HOLD_END"
    CANCEL = "CANCEL"


class TapHoldDetector:
    _TAP_KEYS = {"ctrl", "shift", "space"}
    _BLOCKING_MODIFIERS = {"ctrl", "shift", "win"}

    def __init__(
        self,
        hold_threshold_ms=500,
        max_hold_ms=15000,
        clock=None,
        voice_key="left alt",
    ):
        self.hold_threshold_ms = hold_threshold_ms
        self.max_hold_ms = max_hold_ms
        self._clock = clock or (lambda: __import__("time").monotonic() * 1000)
        self._pressed = set()
        self.voice_key = self._normalize(voice_key)
        if self.voice_key not in {"left alt", "right alt"}:
            raise ValueError("voice_key must be 'left alt' or 'right alt'")
        self._voice_down = False
        self._voice_blocked = False
        self._voice_started = False
        self._voice_started_at = None
        self._voice_ui_active = False
        self.last_decision_reason = None
        self._tap_active = False
        self._tap_invalid = False
        self._tap_started_at = None

    @staticmethod
    def _normalize(name):
        name = " ".join(str(name).casefold().replace("_", " ").split())
        aliases = {
            "alt": "left alt",
            "alt gr": "right alt",
            "altgraph": "right alt",
            "left ctrl": "ctrl",
            "right ctrl": "ctrl",
            "left control": "ctrl",
            "right control": "ctrl",
            "left shift": "shift",
            "right shift": "shift",
            "left win": "win",
            "right win": "win",
            "left windows": "win",
            "right windows": "win",
            "windows": "win",
            "spacebar": "space",
        }
        return aliases.get(name, name)

    def key_down(self, key_name, repeated=False):
        name = self._normalize(key_name)
        if repeated or name in self._pressed:
            return None

        self.last_decision_reason = None
        self._pressed.add(name)
        if name == self.voice_key:
            self._voice_down = True
            self._voice_blocked = (
                bool(self._pressed - {self.voice_key})
                or bool(self._pressed & self._BLOCKING_MODIFIERS)
            )
            self._voice_started = False
            self._voice_started_at = (
                None if self._voice_blocked else self._clock()
            )
            return None

        if self._voice_down and not self._voice_blocked:
            self._voice_blocked = True
            self._voice_started_at = None
            if self._voice_started:
                self._voice_started = False
                self._voice_ui_active = False
                self.last_decision_reason = (
                    "another key pressed while voice recording"
                )
                return VoiceGesture.CANCEL

        if self._voice_ui_active:
            self._voice_ui_active = False
            self.last_decision_reason = (
                "another key pressed while voice session active"
            )
            return VoiceGesture.CANCEL

        if self._tap_active and name not in self._TAP_KEYS:
            self._tap_invalid = True

        if (
            not self._tap_active
            and name in self._TAP_KEYS
            and self._TAP_KEYS <= self._pressed
        ):
            self._tap_active = True
            self._tap_invalid = bool(self._pressed - self._TAP_KEYS)
            self._tap_started_at = self._clock()
        return None

    def poll(self):
        if (
            not self._voice_down
            or self._voice_blocked
            or self._voice_started
            or self._voice_started_at is None
        ):
            return None
        elapsed = self._clock() - self._voice_started_at
        if elapsed >= self.hold_threshold_ms:
            self._voice_started = True
            self._voice_ui_active = True
            self.last_decision_reason = (
                f"held {elapsed:.1f} ms; threshold "
                f"{self.hold_threshold_ms} ms"
            )
            return VoiceGesture.HOLD_START
        return None

    def key_up(self, key_name):
        name = self._normalize(key_name)
        if name == self.voice_key and self._voice_down:
            self._pressed.discard(name)
            gesture = (
                VoiceGesture.HOLD_END
                if self._voice_started and not self._voice_blocked
                else None
            )
            if gesture == VoiceGesture.HOLD_END:
                self.last_decision_reason = "voice key released after hold"
            self._reset_voice_key()
            return gesture

        if name in self._TAP_KEYS and self._tap_active:
            elapsed = self._clock() - self._tap_started_at
            valid_tap = (
                not self._tap_invalid
                and elapsed < self.hold_threshold_ms
                and not (self._pressed - self._TAP_KEYS)
            )
            self._pressed.discard(name)
            self._reset_tap()
            if valid_tap:
                self.last_decision_reason = "Ctrl+Shift+Space released before threshold"
            return VoiceGesture.TAP if valid_tap else None

        self._pressed.discard(name)
        return None

    def finish_voice_session(self):
        """Allow the global listener to stop treating keys as overlay cancel."""
        self._voice_ui_active = False

    def is_key_down(self, key_name):
        return self._normalize(key_name) in self._pressed

    @property
    def voice_session_active(self):
        return self._voice_down or self._voice_ui_active

    def _reset_voice_key(self):
        self._voice_down = False
        self._voice_blocked = False
        self._voice_started = False
        self._voice_started_at = None

    def _reset_tap(self):
        self._tap_active = False
        self._tap_invalid = False
        self._tap_started_at = None
