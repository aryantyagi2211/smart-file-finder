import time
from types import SimpleNamespace

import keyboard
import pytest

from app import hotkey_listener
from app.voice_hotkey import TapHoldDetector, VoiceGesture


def _detector(voice_key="left alt", threshold=500):
    now = [0.0]
    return TapHoldDetector(
        hold_threshold_ms=threshold,
        clock=lambda: now[0],
        voice_key=voice_key,
    ), now


def test_alt_hold_starts_at_threshold_and_ends_on_release():
    detector, now = _detector()
    assert detector.key_down("left alt") is None
    now[0] = 499.9
    assert detector.poll() is None
    now[0] = 500
    assert detector.poll() == VoiceGesture.HOLD_START
    assert detector.key_up("left alt") == VoiceGesture.HOLD_END


def test_short_alt_tap_does_nothing():
    detector, now = _detector()
    detector.key_down("left alt")
    now[0] = 499
    assert detector.poll() is None
    assert detector.key_up("left alt") is None


def test_alt_tab_and_any_pre_threshold_key_never_start_voice():
    for key in ("tab", "f4"):
        detector, now = _detector()
        detector.key_down("left alt")
        detector.key_down(key)
        now[0] = 1000
        assert detector.poll() is None
        assert detector.key_up("left alt") is None
        detector.key_up(key)


def test_another_key_during_recording_cancels():
    detector, now = _detector()
    detector.key_down("left alt")
    now[0] = 500
    assert detector.poll() == VoiceGesture.HOLD_START
    assert detector.key_down("a") == VoiceGesture.CANCEL
    assert detector.key_up("left alt") is None


@pytest.mark.parametrize("modifier", ["ctrl", "shift", "win"])
def test_modifier_held_before_alt_blocks_voice(modifier):
    detector, now = _detector()
    detector.key_down(modifier)
    detector.key_down("left alt")
    now[0] = 1000
    assert detector.poll() is None
    assert detector.key_up("left alt") is None


@pytest.mark.parametrize("modifier", ["ctrl", "shift", "win"])
def test_modifier_pressed_during_alt_hold_blocks_voice(modifier):
    detector, now = _detector()
    detector.key_down("left alt")
    detector.key_down(modifier)
    now[0] = 1000
    assert detector.poll() is None
    assert detector.key_up("left alt") is None


def test_repeated_alt_keydown_is_ignored():
    detector, now = _detector()
    detector.key_down("left alt")
    now[0] = 200
    assert detector.key_down("left alt", repeated=True) is None
    now[0] = 500
    assert detector.poll() == VoiceGesture.HOLD_START


def test_right_alt_is_ignored_by_default_but_can_be_configured():
    detector, now = _detector()
    detector.key_down("right alt")
    now[0] = 1000
    assert detector.poll() is None
    assert detector.key_up("right alt") is None

    detector, now = _detector(voice_key="right alt")
    detector.key_down("right alt")
    now[0] = 500
    assert detector.poll() == VoiceGesture.HOLD_START


def test_ctrl_shift_space_tap_still_opens_bar_but_hold_never_starts_voice():
    detector, now = _detector()
    detector.key_down("ctrl")
    detector.key_down("shift")
    detector.key_down("space")
    now[0] = 499
    assert detector.key_up("space") == VoiceGesture.TAP
    detector.key_up("shift")
    detector.key_up("ctrl")

    detector, now = _detector()
    detector.key_down("ctrl")
    detector.key_down("shift")
    detector.key_down("space")
    now[0] = 500
    assert detector.poll() is None
    assert detector.key_up("space") is None


def test_voice_overlay_keys_cancel_until_session_finishes():
    detector, now = _detector()
    detector.key_down("left alt")
    now[0] = 500
    assert detector.poll() == VoiceGesture.HOLD_START
    assert detector.key_up("left alt") == VoiceGesture.HOLD_END
    assert detector.key_down("esc") == VoiceGesture.CANCEL

    detector.finish_voice_session()
    detector, now = _detector()
    detector.key_down("left alt")
    now[0] = 500
    detector.poll()
    detector.key_up("left alt")
    detector.finish_voice_session()
    assert detector.key_down("esc") is None


def test_menu_key_injection_only_happens_on_voice_hold(monkeypatch):
    event_handler = None
    injected = []
    callbacks = []

    def fake_hook(handler):
        nonlocal event_handler
        event_handler = handler
        return "hook"

    monkeypatch.setattr(keyboard, "hook", fake_hook)
    monkeypatch.setattr(keyboard, "unhook", lambda _hook: None)
    monkeypatch.setattr(
        hotkey_listener, "inject_unassigned_key",
        lambda: injected.append(True),
    )
    unregister = hotkey_listener.register_tap_hold_hotkey(
        callbacks.append, hold_threshold_ms=30
    )
    try:
        event_handler(SimpleNamespace(
            name="left alt", scan_code=56,
            event_type=keyboard.KEY_DOWN, is_repeat=False,
        ))
        time.sleep(0.01)
        event_handler(SimpleNamespace(
            name="left alt", scan_code=56,
            event_type=keyboard.KEY_UP, is_repeat=False,
        ))
        assert callbacks == []
        assert injected == []

        event_handler(SimpleNamespace(
            name="left alt", scan_code=56,
            event_type=keyboard.KEY_DOWN, is_repeat=False,
        ))
        event_handler(SimpleNamespace(
            name="tab", scan_code=15,
            event_type=keyboard.KEY_DOWN, is_repeat=False,
        ))
        time.sleep(0.08)
        assert callbacks == []
        assert injected == []
        event_handler(SimpleNamespace(
            name="left alt", scan_code=56,
            event_type=keyboard.KEY_UP, is_repeat=False,
        ))
        event_handler(SimpleNamespace(
            name="tab", scan_code=15,
            event_type=keyboard.KEY_UP, is_repeat=False,
        ))

        event_handler(SimpleNamespace(
            name="left alt", scan_code=56,
            event_type=keyboard.KEY_DOWN, is_repeat=False,
        ))
        time.sleep(0.08)
        assert callbacks == [VoiceGesture.HOLD_START]
        assert injected == [True]
        event_handler(SimpleNamespace(
            name="left alt", scan_code=56,
            event_type=keyboard.KEY_UP, is_repeat=False,
        ))
        assert callbacks[-1] == VoiceGesture.HOLD_END

        callbacks.clear()
        unregister.finish_voice_session()
    finally:
        unregister()
