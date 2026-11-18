from app.voice_hotkey import TapHoldDetector, VoiceGesture


def test_tap_hold_repeat_modifier_and_threshold():
    now = [1000.0]
    detector = TapHoldDetector(hold_threshold_ms=400, clock=lambda: now[0])

    assert detector.key_down("ctrl") is None
    assert detector.key_down("shift") is None
    assert detector.key_down("space") is None
    assert detector.key_down("space", repeated=True) is None
    now[0] = 1399.9
    assert detector.poll() is None
    now[0] = 1400.0
    assert detector.poll() == VoiceGesture.HOLD_START
    now[0] = 1450.0
    assert detector.key_up("space") == VoiceGesture.HOLD_END

    now[0] = 2000.0
    detector.key_down("ctrl")
    detector.key_down("shift")
    detector.key_down("space")
    now[0] = 2399.0
    assert detector.key_up("shift") == VoiceGesture.TAP
    assert detector._hotkey_down is False


def test_modifier_released_before_space_still_emits_tap():
    now = [0.0]
    detector = TapHoldDetector(400, clock=lambda: now[0])
    detector.key_down("ctrl")
    detector.key_down("shift")
    detector.key_down("space")
    now[0] = 399.0
    assert detector.key_up("ctrl") == VoiceGesture.TAP
    assert detector.key_up("space") is None
