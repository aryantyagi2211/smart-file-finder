"""
app/hotkey_listener.py

Registers a global (system-wide) hotkey -- default Ctrl+Shift+Space per
CLAUDE.md -- that triggers a callback regardless of which window has
focus. Uses the `keyboard` library since PyQt6 alone can only catch key
presses when its own window is focused.
"""

import keyboard
import threading
from app.voice_hotkey import TapHoldDetector

DEFAULT_HOTKEY = "ctrl+shift+space"


def register_hotkey(callback, hotkey=DEFAULT_HOTKEY):
    """
    Register a global hotkey that calls `callback` (no arguments) when
    pressed. Returns a handle that can be passed to unregister_hotkey().
    """
    return keyboard.add_hotkey(hotkey, callback)


def unregister_hotkey(handle):
    """Remove a previously registered hotkey."""
    keyboard.remove_hotkey(handle)


def wait_forever():
    """Block the current thread, keeping hotkey listening active."""
    keyboard.wait()


def register_tap_hold_hotkey(callback, hold_threshold_ms=400,
                             max_hold_ms=15000):
    detector = TapHoldDetector(
        hold_threshold_ms=hold_threshold_ms,
        max_hold_ms=max_hold_ms,
    )
    stopped = threading.Event()

    def on_event(event):
        repeated = bool(getattr(event, "is_repeat", False))
        if event.event_type == keyboard.KEY_DOWN:
            action = detector.key_down(event.name, repeated=repeated)
        else:
            action = detector.key_up(event.name)
        if action is not None:
            callback(action)

    hook = keyboard.hook(on_event)

    def poll_threshold():
        while not stopped.is_set():
            action = detector.poll()
            if action is not None:
                callback(action)
            stopped.wait(0.025)

    polling_thread = threading.Thread(
        target=poll_threshold,
        daemon=True,
        name="voice-hotkey-poll",
    )
    polling_thread.start()

    def unregister():
        stopped.set()
        keyboard.unhook(hook)

    return unregister