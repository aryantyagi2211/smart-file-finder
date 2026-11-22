"""
app/hotkey_listener.py

Registers the global Ctrl+Shift+Space tap and configurable Alt-alone voice
hold regardless of which window has focus. Uses the `keyboard` library since
PyQt6 alone can only catch key presses when its own window is focused.
"""

import keyboard
import threading
import ctypes
import os

from app.voice_hotkey import TapHoldDetector, VoiceGesture
from observability.logger import log

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


def register_tap_hold_hotkey(callback, hold_threshold_ms=500,
                             max_hold_ms=15000, voice_key="left alt",
                             cancel_event=None):
    detector = TapHoldDetector(
        hold_threshold_ms=hold_threshold_ms,
        max_hold_ms=max_hold_ms,
        voice_key=voice_key,
    )
    stopped = threading.Event()

    def event_key_name(event):
        name = str(event.name)
        scan_code = getattr(event, "scan_code", None)
        if name.casefold() == "alt":
            if scan_code in {312, 364}:
                return "right alt"
            return "left alt"
        if name.casefold() == "alt gr":
            return "right alt"
        return name

    def is_injected_unassigned_key(event):
        name = str(getattr(event, "name", "")).casefold()
        return name in {"unknown", "vk 0xe8"} and getattr(
            event, "scan_code", None
        ) in {None, 0, 0xE8}

    def dispatch(action):
        if action == VoiceGesture.HOLD_START:
            try:
                inject_unassigned_key()
            except Exception as error:
                log(f"Could not suppress the Alt menu key: {error}")
        callback(action)

    def on_event(event):
        repeated = bool(getattr(event, "is_repeat", False))
        if is_injected_unassigned_key(event):
            return
        name = event_key_name(event)
        if event.event_type == keyboard.KEY_DOWN:
            action = detector.key_down(name, repeated=repeated)
            if (
                action is None
                and not repeated
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                action = VoiceGesture.CANCEL
        else:
            action = detector.key_up(name)
        if action is not None:
            dispatch(action)

    hook = keyboard.hook(on_event)

    def poll_threshold():
        while not stopped.is_set():
            action = detector.poll()
            if action is not None:
                dispatch(action)
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

    unregister.finish_voice_session = detector.finish_voice_session
    return unregister


def inject_unassigned_key():
    """Send an unassigned key event so Windows does not activate the menu bar."""
    if os.name != "nt":
        raise OSError("Unassigned-key injection is only supported on Windows")
    user32 = ctypes.windll.user32
    user32.keybd_event(0xE8, 0, 0, 0)
    user32.keybd_event(0xE8, 0, 0x0002, 0)