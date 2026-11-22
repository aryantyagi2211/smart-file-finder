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
from observability.logger import log_voice, log_voice_exception

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
        scan_code = getattr(event, "scan_code", None)
        return scan_code == -0xE8 or (
            name in {"unknown", "vk 0xe8"}
            and scan_code in {None, 0, 0xE8}
        )

    def dispatch(action, reason):
        log_voice(
            f"detector decision={action.value} reason={reason or 'unspecified'}"
        )
        if action == VoiceGesture.HOLD_START:
            try:
                log_voice("injecting menu-suppression key VK 0xE8")
                inject_unassigned_key()
            except Exception as error:
                log_voice_exception(
                    f"could not suppress the Alt menu key: {error}"
                )
        try:
            callback(action)
        except Exception:
            log_voice_exception(
                f"voice gesture callback failed for {action.value}"
            )

    def on_event(event):
        try:
            repeated = bool(getattr(event, "is_repeat", False))
            name = event_key_name(event)
            injected = is_injected_unassigned_key(event)
            scan_code = getattr(event, "scan_code", None)
            safe_event_names = {
                detector.voice_key,
                "left alt",
                "right alt",
                "alt",
                "alt gr",
                "ctrl",
                "shift",
                "win",
                "space",
                "tab",
                "esc",
                "enter",
                "f4",
            }
            if injected or name in safe_event_names or detector.voice_session_active:
                display_name = name
                display_scan_code = scan_code
                if (
                    detector.voice_session_active
                    and len(name) == 1
                    and name.isprintable()
                    and not injected
                ):
                    display_name = "<printable key>"
                    display_scan_code = "<redacted>"
                log_voice(
                    f"key event type={event.event_type} name={display_name!r} "
                    f"scan_code={display_scan_code!r} injected={injected} "
                    f"repeat={repeated}"
                )
            if injected:
                return

            if event.event_type == keyboard.KEY_DOWN:
                was_pressed = detector.is_key_down(name)
                action = detector.key_down(name, repeated=repeated)
                reason = detector.last_decision_reason
                if (
                    action is None
                    and not repeated
                    and not was_pressed
                    and cancel_event is not None
                    and cancel_event.is_set()
                ):
                    action = VoiceGesture.CANCEL
                    reason = "key pressed while voice overlay was active"
            else:
                action = detector.key_up(name)
                reason = detector.last_decision_reason
            if action is not None:
                dispatch(action, reason)
        except Exception:
            log_voice_exception("keyboard hook callback failed")

    try:
        hook = keyboard.hook(on_event)
    except Exception:
        log_voice_exception(f"could not install hook for {voice_key!r}")
        raise
    log_voice(f"hook installed voice_key={voice_key!r}")

    def poll_threshold():
        try:
            while not stopped.is_set():
                action = detector.poll()
                if action is not None:
                    dispatch(action, detector.last_decision_reason)
                stopped.wait(0.025)
        except Exception:
            log_voice_exception("voice hold polling thread failed")
            stopped.set()

    polling_thread = threading.Thread(
        target=poll_threshold,
        daemon=True,
        name="voice-hotkey-poll",
    )
    polling_thread.start()
    log_voice(f"hold polling thread started voice_key={voice_key!r}")

    def unregister():
        stopped.set()
        keyboard.unhook(hook)
        log_voice(f"hook unregistered voice_key={voice_key!r}")

    unregister.finish_voice_session = detector.finish_voice_session
    return unregister


def inject_unassigned_key():
    """Send an unassigned key event so Windows does not activate the menu bar."""
    if os.name != "nt":
        raise OSError("Unassigned-key injection is only supported on Windows")
    user32 = ctypes.windll.user32
    user32.keybd_event(0xE8, 0, 0, 0)
    user32.keybd_event(0xE8, 0, 0x0002, 0)