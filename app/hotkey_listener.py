"""
app/hotkey_listener.py

Registers a global (system-wide) hotkey -- default Ctrl+Shift+Space per
CLAUDE.md -- that triggers a callback regardless of which window has
focus. Uses the `keyboard` library since PyQt6 alone can only catch key
presses when its own window is focused.
"""

import keyboard

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