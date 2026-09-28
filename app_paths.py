"""
app_paths.py

Resolves where the app should store its working data (databases, logs,
indexes) -- always in a user-writable folder, never inside the app's own
installation directory. This matters specifically once installed via the
Inno Setup installer to Program Files, which is read-only for normal
(non-admin) processes -- writing there fails with "attempt to write a
readonly database" errors, confirmed during real installer testing.
"""

import sys
import os
import shutil
from pathlib import Path

APP_FOLDER_NAME = "SmartFileFinder"


def get_base_dir():
    """
    Returns a writable, per-user folder for the app's working data
    (%LOCALAPPDATA%\\SmartFileFinder on Windows), regardless of whether
    running from source or as a packaged/installed .exe -- since the
    installed .exe's own folder (Program Files) is not writable.
    """
    app_data = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    base_dir = app_data / APP_FOLDER_NAME
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def ensure_config_exists():
    """
    Copies config/harness_limits.json into the user-writable app-data
    folder if it doesn't already exist there -- needed since the
    packaged .exe's own folder (or Program Files) isn't writable, and
    config files can't just live next to the .exe like they do when
    running from source.
    """
    base_dir = get_base_dir()
    dest_config_dir = base_dir / "config"
    dest_config_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_config_dir / "harness_limits.json"

    if not dest_file.exists():
        # When running from source, the real config file sits at
        # <project_root>/config/harness_limits.json
        source_file = Path(__file__).parent / "config" / "harness_limits.json"
        if source_file.exists():
            shutil.copy(source_file, dest_file)
        else:
            # Fallback: write sensible defaults if source is also missing
            dest_file.write_text('{\n  "max_reformulations": 2,\n  "timeout_seconds": 5,\n  "confidence_threshold": 0.5\n}')


def get_resource_path(filename):
    """
    Returns the path to a bundled static resource (icons, etc.) that
    ships with the app -- resolved relative to the .exe's own folder
    when frozen, or the project root when running from source.
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / filename
    else:
        return Path(__file__).parent / filename