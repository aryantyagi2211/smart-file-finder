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
import json
from pathlib import Path

APP_FOLDER_NAME = "SmartFileFinder"


def get_app_root_dir():
    """Return the base for resolving app-relative paths."""
    if getattr(sys, "frozen", False):
        return get_base_dir()
    return Path(__file__).resolve().parent


def normalize_path(path, relative_to=None):
    """Return an absolute path using consistent forward slashes."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (Path(relative_to) if relative_to is not None
                    else get_app_root_dir()) / resolved
    return resolved.resolve().as_posix()


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


def get_config_path(filename):
    """Return a config file path independent of the current working directory."""
    if getattr(sys, "frozen", False):
        config_dir = Path(sys.executable).resolve().parent / "config"
    else:
        config_dir = Path(__file__).resolve().parent / "config"

    config_path = config_dir / filename
    defaults = {
        "settings.json": {"indexed_folders": []},
        "harness_limits.json": {
            "max_reformulations": 2,
            "timeout_seconds": 5,
            "confidence_threshold": 0.55,
            "ambiguity_margin": 0.05,
            "options_floor": 0.45,
            "max_options": 3,
        },
    }
    if not config_path.exists() and filename in defaults:
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(defaults[filename], indent=2) + "\n",
            encoding="utf-8",
        )
        from observability.logger import log

        log(f"Created missing config file with defaults: {config_path}")

    return config_path


def ensure_config_exists():
    """Ensure the runtime configuration files exist with usable defaults."""
    get_config_path("settings.json")
    get_config_path("harness_limits.json")


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