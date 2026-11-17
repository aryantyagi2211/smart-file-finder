"""
indexing/folder_detection.py

Auto-detects standard Windows user folders (Desktop, Documents, Downloads)
so the app can index a user's real files without requiring manual folder
configuration. Falls back gracefully if a folder doesn't exist.
"""

import os
from pathlib import Path


def get_default_folders():
    """
    Return absolute standard user folder paths. Keeping unavailable paths
    lets cleanup protect their existing index entries until they return.
    """
    home = Path(os.path.expanduser("~"))
    candidates = [
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
    ]
    return [folder.resolve().as_posix() for folder in candidates]