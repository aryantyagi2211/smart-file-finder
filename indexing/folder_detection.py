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
    Return a list of real, existing standard user folders to index:
    Desktop, Documents, Downloads. Skips any that don't exist on this
    system rather than erroring.
    """
    home = Path(os.path.expanduser("~"))
    candidates = [
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
    ]
    return [str(folder) for folder in candidates if folder.exists()]