"""Prevent multiple app processes from sharing the persistent index."""

import os

from app_paths import get_base_dir


_lock_file = None


def acquire_single_instance_lock():
    """Return True when this process owns the app lock, otherwise False."""
    global _lock_file
    if _lock_file is not None:
        return True

    lock_path = get_base_dir() / "app.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "a+b")
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)

    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        return False

    _lock_file = lock_file
    return True