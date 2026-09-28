"""
indexing/file_watcher.py

Watches a folder and prints a message whenever a file is created or
modified. This is Phase 1 plumbing only — no routing, extraction, or
storage happens here yet (that's file_router.py and beyond).
"""

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class PrintHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"[CREATED] {event.src_path}")

    def on_modified(self, event):
        if not event.is_directory:
            print(f"[MODIFIED] {event.src_path}")


def watch_folder(folder_path):
    """Start watching folder_path. Blocks until Ctrl+C."""
    event_handler = PrintHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f"Watching: {folder_path} (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    watch_folder("test_data")