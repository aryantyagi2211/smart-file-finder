"""
indexing/pipeline_runner.py

Phase 1 integration: file_watcher -> file_router -> text_pipeline ->
metadata_store, for text/document files only. Image/video files are
detected and routed but not processed yet (that's Phase 2).
"""

import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from indexing.vision_pipeline import describe_image
from indexing.embedder import embed_text
from storage.vector_db.vector_store import add_item

from indexing.file_router import route_file
from indexing.text_pipeline import process_document
from storage.metadata_store import init_db, init_chunks_table, save_file_metadata, save_chunks


class PipelineHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def _handle(self, file_path):
        category = route_file(file_path)

        if category == "text/document":
            try:
                chunks = process_document(file_path)
                save_file_metadata(file_path, category)
                save_chunks(file_path, chunks)
                print(f"[INDEXED] {file_path} -> {len(chunks)} chunks")
            except Exception as e:
                print(f"[ERROR] {file_path} -> {e}")
        elif category == "image":
            try:
                description = describe_image(file_path)
                vector = embed_text(description)
                add_item(file_path, vector, metadata={
                    "path": file_path,
                    "description": description
                })
                save_file_metadata(file_path, category)
                print(f"[INDEXED - image] {file_path} -> \"{description}\"")
            except Exception as e:
                print(f"[ERROR] {file_path} -> {e}")
        elif category == "video":
            print(f"[UNSUPPORTED - video not yet processed] {file_path}")
        else:
            print(f"[SKIPPED - unknown type] {file_path}")


def run(folder_path="test_data"):
    init_db()
    init_chunks_table()

    event_handler = PipelineHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f"Pipeline running on: {folder_path} (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run()