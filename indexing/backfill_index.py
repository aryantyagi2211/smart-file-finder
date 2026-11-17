"""
indexing/backfill_index.py

Scans all folders listed in config/settings.json (recursively) and indexes
every file found -- both text/document and image/video -- into
metadata_store + vector_db. This is the "first-run full scan" a real
installed app needs, since indexing/file_watcher.py only catches
new/changed files going forward, not files that already existed before
the app started watching.

Safe to re-run: metadata_store.save_chunks() clears old chunks per-path
first, and vector_store.add_item() uses upsert, so re-indexing the same
file just refreshes it rather than duplicating.
"""

import json
from pathlib import Path

from indexing.file_router import route_file
from indexing.text_pipeline import process_document
from indexing.vision_pipeline import describe_image
from indexing.embedder import embed_text, embed_texts
from storage.metadata_store import (
    delete_file,
    init_db,
    init_chunks_table,
    list_all_files,
    record_index_failure,
    save_file_metadata,
    save_chunks,
)
from storage.vector_db.vector_store import add_item, count_items, delete_by_file

import os
from datetime import datetime
from storage.metadata_store import get_file_metadata
from indexing.file_router import (
    IMAGE_EXTENSIONS,
    TEXT_DOCUMENT_EXTENSIONS,
    UNSUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from observability.logger import log
from app_paths import get_config_path

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_DOCUMENT_EXTENSIONS
SCAN_EXTENSIONS = SUPPORTED_EXTENSIONS | VIDEO_EXTENSIONS | UNSUPPORTED_EXTENSIONS
EXCLUDED_FOLDER_NAMES = {".git", "node_modules", "venv", "__pycache__", ".venv", "env"}

SETTINGS_PATH = get_config_path("settings.json")
MAX_INDEX_FAILURES = 3


def load_indexed_folders(settings_path=None):
    """
    Return folders to index: auto-detected standard user folders
    (Desktop, Documents, Downloads) always included, plus any additional
    folders explicitly listed in config/settings.json (optional, for
    manual additions beyond the defaults).
    """
    from indexing.folder_detection import get_default_folders

    if settings_path is None:
        settings_path = get_config_path("settings.json")
    else:
        settings_path = Path(settings_path)
        if not settings_path.exists():
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(
                json.dumps({"indexed_folders": []}, indent=2) + "\n",
                encoding="utf-8",
            )
            log(f"Created missing config file with defaults: {settings_path}")

    folders = get_default_folders()
    with open(settings_path, encoding="utf-8-sig") as f:
        settings = json.load(f)
    extra_folders = settings.get("indexed_folders", [])
    for folder in extra_folders:
        if folder not in folders:
            folders.append(folder)

    return folders


def remove_file_index(file_path, reason="file removed"):
    """Remove all searchable and metadata entries for one indexed file."""
    delete_by_file(file_path)
    delete_file(file_path)
    log(f"Removed index for {file_path}: {reason}")


def cleanup_missing_files():
    """Remove indexed records whose source files no longer exist."""
    removed = []
    for metadata in list_all_files():
        file_path = metadata["path"]
        if not Path(file_path).exists():
            remove_file_index(file_path, reason="source file no longer exists")
            removed.append(file_path)
    return removed


def index_one_file(file_path, force=False):
    """
    Route, process, and store a single file (text or image) -- but skip
    it entirely if it's already indexed and hasn't changed since. Also
    indexes the filename itself as a separate searchable entry, so exact
    or partial filename queries can match even when content-based
    semantic search doesn't surface the file.
    """
    path_str = str(file_path)

    category = route_file(file_path)
    if category in {"unsupported", "video"}:
        return f"[UNSUPPORTED - {category}] {path_str}"
    if category == "unknown":
        return f"[SKIPPED - unknown type] {path_str}"

    init_db()
    init_chunks_table()
    existing = get_file_metadata(path_str)
    file_mtime_ns = os.stat(file_path).st_mtime_ns
    previous_failures = 0
    if (
        existing is not None
        and existing.get("file_mtime_ns") is None
        and existing.get("failure_mtime_ns") == file_mtime_ns
        and existing.get("fail_count", 0) >= MAX_INDEX_FAILURES
    ):
        log(
            f"Skipping {path_str}: indexing failed {existing['fail_count']} "
            "consecutive times; waiting for file modification"
        )
        return f"[SKIPPED - failed {existing['fail_count']} times] {path_str}"

    if existing is not None and not force:
        indexed_mtime_ns = existing.get("file_mtime_ns")
        if indexed_mtime_ns is None:
            failure_mtime_ns = existing.get("failure_mtime_ns")
            failure_count = existing.get("fail_count", 0)
            if failure_mtime_ns == file_mtime_ns and failure_count:
                previous_failures = failure_count
            elif failure_mtime_ns is None and not failure_count:
                # Migrate legacy indexed rows without needlessly re-indexing
                # files whose indexed timestamp is at least as recent as the file.
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                if file_mtime <= existing["indexed_at"]:
                    save_file_metadata(
                        path_str,
                        existing["file_type"],
                        indexed_at=existing["indexed_at"],
                        file_mtime_ns=file_mtime_ns,
                    )
                    return f"[SKIPPED - unchanged] {path_str}"
                remove_file_index(path_str, reason="file changed")
            else:
                remove_file_index(path_str, reason="file changed")
        elif file_mtime_ns == indexed_mtime_ns:
            return f"[SKIPPED - unchanged] {path_str}"
        else:
            remove_file_index(path_str, reason="file changed")
    elif existing is not None:
        if existing.get("file_mtime_ns") is not None:
            remove_file_index(path_str, reason="forced re-index")
        elif existing.get("failure_mtime_ns") != file_mtime_ns:
            remove_file_index(path_str, reason="file changed")
        else:
            previous_failures = existing.get("fail_count", 0)

    try:
        if category == "text/document":
            chunks = process_document(file_path)
            vectors = embed_texts(chunks)

            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                add_item(f"{path_str}::{i}", vector, metadata={
                    "path": path_str,
                    "chunk_index": i,
                    "chunk_text": chunk[:200],
                    "match_type": "content"
                })

            save_chunks(path_str, chunks)
        elif category == "image":
            description = describe_image(path_str)
            vector = embed_text(description)
            add_item(path_str, vector, metadata={
                "path": path_str,
                "description": description,
                "match_type": "content"
            })
            chunks = None
        else:
            raise ValueError(f"Unsupported file type for indexing: {category}")

        filename = Path(file_path).stem
        filename_vector = embed_text(filename)
        add_item(f"{path_str}::filename", filename_vector, metadata={
            "path": path_str,
            "match_type": "filename",
            "chunk_text": filename
        })

        save_file_metadata(path_str, category, file_mtime_ns=file_mtime_ns)
    except Exception as error:
        log(f"[ERROR] Failed to index {path_str}: {error}")
        remove_file_index(path_str, reason=f"indexing failed: {error}")
        record_index_failure(
            path_str,
            category,
            previous_failures + 1,
            file_mtime_ns,
        )
        raise

    if category == "text/document":
        return f"[INDEXED - text] {path_str} -> {len(chunks)} chunks"
    return f"[INDEXED - image] {path_str} -> \"{description}\""

def backfill(settings_path=None, progress_callback=None, error_callback=None):
    """
    Recursively scan every folder listed in settings.json and index all
    supported files found.

    progress_callback(current_folder_name, current_filename_short,
                       files_done, files_total) -- called before each
    file is processed, giving real (not estimated) progress.

    error_callback(filename_short) -- called whenever a file fails to
    index, so the UI can show a clear error state.
    """
    init_db()
    init_chunks_table()
    rebuild_index = count_items() == 0

    folders = load_indexed_folders(settings_path)

    # First pass: count total matching files across all folders, so we
    # can report real progress (X of Y), not a fake estimate.
    all_files = []
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue
        for file_path in folder_path.rglob("*"):
            if not file_path.is_file():
                continue
            if any(excluded in file_path.parts for excluded in EXCLUDED_FOLDER_NAMES):
                continue
            if file_path.suffix.lower() not in SCAN_EXTENSIONS:
                continue
            all_files.append((folder_path.name, file_path))

    total = len(all_files)
    results = []

    for i, (folder_name, file_path) in enumerate(all_files):
        short_name = file_path.stem
        short_name = " ".join(short_name.split()[:3])  # first 2-3 words only

        if progress_callback:
            progress_callback(folder_name, short_name, i, total)

        try:
            status = index_one_file(file_path, force=rebuild_index)
            if "SKIPPED - unknown type" in status:
                pass  # not an error, just an unsupported file type, no red icon
        except Exception as e:
            # A genuine, unexpected failure -- this IS worth flagging to
            # the user with the red error indicator.
            status = f"[ERROR] {file_path} -> {e}"
            if error_callback:
                error_callback(short_name)

        results.append(status)
        print(status)

    cleanup_missing_files()


if __name__ == "__main__":
    backfill()