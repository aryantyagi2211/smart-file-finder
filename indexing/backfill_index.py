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
import sys
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
    migrate_relative_paths,
    path_migration_complete,
    record_index_failure,
    save_file_metadata,
    save_chunks,
)
from storage.vector_db.vector_store import (
    add_item,
    count_items,
    delete_by_file,
    delete_relative_path_entries,
)

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
from app_paths import get_app_root_dir, get_base_dir, get_config_path, normalize_path

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
    relative_root = (
        get_base_dir() if getattr(sys, "frozen", False) else get_app_root_dir()
    )
    for folder in extra_folders:
        try:
            resolved_folder = normalize_path(folder, relative_to=relative_root)
        except (OSError, RuntimeError) as error:
            log(f"WARNING: Cannot resolve indexed folder {folder!r}: {error}")
            continue
        if resolved_folder.casefold() not in {item.casefold() for item in folders}:
            folders.append(resolved_folder)

    return folders


def remove_file_index(file_path, reason="file removed"):
    """Remove all searchable and metadata entries for one indexed file."""
    normalized_path = normalize_path(file_path)
    delete_by_file(normalized_path)
    delete_file(normalized_path)
    log(f"Removed index for {normalized_path}: {reason}")


def _is_within(path, folder):
    try:
        Path(path).relative_to(Path(folder))
        return True
    except ValueError:
        return False


def cleanup_missing_files(indexed_folders=None):
    """Remove indexed records whose source files no longer exist."""
    removed = []
    folders = indexed_folders if indexed_folders is not None else load_indexed_folders()
    reachable_folders = []
    unreachable_folders = []
    for folder in folders:
        normalized_folder = normalize_path(folder)
        folder_path = Path(normalized_folder)
        try:
            resolved_folder = folder_path.resolve(strict=True)
            if not resolved_folder.is_dir() or not os.access(resolved_folder, os.R_OK):
                raise OSError("folder is not a readable directory")
            reachable_folders.append(resolved_folder.as_posix())
        except (OSError, RuntimeError) as error:
            unreachable_folders.append(normalized_folder)
            log(f"WARNING: Skipping cleanup for unreachable folder {normalized_folder}: {error}")

    for metadata in list_all_files():
        file_path = metadata["path"]
        try:
            normalized_path = normalize_path(file_path)
        except (OSError, RuntimeError) as error:
            log(f"WARNING: Skipping cleanup for unresolvable indexed path {file_path!r}: {error}")
            continue
        if not Path(normalized_path).exists():
            if any(_is_within(normalized_path, folder) for folder in unreachable_folders):
                continue
            if indexed_folders is not None and not any(
                _is_within(normalized_path, folder) for folder in reachable_folders
            ) and not any(_is_within(normalized_path, folder) for folder in unreachable_folders):
                continue
            remove_file_index(normalized_path, reason="source file no longer exists")
            removed.append(normalized_path)
    return removed


def index_one_file(file_path, force=False):
    """
    Route, process, and store a single file (text or image) -- but skip
    it entirely if it's already indexed and hasn't changed since. Also
    indexes the filename itself as a separate searchable entry, so exact
    or partial filename queries can match even when content-based
    semantic search doesn't surface the file.
    """
    path_str = normalize_path(file_path)
    file_path = Path(path_str)

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

def backfill(settings_path=None, progress_callback=None, error_callback=None,
             indexed_folders=None):
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
    if not path_migration_complete():
        for legacy_path in delete_relative_path_entries():
            log(f"Removed legacy relative vector path: {legacy_path}")
        for legacy_path in migrate_relative_paths():
            log(f"Removed legacy relative metadata path and chunks: {legacy_path}")
    rebuild_index = count_items() == 0

    folders = (
        [normalize_path(folder) for folder in indexed_folders]
        if indexed_folders is not None
        else load_indexed_folders(settings_path)
    )

    # First pass: count total matching files across all folders, so we
    # can report real progress (X of Y), not a fake estimate.
    all_files = []
    for folder in folders:
        folder_path = Path(folder)
        try:
            folder_path = folder_path.resolve(strict=True)
            if not folder_path.is_dir() or not os.access(folder_path, os.R_OK):
                raise OSError("folder is not a readable directory")
        except (OSError, RuntimeError) as error:
            log(f"WARNING: Skipping backfill for unreachable folder {folder}: {error}")
            continue
        for file_path in folder_path.rglob("*"):
            if not file_path.is_file():
                continue
            if any(excluded in file_path.parts for excluded in EXCLUDED_FOLDER_NAMES):
                continue
            if file_path.suffix.lower() not in SCAN_EXTENSIONS:
                continue
            all_files.append((folder_path.name, file_path.resolve()))

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

    cleanup_missing_files(indexed_folders=folders)


if __name__ == "__main__":
    backfill()