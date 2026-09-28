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
from storage.metadata_store import init_db, init_chunks_table, save_file_metadata, save_chunks
from storage.vector_db.vector_store import add_item, count_items

import os
from datetime import datetime
from storage.metadata_store import get_file_metadata
from indexing.file_router import (
    IMAGE_EXTENSIONS,
    TEXT_DOCUMENT_EXTENSIONS,
    UNSUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_DOCUMENT_EXTENSIONS
SCAN_EXTENSIONS = SUPPORTED_EXTENSIONS | VIDEO_EXTENSIONS | UNSUPPORTED_EXTENSIONS
EXCLUDED_FOLDER_NAMES = {".git", "node_modules", "venv", "__pycache__", ".venv", "env"}

SETTINGS_PATH = Path("config/settings.json")


def load_indexed_folders(settings_path=SETTINGS_PATH):
    """
    Return folders to index: auto-detected standard user folders
    (Desktop, Documents, Downloads) always included, plus any additional
    folders explicitly listed in config/settings.json (optional, for
    manual additions beyond the defaults).
    """
    from indexing.folder_detection import get_default_folders

    folders = get_default_folders()

    try:
        with open(settings_path, encoding="utf-8-sig") as f:
            settings = json.load(f)
        extra_folders = settings.get("indexed_folders", [])
        for folder in extra_folders:
            if folder not in folders:
                folders.append(folder)
    except FileNotFoundError:
        pass

    return folders

def index_one_file(file_path, force=False):
    """
    Route, process, and store a single file (text or image) -- but skip
    it entirely if it's already indexed and hasn't changed since. Also
    indexes the filename itself as a separate searchable entry, so exact
    or partial filename queries can match even when content-based
    semantic search doesn't surface the file.
    """
    path_str = str(file_path)

    if not force:
        existing = get_file_metadata(path_str)
        if existing is not None:
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            if file_mtime <= existing["indexed_at"]:
                return f"[SKIPPED - unchanged] {path_str}"

    category = route_file(file_path)

    if category in {"unsupported", "video"}:
        return f"[UNSUPPORTED - {category}] {path_str}"
    if category == "unknown":
        return f"[SKIPPED - unknown type] {path_str}"

    # Filename entry -- separate from content, so a query matching the
    # filename itself (e.g. "MySQL Handbook") can surface the file even
    # if its content embedding alone wouldn't score highly enough.
    filename = Path(file_path).stem  # filename without extension
    filename_vector = embed_text(filename)
    add_item(f"{path_str}::filename", filename_vector, metadata={
        "path": path_str,
        "match_type": "filename",
        "chunk_text": filename
    })

    if category == "text/document":
        chunks = process_document(file_path)
        save_file_metadata(path_str, category)
        save_chunks(path_str, chunks)

        vectors = embed_texts(chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            add_item(f"{path_str}::{i}", vector, metadata={
                "path": path_str,
                "chunk_index": i,
                "chunk_text": chunk[:200],
                "match_type": "content"
            })

        # Document-level summary via Qwen3-1.7B -- DISABLED for now.
        # Empirically tested (see decisions log): summary embeddings never
        # outperformed existing content chunks or filename matches across
        # multiple real queries (vague and specific), likely because a
        # single broad summary vector gets diluted against hundreds of
        # narrow, specific chunk embeddings in dense retrieval. No
        # measurable search-quality benefit found, but real added cost
        # (several seconds/document indexing time, extra VRAM for a third
        # model). Code kept for potential future refinement -- not deleted.
        #
        # try:
        #     from indexing.summarizer import summarize_document
        #     summary = summarize_document(chunks)
        #     if summary:
        #         summary_vector = embed_text(summary)
        #         add_item(f"{path_str}::summary", summary_vector, metadata={
        #             "path": path_str,
        #             "match_type": "summary",
        #             "chunk_text": summary[:200]
        #         })
        # except Exception as e:
        #     print(f"[WARNING] Summary generation failed for {path_str}: {e}")

        return f"[INDEXED - text] {path_str} -> {len(chunks)} chunks"

    elif category == "image":
        description = describe_image(path_str)
        vector = embed_text(description)
        add_item(path_str, vector, metadata={
            "path": path_str,
            "description": description,
            "match_type": "content"
        })
        save_file_metadata(path_str, category)
        return f"[INDEXED - image] {path_str} -> \"{description}\""
    
    elif category == "video":
        return f"[UNSUPPORTED - video not yet processed] {path_str}"

    else:
        return f"[SKIPPED - unknown type] {path_str}"

def backfill(settings_path=SETTINGS_PATH, progress_callback=None, error_callback=None):
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
        except ValueError as e:
            # ValueError specifically means "unsupported file type" (raised
            # by text_pipeline.py's process_document) -- not a real failure,
            # just something we don't support yet. Don't alarm the user
            # with a red error icon for this.
            status = f"[UNSUPPORTED] {file_path} -> {e}"
        except Exception as e:
            # A genuine, unexpected failure -- this IS worth flagging to
            # the user with the red error indicator.
            status = f"[ERROR] {file_path} -> {e}"
            if error_callback:
                error_callback(short_name)

        results.append(status)
        print(status)


if __name__ == "__main__":
    backfill()