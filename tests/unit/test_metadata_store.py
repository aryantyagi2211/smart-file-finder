"""
tests/unit/test_metadata_store.py

Basic unit tests for storage/metadata_store.py. Uses a temporary DB file
per test (via pytest's tmp_path fixture) so these tests never touch the
real project metadata.db.
"""

import sqlite3

from storage.metadata_store import (
    init_db,
    init_chunks_table,
    save_file_metadata,
    get_file_metadata,
    list_all_files,
    save_chunks,
    get_chunks,
)
from app_paths import normalize_path


def test_save_and_get_file_metadata(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    save_file_metadata("some/file.pdf", "text/document", db_path=db_path)
    result = get_file_metadata("some/file.pdf", db_path=db_path)

    assert result is not None
    assert result["path"] == normalize_path("some/file.pdf")
    assert result["file_type"] == "text/document"
    assert "indexed_at" in result


def test_get_file_metadata_missing(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    result = get_file_metadata("does/not/exist.pdf", db_path=db_path)
    assert result is None


def test_save_file_metadata_overwrites_on_conflict(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    save_file_metadata("file.pdf", "text/document", indexed_at="2026-01-01T00:00:00", db_path=db_path)
    save_file_metadata("file.pdf", "text/document", indexed_at="2026-02-02T00:00:00", db_path=db_path)

    result = get_file_metadata("file.pdf", db_path=db_path)
    assert result["indexed_at"] == "2026-02-02T00:00:00"


def test_list_all_files(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    save_file_metadata("a.pdf", "text/document", db_path=db_path)
    save_file_metadata("b.png", "image/video", db_path=db_path)

    all_files = list_all_files(db_path=db_path)
    paths = {f["path"] for f in all_files}

    assert paths == {normalize_path("a.pdf"), normalize_path("b.png")}


def test_save_and_get_chunks(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    init_chunks_table(db_path)

    save_chunks("doc.pdf", ["chunk one", "chunk two", "chunk three"], db_path=db_path)
    result = get_chunks("doc.pdf", db_path=db_path)

    assert result == ["chunk one", "chunk two", "chunk three"]


def test_legacy_database_migrates_vision_model_column(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE files (
                path TEXT PRIMARY KEY,
                file_type TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO files VALUES (?, ?, ?)",
            ("/tmp/legacy.png", "image", "2026-01-01T00:00:00"),
        )

    init_db(db_path)
    save_file_metadata(
        "/tmp/legacy.png",
        "image",
        vision_model="HuggingFaceTB/SmolVLM-500M-Instruct",
        db_path=db_path,
    )

    metadata = get_file_metadata("/tmp/legacy.png", db_path=db_path)
    assert metadata["vision_model"] == "HuggingFaceTB/SmolVLM-500M-Instruct"