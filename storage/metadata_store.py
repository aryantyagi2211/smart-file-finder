"""SQLite persistence for indexed file metadata and extracted text chunks."""

import sqlite3
from datetime import datetime
from pathlib import Path

from app_paths import get_base_dir


DB_PATH = get_base_dir() / "storage" / "metadata.db"


def _connect(db_path=DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(db_path=DB_PATH):
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                file_type TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )


def init_chunks_table(db_path=DB_PATH):
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_text TEXT NOT NULL,
                UNIQUE (path, chunk_index)
            )
            """
        )


def save_file_metadata(path, file_type, indexed_at=None, db_path=DB_PATH):
    timestamp = indexed_at or datetime.now().isoformat()
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO files (path, file_type, indexed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_type = excluded.file_type,
                indexed_at = excluded.indexed_at
            """,
            (str(path), file_type, timestamp),
        )


def get_file_metadata(path, db_path=DB_PATH):
    with _connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT path, file_type, indexed_at FROM files WHERE path = ?",
            (str(path),),
        ).fetchone()
    return dict(row) if row is not None else None


def list_all_files(db_path=DB_PATH):
    with _connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT path, file_type, indexed_at FROM files ORDER BY path"
        ).fetchall()
    return [dict(row) for row in rows]


def save_chunks(path, chunks, db_path=DB_PATH):
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM chunks WHERE path = ?", (str(path),))
        connection.executemany(
            "INSERT INTO chunks (path, chunk_index, chunk_text) VALUES (?, ?, ?)",
            [(str(path), index, chunk) for index, chunk in enumerate(chunks)],
        )


def get_chunks(path, db_path=DB_PATH):
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT chunk_text FROM chunks WHERE path = ? ORDER BY chunk_index",
            (str(path),),
        ).fetchall()
    return [row[0] for row in rows]