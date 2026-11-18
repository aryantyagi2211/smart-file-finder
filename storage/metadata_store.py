"""SQLite persistence for indexed file metadata and extracted text chunks."""

import sqlite3
from datetime import datetime
from pathlib import Path

from app_paths import get_base_dir, normalize_path


DB_PATH = get_base_dir() / "storage" / "metadata.db"
PATH_MIGRATION_NAME = "absolute_file_paths_v1"


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
                indexed_at TEXT NOT NULL,
                file_mtime_ns INTEGER,
                fail_count INTEGER NOT NULL DEFAULT 0,
                failure_mtime_ns INTEGER
            )
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(files)")
        }
        if "file_mtime_ns" not in columns:
            connection.execute("ALTER TABLE files ADD COLUMN file_mtime_ns INTEGER")
        if "fail_count" not in columns:
            connection.execute(
                "ALTER TABLE files ADD COLUMN fail_count INTEGER NOT NULL DEFAULT 0"
            )
        if "failure_mtime_ns" not in columns:
            connection.execute("ALTER TABLE files ADD COLUMN failure_mtime_ns INTEGER")
        if "vision_model" not in columns:
            connection.execute("ALTER TABLE files ADD COLUMN vision_model TEXT")


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


def path_migration_complete(db_path=DB_PATH):
    with _connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name TEXT PRIMARY KEY, completed_at TEXT NOT NULL)"
        )
        return connection.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?",
            (PATH_MIGRATION_NAME,),
        ).fetchone() is not None


def migrate_relative_paths(db_path=DB_PATH):
    """Remove legacy relative file/chunk paths and mark the migration done."""
    init_db(db_path)
    init_chunks_table(db_path)
    with _connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name TEXT PRIMARY KEY, completed_at TEXT NOT NULL)"
        )
        if connection.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?",
            (PATH_MIGRATION_NAME,),
        ).fetchone():
            return []

        paths = {
            row[0] for row in connection.execute(
                "SELECT path FROM files UNION SELECT path FROM chunks"
            )
        }
        relative_paths = [path for path in paths if not Path(path).is_absolute()]
        for path in relative_paths:
            connection.execute("DELETE FROM chunks WHERE path = ?", (path,))
            connection.execute("DELETE FROM files WHERE path = ?", (path,))
        connection.execute(
            "INSERT INTO schema_migrations (name, completed_at) VALUES (?, ?)",
            (PATH_MIGRATION_NAME, datetime.now().isoformat()),
        )
    return relative_paths


def save_file_metadata(path, file_type, indexed_at=None, db_path=DB_PATH,
                       file_mtime_ns=None, vision_model=None):
    init_db(db_path)
    timestamp = indexed_at or datetime.now().isoformat()
    normalized_path = normalize_path(path)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO files (
                path, file_type, indexed_at, file_mtime_ns, vision_model
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_type = excluded.file_type,
                indexed_at = excluded.indexed_at,
                file_mtime_ns = COALESCE(excluded.file_mtime_ns, files.file_mtime_ns),
                fail_count = 0,
                failure_mtime_ns = NULL,
                vision_model = excluded.vision_model
            """,
            (normalized_path, file_type, timestamp, file_mtime_ns, vision_model),
        )


def get_file_metadata(path, db_path=DB_PATH):
    init_db(db_path)
    normalized_path = normalize_path(path)
    with _connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT path, file_type, indexed_at, file_mtime_ns,
                   fail_count, failure_mtime_ns, vision_model
            FROM files WHERE path = ?
            """,
            (normalized_path,),
        ).fetchone()
    return dict(row) if row is not None else None


def list_all_files(db_path=DB_PATH):
    init_db(db_path)
    with _connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT path, file_type, indexed_at, vision_model FROM files ORDER BY path"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_file(path, db_path=DB_PATH):
    """Delete a file's metadata and all extracted chunks."""
    init_db(db_path)
    init_chunks_table(db_path)
    normalized_path = normalize_path(path)
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM chunks WHERE path = ?", (normalized_path,))
        connection.execute("DELETE FROM files WHERE path = ?", (normalized_path,))


def record_index_failure(path, file_type, fail_count, failure_mtime_ns,
                         db_path=DB_PATH):
    """Record consecutive failures without marking the file as indexed."""
    init_db(db_path)
    timestamp = datetime.now().isoformat()
    normalized_path = normalize_path(path)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO files (
                path, file_type, indexed_at, file_mtime_ns,
                fail_count, failure_mtime_ns
            )
            VALUES (?, ?, ?, NULL, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_type = excluded.file_type,
                indexed_at = excluded.indexed_at,
                file_mtime_ns = NULL,
                fail_count = excluded.fail_count,
                failure_mtime_ns = excluded.failure_mtime_ns
            """,
            (normalized_path, file_type, timestamp, fail_count, failure_mtime_ns),
        )


def save_chunks(path, chunks, db_path=DB_PATH):
    normalized_path = normalize_path(path)
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM chunks WHERE path = ?", (normalized_path,))
        connection.executemany(
            "INSERT INTO chunks (path, chunk_index, chunk_text) VALUES (?, ?, ?)",
            [(normalized_path, index, chunk) for index, chunk in enumerate(chunks)],
        )


def get_chunks(path, db_path=DB_PATH):
    normalized_path = normalize_path(path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT chunk_text FROM chunks WHERE path = ? ORDER BY chunk_index",
            (normalized_path,),
        ).fetchall()
    return [row[0] for row in rows]