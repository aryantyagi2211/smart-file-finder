import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from indexing import backfill_index
from storage import metadata_store
from storage.vector_db import vector_store


class MemoryCollection:
    def __init__(self):
        self.items = {}

    def upsert(self, ids, embeddings, metadatas):
        for item_id, embedding, metadata in zip(ids, embeddings, metadatas):
            self.items[item_id] = {
                "embedding": embedding,
                "metadata": metadata,
            }

    def delete(self, where=None):
        if where is None:
            return
        path = where.get("path")
        self.items = {
            item_id: item
            for item_id, item in self.items.items()
            if item["metadata"].get("path") != path
        }


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    db_path = tmp_path / "metadata.db"
    metadata_store.init_db(db_path)
    metadata_store.init_chunks_table(db_path)
    collection = MemoryCollection()

    monkeypatch.setattr(vector_store, "_get_collection", lambda: collection)
    monkeypatch.setattr(backfill_index, "init_db", lambda: None)
    monkeypatch.setattr(backfill_index, "init_chunks_table", lambda: None)
    monkeypatch.setattr(
        backfill_index,
        "get_file_metadata",
        lambda path: metadata_store.get_file_metadata(path, db_path=db_path),
    )
    monkeypatch.setattr(
        backfill_index,
        "list_all_files",
        lambda: metadata_store.list_all_files(db_path=db_path),
    )
    monkeypatch.setattr(
        backfill_index,
        "save_file_metadata",
        lambda path, file_type, indexed_at=None, file_mtime_ns=None: (
            metadata_store.save_file_metadata(
                path,
                file_type,
                indexed_at=indexed_at,
                db_path=db_path,
                file_mtime_ns=file_mtime_ns,
            )
        ),
    )
    monkeypatch.setattr(
        backfill_index,
        "save_chunks",
        lambda path, chunks: metadata_store.save_chunks(path, chunks, db_path=db_path),
    )
    monkeypatch.setattr(
        backfill_index,
        "record_index_failure",
        lambda path, file_type, fail_count, failure_mtime_ns: (
            metadata_store.record_index_failure(
                path,
                file_type,
                fail_count,
                failure_mtime_ns,
                db_path=db_path,
            )
        ),
    )
    monkeypatch.setattr(
        backfill_index,
        "delete_file",
        lambda path: metadata_store.delete_file(path, db_path=db_path),
    )
    monkeypatch.setattr(backfill_index, "log", lambda message: None)
    monkeypatch.setattr(
        backfill_index,
        "process_document",
        lambda path: Path(path).read_text(encoding="utf-8").split("|"),
    )
    monkeypatch.setattr(backfill_index, "embed_text", lambda text: [float(len(text))])
    monkeypatch.setattr(
        backfill_index,
        "embed_texts",
        lambda texts: [[float(len(text))] for text in texts],
    )

    return db_path, collection


def test_changed_file_removes_old_chunks_and_indexes_new_content(
    tmp_path, isolated_index
):
    db_path, collection = isolated_index
    file_path = tmp_path / "changing.txt"
    file_path.write_text("old_unique|old_second_chunk", encoding="utf-8")

    backfill_index.index_one_file(file_path)
    old_mtime_ns = file_path.stat().st_mtime_ns
    assert f"{file_path}::1" in collection.items

    file_path.write_text("new_unique", encoding="utf-8")
    file_path.touch()
    changed_mtime_ns = old_mtime_ns + 2_000_000_000
    os.utime(file_path, ns=(changed_mtime_ns, changed_mtime_ns))

    backfill_index.index_one_file(file_path)

    assert f"{file_path}::1" not in collection.items
    assert f"{file_path}::0" in collection.items
    indexed_content = [
        item["metadata"].get("chunk_text")
        for item in collection.items.values()
        if item["metadata"].get("match_type") == "content"
    ]
    assert "old_unique" not in indexed_content
    assert "new_unique" in indexed_content
    assert metadata_store.get_chunks(file_path, db_path=db_path) == ["new_unique"]


def test_deleted_file_event_removes_vectors_metadata_and_chunks(
    tmp_path, isolated_index
):
    db_path, collection = isolated_index
    file_path = tmp_path / "deleted.txt"
    file_path.write_text("deleted_unique", encoding="utf-8")
    backfill_index.index_one_file(file_path)

    from app.main import LiveIndexHandler

    handler = LiveIndexHandler()
    handler.on_deleted(
        SimpleNamespace(is_directory=False, src_path=str(file_path))
    )

    assert not any(
        item["metadata"].get("path") == str(file_path)
        for item in collection.items.values()
    )
    assert metadata_store.get_file_metadata(file_path, db_path=db_path) is None
    assert metadata_store.get_chunks(file_path, db_path=db_path) == []


def test_cleanup_removes_files_missing_from_disk(tmp_path, isolated_index):
    db_path, collection = isolated_index
    file_path = tmp_path / "closed_while_app_was_off.txt"
    file_path.write_text("missing_unique", encoding="utf-8")
    backfill_index.index_one_file(file_path)
    file_path.unlink()

    removed = backfill_index.cleanup_missing_files()

    assert str(file_path) in removed
    assert not any(
        item["metadata"].get("path") == str(file_path)
        for item in collection.items.values()
    )
    assert metadata_store.get_file_metadata(file_path, db_path=db_path) is None
    assert metadata_store.get_chunks(file_path, db_path=db_path) == []


def test_move_event_removes_old_path_and_indexes_new_path(
    tmp_path, isolated_index
):
    db_path, collection = isolated_index
    old_path = tmp_path / "old_name.txt"
    new_path = tmp_path / "new_name.txt"
    old_path.write_text("moved_unique", encoding="utf-8")
    backfill_index.index_one_file(old_path)
    new_path.write_text("moved_unique", encoding="utf-8")

    from app.main import LiveIndexHandler

    LiveIndexHandler().on_moved(
        SimpleNamespace(
            is_directory=False,
            src_path=str(old_path),
            dest_path=str(new_path),
        )
    )

    assert metadata_store.get_file_metadata(old_path, db_path=db_path) is None
    assert metadata_store.get_file_metadata(new_path, db_path=db_path) is not None
    assert not any(
        item["metadata"].get("path") == str(old_path)
        for item in collection.items.values()
    )
    assert any(
        item["metadata"].get("path") == str(new_path)
        for item in collection.items.values()
    )
    assert str(old_path) not in [
        file["path"] for file in metadata_store.list_all_files(db_path=db_path)
    ]


def test_live_watcher_ignores_unsupported_and_git_paths(isolated_index):
    _, collection = isolated_index
    from app.main import LiveIndexHandler

    handler = LiveIndexHandler()
    handler._index(r"C:\workspace\notes.rtf")
    handler._index(r"C:\workspace\.git\ignored.txt")

    assert collection.items == {}


def test_embedding_failure_cleans_partial_data_and_does_not_mark_indexed(
    tmp_path, isolated_index, monkeypatch
):
    db_path, collection = isolated_index
    file_path = tmp_path / "embedding_failure.txt"
    file_path.write_text("retry_me", encoding="utf-8")

    def fail_embedding(texts):
        raise RuntimeError("intentional embedding failure")

    monkeypatch.setattr(backfill_index, "embed_texts", fail_embedding)
    with pytest.raises(RuntimeError, match="intentional embedding failure"):
        backfill_index.index_one_file(file_path)

    metadata = metadata_store.get_file_metadata(file_path, db_path=db_path)
    assert metadata["file_mtime_ns"] is None
    assert metadata["fail_count"] == 1
    assert metadata["failure_mtime_ns"] == file_path.stat().st_mtime_ns
    assert metadata_store.get_chunks(file_path, db_path=db_path) == []
    assert not any(
        item["metadata"].get("path") == str(file_path)
        for item in collection.items.values()
    )


def test_next_scan_retries_failed_file_and_indexes_successfully(
    tmp_path, isolated_index, monkeypatch
):
    db_path, collection = isolated_index
    file_path = tmp_path / "retry_after_failure.txt"
    file_path.write_text("retry_success", encoding="utf-8")

    monkeypatch.setattr(
        backfill_index,
        "embed_texts",
        lambda texts: (_ for _ in ()).throw(RuntimeError("temporary failure")),
    )
    with pytest.raises(RuntimeError, match="temporary failure"):
        backfill_index.index_one_file(file_path)

    monkeypatch.setattr(
        backfill_index,
        "embed_texts",
        lambda texts: [[float(len(text))] for text in texts],
    )
    status = backfill_index.index_one_file(file_path)

    metadata = metadata_store.get_file_metadata(file_path, db_path=db_path)
    assert status.startswith("[INDEXED - text]")
    assert metadata["file_mtime_ns"] == file_path.stat().st_mtime_ns
    assert metadata["fail_count"] == 0
    assert any(
        item["metadata"].get("chunk_text") == "retry_success"
        for item in collection.items.values()
    )


def test_file_is_skipped_after_three_failures_until_its_mtime_changes(
    tmp_path, isolated_index, monkeypatch
):
    _, collection = isolated_index
    file_path = tmp_path / "repeated_failure.txt"
    file_path.write_text("always_fails", encoding="utf-8")
    attempts = {"count": 0}

    def fail_embedding(texts):
        attempts["count"] += 1
        raise RuntimeError("repeated failure")

    monkeypatch.setattr(backfill_index, "embed_texts", fail_embedding)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="repeated failure"):
            backfill_index.index_one_file(file_path)

    skipped_status = backfill_index.index_one_file(file_path)

    assert skipped_status.startswith("[SKIPPED - failed 3 times]")
    assert attempts["count"] == 3
    assert collection.items == {}

    changed_mtime_ns = file_path.stat().st_mtime_ns + 2_000_000_000
    os.utime(file_path, ns=(changed_mtime_ns, changed_mtime_ns))
    with pytest.raises(RuntimeError, match="repeated failure"):
        backfill_index.index_one_file(file_path)

    metadata = metadata_store.get_file_metadata(file_path, db_path=isolated_index[0])
    assert attempts["count"] == 4
    assert metadata["fail_count"] == 1
    assert metadata["failure_mtime_ns"] == changed_mtime_ns


def test_backfill_continues_after_one_file_fails(
    tmp_path, isolated_index, monkeypatch
):
    db_path, collection = isolated_index
    bad_file = tmp_path / "bad.txt"
    good_file = tmp_path / "good.txt"
    bad_file.write_text("bad content", encoding="utf-8")
    good_file.write_text("good content", encoding="utf-8")
    errors = []

    def fail_bad_file(texts):
        if any("bad content" in text for text in texts):
            raise RuntimeError("bad file embedding failed")
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(backfill_index, "embed_texts", fail_bad_file)
    monkeypatch.setattr(
        backfill_index,
        "load_indexed_folders",
        lambda settings_path: [str(tmp_path)],
    )
    monkeypatch.setattr(backfill_index, "count_items", lambda: 1)

    backfill_index.backfill(
        settings_path=tmp_path / "unused-settings.json",
        error_callback=errors.append,
    )

    assert len(errors) == 1
    assert metadata_store.get_file_metadata(bad_file, db_path=db_path)["fail_count"] == 1
    assert metadata_store.get_file_metadata(good_file, db_path=db_path)["file_mtime_ns"] == (
        good_file.stat().st_mtime_ns
    )
    assert any(
        item["metadata"].get("path") == str(good_file)
        for item in collection.items.values()
    )
