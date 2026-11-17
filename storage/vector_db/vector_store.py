"""Persistent ChromaDB storage for file embeddings."""

from pathlib import Path

from app_paths import get_base_dir, normalize_path


_client = None
_collection = None
COLLECTION_NAME = "file_index"
DATA_DIR_NAME = "chroma_data_v2"


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb

        persist_path = get_base_dir() / "vector_db" / DATA_DIR_NAME
        persist_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(persist_path))
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def add_item(item_id, embedding, metadata=None):
    collection = _get_collection()
    normalized_metadata = dict(metadata or {})
    if "path" in normalized_metadata:
        stored_path = normalized_metadata["path"]
        normalized_metadata["path"] = normalize_path(stored_path)
        item_id = str(item_id)
        if item_id == str(stored_path):
            item_id = normalized_metadata["path"]
        elif item_id.startswith(f"{stored_path}::"):
            item_id = f"{normalized_metadata['path']}{item_id[len(str(stored_path)):]}"
    collection.upsert(
        ids=[str(item_id)],
        embeddings=[[float(value) for value in embedding]],
        metadatas=[normalized_metadata],
    )


def query(query_embedding, n_results=5):
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return {"ids": [[]], "metadatas": [[]], "distances": [[]]}

    return collection.query(
        query_embeddings=[[float(value) for value in query_embedding]],
        n_results=min(max(1, int(n_results)), count),
        include=["metadatas", "distances"],
    )


def count_items():
    return _get_collection().count()


def list_indexed_paths():
    """Return normalized source paths represented in the vector collection."""
    stored = _get_collection().get(include=["metadatas"])
    return {
        normalize_path(metadata["path"])
        for metadata in stored.get("metadatas", [])
        if metadata and metadata.get("path")
    }


def delete_items_for_path(path):
    delete_by_file(path)


def delete_by_file(file_path):
    """Delete every vector entry associated with a file path."""
    _get_collection().delete(where={"path": normalize_path(file_path)})


def delete_relative_path_entries():
    """Remove vector entries whose stored path predates absolute-path indexing."""
    collection = _get_collection()
    stored = collection.get(include=["metadatas"])
    relative_paths = {
        metadata.get("path")
        for metadata in stored.get("metadatas", [])
        if metadata and metadata.get("path")
        and not Path(metadata["path"]).is_absolute()
    }
    relative_ids = [
        item_id
        for item_id, metadata in zip(
            stored.get("ids", []), stored.get("metadatas", [])
        )
        if (
            metadata and metadata.get("path")
            and not Path(metadata["path"]).is_absolute()
        ) or not Path(item_id.split("::", 1)[0]).is_absolute()
    ]
    for path in relative_paths:
        collection.delete(where={"path": path})
    if relative_ids:
        collection.delete(ids=relative_ids)
    return sorted(relative_paths)