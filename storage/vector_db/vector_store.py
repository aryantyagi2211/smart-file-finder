"""Persistent ChromaDB storage for file embeddings."""

from app_paths import get_base_dir


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
    collection.upsert(
        ids=[str(item_id)],
        embeddings=[[float(value) for value in embedding]],
        metadatas=[metadata or {}],
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


def delete_items_for_path(path):
    _get_collection().delete(where={"path": str(path)})