"""
indexing/index_chunks.py

Takes text chunks already stored in metadata_store (via text_pipeline,
Task 1.7) and embeds + stores them in the vector store, making them
searchable by meaning. This is the Task 2.2 wiring step.
"""

from storage.metadata_store import get_chunks
from storage.vector_db.vector_store import add_item
from indexing.embedder import embed_texts


def index_file_chunks(file_path):
    """
    Embed all stored chunks for a file and add them to the vector store.
    Each chunk gets its own vector-store entry, ID'd as "path::chunk_index".
    """
    chunks = get_chunks(file_path)
    if not chunks:
        print(f"No chunks found for {file_path}")
        return 0

    vectors = embed_texts(chunks)

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        item_id = f"{file_path}::{i}"
        add_item(item_id, vector, metadata={
            "path": file_path,
            "chunk_index": i,
            "chunk_text": chunk[:200]  # short preview, not full text
        })

    print(f"Indexed {len(chunks)} chunks for {file_path}")
    return len(chunks)