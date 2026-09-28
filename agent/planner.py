"""
agent/planner.py

Decides what/how to retrieve for a given query. Embeds the query, searches
the vector store, and normalizes ChromaDB's raw output into a clean list
of Candidate objects with a .score field (higher = better), so
agent/corrective.py and agent/reasoner.py (Tasks 2.7-2.8) can work with a
simple, consistent shape rather than ChromaDB's raw nested-list format.

ChromaDB returns "distance" (0 = identical, higher = worse). We convert
that into "score" (1.0 = identical, decreasing toward 0 = worse) via
score = 1 / (1 + distance), so it lines up with confidence_threshold in
config/harness_limits.json, which expects higher-is-better similarity.
"""

from dataclasses import dataclass

from indexing.embedder import embed_text
from storage.vector_db.vector_store import query as vector_query


@dataclass
class Candidate:
    path: str
    score: float
    metadata: dict


def distance_to_score(distance):
    """Convert ChromaDB distance (lower=better) to a 0-1 score (higher=better)."""
    return 1 / (1 + distance)


def retrieve(query_text, n_results=5):
    """
    Embed query_text and search the vector store, returning a list of
    Candidate objects sorted best-first (highest score first).
    """
    embedding = embed_text(query_text)
    raw = vector_query(embedding, n_results=n_results)

    candidates = []

    ids = raw.get("ids", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for item_id, metadata, distance in zip(ids, metadatas, distances):
        candidates.append(Candidate(
            path=metadata.get("path", item_id),
            score=distance_to_score(distance),
            metadata=metadata
        ))

    # already sorted best-first by ChromaDB (ascending distance), but be explicit
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates