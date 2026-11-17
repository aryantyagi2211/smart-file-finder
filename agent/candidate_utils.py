"""Helpers for ranking and grouping retrieval candidates by source file."""

from app_paths import normalize_path


def group_candidates_by_file(candidates):
    """Keep the highest-scoring candidate for each resolved source path."""
    best_by_file = {}
    for candidate in candidates:
        path = getattr(candidate, "path", None)
        key = normalize_path(path).casefold() if path else repr(candidate)
        current = best_by_file.get(key)
        if current is None or _score(candidate) > _score(current):
            best_by_file[key] = candidate
    return sorted(best_by_file.values(), key=_score, reverse=True)


def _score(candidate):
    try:
        return float(candidate.score)
    except (AttributeError, TypeError, ValueError):
        return 0.0
