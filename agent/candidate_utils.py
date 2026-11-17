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


def select_options_by_score(options, min_option_score, option_score_window,
                            max_options=3):
    """Keep the best option and only close, sufficiently scored alternatives."""
    ranked = sorted(options, key=_score, reverse=True)
    if not ranked:
        return []

    best_score = _score(ranked[0])
    selected = [ranked[0]]
    selected.extend(
        option for option in ranked[1:]
        if _score(option) >= min_option_score
        and best_score - _score(option) <= option_score_window
    )
    return selected[:max_options]


def _score(candidate):
    try:
        return float(candidate.score)
    except (AttributeError, TypeError, ValueError):
        try:
            return float(candidate["score"])
        except (KeyError, TypeError, ValueError):
            return 0.0
