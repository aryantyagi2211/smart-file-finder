"""
agent/corrective.py

Confidence check + reformulation logic, per skills/corrective-rag-retry/.
confidence_threshold is always read from config/harness_limits.json --
never hardcoded here, to stay in sync with agent/harness.py's limits.

Two distinct reformulation strategies (per the skill file, not conflated):
  - broaden_query(): zero/few candidates -> query too narrow -> strip
    specific terms, keep general intent. Fully implemented (pure text
    manipulation, no model needed).
  - refine_query(): candidates exist but confidence is low -> query too
    vague/mismatched -> ideally uses a reasoning model to suggest better
    terms based on the gap between query and near-misses. NO real
    reasoning model exists yet (that's Phase 3, Qwen3-1.7B/4B on NPU --
    see CLAUDE.md). For now this is a clearly-marked PLACEHOLDER using a
    naive heuristic, matching the same "mock now, swap in Phase 3" pattern
    CLAUDE.md already uses for vision_pipeline.py.
"""

from dataclasses import dataclass
import json

# Words too generic to carry search meaning on their own -- stripped
# during broadening so the reformulated query keeps only distinctive terms.
STOPWORDS = {
    "the", "a", "an", "that", "this", "with", "and", "or", "of", "in",
    "on", "for", "to", "is", "are", "was", "were", "find", "show", "me",
    "my", "file", "document", "photo", "picture", "image"
}


@dataclass
class CorrectiveResult:
    confident: bool
    best_score: float
    reformulated_query: str = None


from app_paths import get_base_dir

def load_threshold(config_path=None):
    if config_path is None:
        config_path = get_base_dir() / "config" / "harness_limits.json"
    with open(config_path, encoding="utf-8-sig") as f:
        return json.load(f)["confidence_threshold"]

def broaden_query(query):
    """
    Strip specific/stop words, keep the general intent. Used when there
    are zero or very few candidates -- the query was likely too narrow.
    """
    words = [w for w in query.split() if w.lower() not in STOPWORDS]
    if not words:
        return query  # nothing left to strip further, return as-is
    # Keep roughly the first half of remaining words -- a simple way to
    # generalize without needing a model to judge what's "core" intent.
    keep_count = max(1, len(words) // 2)
    return " ".join(words[:keep_count])


def refine_query(query, candidates):
    """
    PLACEHOLDER refinement strategy. Real refinement needs a reasoning
    model to look at near-miss candidates and suggest better terms (see
    module docstring) -- that model doesn't exist in this project yet.
    For now: re-run the same query unchanged. This is intentionally a
    no-op stand-in, not a real strategy, so it's obvious this needs
    revisiting once agent/reasoner.py + a real reasoning model exist.
    """
    return query


def check_results(candidates, original_query, config_path=None):
    threshold = load_threshold(config_path)

    if not candidates:
        return CorrectiveResult(
            confident=False,
            best_score=0.0,
            reformulated_query=broaden_query(original_query)
        )

    best = max(candidates, key=lambda c: c.score)

    if best.score >= threshold:
        return CorrectiveResult(confident=True, best_score=best.score)

    # Distinguish "query too narrow/off-target" (broaden) from "close but
    # not confident enough" (refine) using the best score itself, since
    # ChromaDB always returns n_results candidates regardless of quality
    # -- candidate count alone can't signal "no good matches".
    BROADEN_SCORE_FLOOR = 0.45
    if best.score < BROADEN_SCORE_FLOOR:
        reformulated = broaden_query(original_query)
    else:
        reformulated = refine_query(original_query, candidates)

    return CorrectiveResult(
        confident=False,
        best_score=best.score,
        reformulated_query=reformulated
    )