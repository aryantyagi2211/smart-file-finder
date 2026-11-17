"""
agent/corrective.py

Confidence checking and fast, rule-based query reformulation.
confidence_threshold is always read from config/harness_limits.json --
never hardcoded here, to stay in sync with agent/harness.py's limits.

Two distinct reformulation strategies (per the skill file, not conflated):
  - broaden_query(): zero/few candidates -> query too narrow -> strip
    specific terms, keep general intent. Fully implemented (pure text
    manipulation, no model needed).
  - refine_query(): candidates exist but confidence is low -> query too
    vague/mismatched -> retain content terms and add distinctive candidate
    terms with simple local rules. No model calls are made.
"""

from dataclasses import dataclass
import json
import re

# Words too generic to carry search meaning on their own -- stripped
# during broadening so the reformulated query keeps only distinctive terms.
STOPWORDS = {
    "the", "a", "an", "that", "this", "with", "and", "or", "of", "in",
    "on", "for", "to", "is", "are", "was", "were", "find", "show", "me",
    "my", "file", "files", "document", "documents", "photo", "picture",
    "image", "which", "about", "talk", "talks", "tell", "give", "get",
    "please", "search", "look", "looking", "want", "need", "where", "what",
    "who", "when", "how", "can", "could", "would", "do", "does", "did",
    "from", "as", "by", "at", "into", "any", "some", "all", "thing",
    "things", "stuff", "something", "anything"
}
_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*", re.IGNORECASE)


@dataclass
class CorrectiveResult:
    confident: bool
    best_score: float
    reformulated_query: str = None
    strategy: str = None


from app_paths import get_config_path

def load_threshold(config_path=None):
    if config_path is None:
        config_path = get_config_path("harness_limits.json")
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
    """Keep query content words and add distinctive terms from near misses."""
    query_words = [
        word.lower()
        for word in _WORD_PATTERN.findall(query)
        if word.lower() not in STOPWORDS
    ]
    query_terms = set(query_words)
    refined_words = list(dict.fromkeys(query_words))
    added_terms = set()

    top_candidates = sorted(
        candidates, key=lambda candidate: candidate.score, reverse=True
    )[:3]
    for candidate in top_candidates:
        filename = candidate.path.replace("\\", "/").rsplit("/", 1)[-1]
        filename = re.sub(r"\.[^.]+$", "", filename)
        preview = (
            candidate.metadata.get("chunk_text")
            or candidate.metadata.get("description")
            or ""
        )
        candidate_terms = 0
        for source in (filename, preview):
            for word in _WORD_PATTERN.findall(source):
                term = word.lower()
                if term in STOPWORDS or term in query_terms or term in added_terms:
                    continue
                if len(term) < 3:
                    continue
                added_terms.add(term)
                refined_words.append(term)
                candidate_terms += 1
                if candidate_terms >= 2 or len(added_terms) >= 4:
                    break
            if candidate_terms >= 2 or len(added_terms) >= 4:
                break
        if len(added_terms) >= 4:
            break

    refined_query = " ".join(refined_words[:12])
    if not refined_query or _normalize_query(refined_query) == _normalize_query(query):
        return None
    return refined_query


def _normalize_query(query):
    return " ".join(_WORD_PATTERN.findall(query.lower()))


def check_results(candidates, original_query, config_path=None):
    threshold = load_threshold(config_path)

    if not candidates:
        return CorrectiveResult(
            confident=False,
            best_score=0.0,
            reformulated_query=broaden_query(original_query),
            strategy="broaden"
        )

    best = max(candidates, key=lambda c: c.score)

    if best.score >= threshold:
        return CorrectiveResult(confident=True, best_score=best.score)

    # Distinguish "query too narrow/off-target" (broaden) from "close but
    # not confident enough" (refine) using the best score itself, since
    # ChromaDB always returns n_results candidates regardless of quality
    # -- candidate count alone can't signal "no good matches".
    BROADEN_SCORE_FLOOR = 0.45
    if len(candidates) <= 1 or best.score < BROADEN_SCORE_FLOOR:
        reformulated = broaden_query(original_query)
        strategy = "broaden"
    else:
        reformulated = refine_query(original_query, candidates)
        strategy = "refine"

    return CorrectiveResult(
        confident=False,
        best_score=best.score,
        reformulated_query=reformulated,
        strategy=strategy
    )