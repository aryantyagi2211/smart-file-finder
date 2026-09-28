"""
agent/reasoner.py

Picks the single best candidate from filtered/confident results. Real
reasoning would use an LLM (Qwen3-1.7B/4B via NPU, Phase 3 -- see
CLAUDE.md) to judge relevance beyond raw similarity score -- e.g.
disambiguating between two close-scoring candidates by actually reading
their content. That model doesn't exist in this project yet.

For now (Phase 2 placeholder, same "mock now, swap in Phase 3" pattern as
indexing/vision_pipeline.py and agent/corrective.py's refine_query()):
candidates arrive already sorted best-first by score (see
agent/planner.py), so the placeholder strategy is simply picking the
top-scored one.
"""


def pick_best(candidates, query):
    """
    Select the single best candidate. PLACEHOLDER: picks the top-scored
    candidate (candidates are pre-sorted by agent/planner.py). Real
    LLM-based reasoning replaces this in Phase 3.
    """
    if not candidates:
        return None
    return candidates[0]