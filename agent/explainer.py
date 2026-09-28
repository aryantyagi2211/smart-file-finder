"""
agent/explainer.py

Generates the "why this matched" text (the XAI RAG layer, per CLAUDE.md's
locked-in architecture). Real explanation generation would ideally use an
LLM (Qwen3-1.7B/4B, Phase 3) to articulate *why* a candidate is relevant in
natural language -- that model doesn't exist in this project yet.

For now (Phase 2 placeholder): a template-based explanation using the
candidate's real score and a real content preview (chunk_text for text
files, description for images -- both already stored in metadata by
agent/planner.py / indexing pipelines). This is more useful than a fully
generic stub for the Phase 2 checkpoint, but it's still not genuine
reasoning about *why* the content matches semantically -- just a
templated confidence + preview. Swap for real LLM-generated explanations
in Phase 3.
"""


def explain(candidate, query):
    """
    Generate a human-readable explanation for why `candidate` matched
    `query`. PLACEHOLDER: templates the score + a content preview from
    the candidate's stored metadata.
    """
    if candidate is None:
        return "No confident match was found for this query."

    confidence_pct = round(candidate.score * 100)

    preview = candidate.metadata.get("chunk_text") or candidate.metadata.get("description") or ""
    preview = preview.strip().replace("\n", " ")
    if len(preview) > 120:
        preview = preview[:120] + "..."

    return (
        f"Matched with {confidence_pct}% confidence — this file's content "
        f"is semantically similar to your query. Preview: \"{preview}\""
    )