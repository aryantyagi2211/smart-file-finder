from types import SimpleNamespace

from agent.corrective import check_results, refine_query
from observability.tracer import record_attempt, start_trace


def _candidate(path, score, preview):
    return SimpleNamespace(
        path=path,
        score=score,
        metadata={"chunk_text": preview},
    )


def test_refine_query_removes_fillers_and_adds_candidate_keywords():
    candidates = [
        _candidate(
            r"C:\docs\adaptive_retrieval.pdf",
            0.72,
            "retrieval augmented generation complexity routing",
        ),
        _candidate(
            r"C:\docs\question_classifier.pdf",
            0.68,
            "classifier selects search depth",
        ),
    ]

    refined = refine_query(
        "Find the file which talks about adaptive retrieval",
        candidates,
    )

    assert refined != "Find the file which talks about adaptive retrieval"
    assert "find" not in refined.split()
    assert "file" not in refined.split()
    assert "which" not in refined.split()
    assert "talks" not in refined.split()
    assert "retrieval" in refined.split()
    assert any(term in refined.split() for term in ("augmented", "generation", "complexity"))
    assert len(refined.split()) <= 12


def test_refine_query_returns_none_when_no_new_distinctive_terms():
    candidate = _candidate(
        r"C:\docs\retrieval.pdf",
        0.6,
        "retrieval augmented",
    )

    assert refine_query("retrieval augmented", [candidate]) is None


def test_corrective_check_keeps_broaden_and_refine_strategies_separate(tmp_path):
    config_path = tmp_path / "limits.json"
    config_path.write_text(
        '{"confidence_threshold": 0.9}',
        encoding="utf-8",
    )
    candidates = [
        _candidate(r"C:\docs\retrieval.pdf", 0.6, "augmented generation"),
        _candidate(r"C:\docs\classifier.pdf", 0.58, "query routing"),
    ]

    refined = check_results(candidates, "adaptive retrieval", str(config_path))
    broadened = check_results(
        [_candidate(r"C:\docs\other.pdf", 0.3, "unrelated")],
        "find the adaptive retrieval file",
        str(config_path),
    )
    empty = check_results([], "find the adaptive retrieval file", str(config_path))

    assert refined.strategy == "refine"
    assert refined.reformulated_query != "adaptive retrieval"
    assert broadened.strategy == "broaden"
    assert empty.strategy == "broaden"


def test_trace_records_query_reformulation_and_strategy():
    trace = start_trace("original query")
    candidate = _candidate(r"C:\docs\candidate.pdf", 0.6, "preview")

    record_attempt(
        trace,
        "original query",
        [candidate],
        False,
        0.6,
        "original preview",
        "refine",
    )

    assert trace["query"] == "original query"
    assert trace["attempts"][0]["query_used"] == "original query"
    assert trace["attempts"][0]["reformulated_query"] == "original preview"
    assert trace["attempts"][0]["strategy"] == "refine"
