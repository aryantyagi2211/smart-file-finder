"""
tests/unit/test_harness.py

Tests agent/harness.py's loop control using stub retriever/corrective/
reasoner/explainer functions -- proves the loop mechanics (confident path,
reformulation path, max-attempts exhaustion, timeout) work correctly in
isolation, before the real modules exist (Tasks 2.6-2.9).
"""

import time
from dataclasses import dataclass

from agent import harness
from agent.harness import run_query, AgentAction


@dataclass
class FakeCorrectiveResult:
    confident: bool
    reformulated_query: str = None


def test_confident_on_first_try(tmp_path):
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=5)

    retriever = lambda q: ["candidate1"]
    corrective_check = lambda candidates, q: FakeCorrectiveResult(confident=True)
    reasoner = lambda candidates, q: "candidate1"
    explainer = lambda best, q: f"matched because of {q}"

    result = run_query("find my photo", retriever, corrective_check, reasoner, explainer,
                        config_path=config_path)

    assert result["action"] == AgentAction.ANSWER
    assert result["result"] == "candidate1"


def test_reformulates_then_succeeds(tmp_path):
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=1)

    call_count = {"n": 0}

    def corrective_check(candidates, q):
        call_count["n"] += 1
        if call_count["n"] < 2:
            return FakeCorrectiveResult(confident=False, reformulated_query="broader query")
        return FakeCorrectiveResult(confident=True)

    retriever = lambda q: ["candidate1"]
    reasoner = lambda candidates, q: "candidate1"
    explainer = lambda best, q: "explained"

    result = run_query("vague query", retriever, corrective_check, reasoner, explainer,
                        config_path=config_path)

    assert result["action"] == AgentAction.ANSWER
    assert call_count["n"] == 2  # confirms it actually reformulated once before succeeding


def test_exhausts_reformulations_returns_no_match(tmp_path):
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=5)

    retriever = lambda q: []
    corrective_check = lambda candidates, q: FakeCorrectiveResult(confident=False, reformulated_query="try again")
    reasoner = lambda candidates, q: None
    explainer = lambda best, q: None

    result = run_query("nonsense query", retriever, corrective_check, reasoner, explainer,
                        config_path=config_path)

    assert result["action"] == AgentAction.NO_MATCH
    assert "retries" in result["reason"]


def test_timeout_returns_no_match(tmp_path):
    config_path = _write_limits(tmp_path, max_reformulations=5, timeout=0.05)

    def slow_retriever(q):
        time.sleep(0.1)  # slower than the 0.05s budget
        return ["candidate1"]

    # never confident, so the loop keeps going until the timeout check catches it
    corrective_check = lambda candidates, q: FakeCorrectiveResult(confident=False, reformulated_query="try again")
    reasoner = lambda candidates, q: "candidate1"
    explainer = lambda best, q: "explained"

    result = run_query("any query", slow_retriever, corrective_check, reasoner, explainer,
                        config_path=config_path)

    assert result["action"] == AgentAction.NO_MATCH
    assert result["reason"] == "timeout"


def test_slow_retriever_obeys_hard_deadline(tmp_path):
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=0.1)

    def slow_retriever(query):
        time.sleep(0.3)
        return []

    start = time.monotonic()
    result = run_query(
        "slow retrieval",
        slow_retriever,
        lambda candidates, query: FakeCorrectiveResult(confident=True),
        lambda candidates, query: candidates[0],
        lambda best, query: "explanation",
        config_path=config_path,
    )
    elapsed = time.monotonic() - start

    assert result["action"] == AgentAction.NO_MATCH
    assert result["reason"] == "timeout"
    assert result["stage"] == "retriever"
    assert elapsed < 1.1


def test_slow_reasoner_obeys_hard_deadline(tmp_path):
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=0.1)

    def slow_reasoner(candidates, query):
        time.sleep(0.3)
        return candidates[0]

    start = time.monotonic()
    result = run_query(
        "slow reasoning",
        lambda query: ["candidate"],
        lambda candidates, query: FakeCorrectiveResult(confident=True),
        slow_reasoner,
        lambda best, query: "explanation",
        config_path=config_path,
    )
    elapsed = time.monotonic() - start

    assert result["action"] == AgentAction.NO_MATCH
    assert result["reason"] == "timeout"
    assert result["stage"] == "reasoner"
    assert elapsed < 1.1


def test_slow_explainer_obeys_hard_deadline(tmp_path):
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=0.1)

    def slow_explainer(best, query):
        time.sleep(0.3)
        return "explanation"

    start = time.monotonic()
    result = run_query(
        "slow explanation",
        lambda query: ["candidate"],
        lambda candidates, query: FakeCorrectiveResult(confident=True),
        lambda candidates, query: candidates[0],
        slow_explainer,
        config_path=config_path,
    )
    elapsed = time.monotonic() - start

    assert result["action"] == AgentAction.NO_MATCH
    assert result["reason"] == "timeout"
    assert result["stage"] == "explainer"
    assert elapsed < 1.1


def test_step_exception_returns_error_and_logs(tmp_path, monkeypatch):
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=1)
    logged = []
    monkeypatch.setattr(harness, "log", logged.append)

    def failing_retriever(query):
        raise RuntimeError("retrieval broke")

    result = run_query(
        "failing query",
        failing_retriever,
        lambda candidates, query: FakeCorrectiveResult(confident=True),
        lambda candidates, query: None,
        lambda best, query: None,
        config_path=config_path,
    )

    assert result["action"] == AgentAction.NO_MATCH
    assert result["reason"] == "error"
    assert result["stage"] == "retriever"
    assert "retrieval broke" in logged[0]


def test_reformulation_completes_within_query_budget(tmp_path):
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=0.5)
    calls = []

    def retriever(query):
        calls.append(query)
        time.sleep(0.02)
        return ["candidate"]

    def corrective_check(candidates, query):
        time.sleep(0.02)
        return FakeCorrectiveResult(
            confident=len(calls) == 2,
            reformulated_query="broader query",
        )

    result = run_query(
        "narrow query",
        retriever,
        corrective_check,
        lambda candidates, query: candidates[0],
        lambda best, query: "explained",
        config_path=config_path,
    )

    assert result["action"] == AgentAction.ANSWER
    assert calls == ["narrow query", "broader query"]


def _write_limits(tmp_path, max_reformulations, timeout):
    import json
    config_path = tmp_path / "harness_limits.json"
    config_path.write_text(json.dumps({
        "max_reformulations": max_reformulations,
        "timeout_seconds": timeout,
        "confidence_threshold": 0.75
    }))
    return str(config_path)