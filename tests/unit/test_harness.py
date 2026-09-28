"""
tests/unit/test_harness.py

Tests agent/harness.py's loop control using stub retriever/corrective/
reasoner/explainer functions -- proves the loop mechanics (confident path,
reformulation path, max-attempts exhaustion, timeout) work correctly in
isolation, before the real modules exist (Tasks 2.6-2.9).
"""

import time
from dataclasses import dataclass
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
    config_path = _write_limits(tmp_path, max_reformulations=2, timeout=5)

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


def _write_limits(tmp_path, max_reformulations, timeout):
    import json
    config_path = tmp_path / "harness_limits.json"
    config_path.write_text(json.dumps({
        "max_reformulations": max_reformulations,
        "timeout_seconds": timeout,
        "confidence_threshold": 0.75
    }))
    return str(config_path)