"""
agent/harness.py

The agent's core think-act-observe loop. Per skills/agent-harness-loop/,
this file owns four responsibilities: prompt assembly, tool execution,
output parsing, and loop control with hard stops. It does NOT implement
retrieval/reasoning/explanation itself -- those are injected as functions
(retriever, corrective_check, reasoner, explainer) so this file stays
testable in isolation, per Task 2.5's approach.

Real implementations get wired in as they're built:
  retriever        -> Task 2.6 (agent/planner.py) + storage/vector_db
  corrective_check -> Task 2.7 (agent/corrective.py)
  reasoner         -> Task 2.8 (agent/reasoner.py)
  explainer        -> Task 2.9 (agent/explainer.py)
"""

from enum import Enum
import time
import json
from app_paths import get_base_dir


class AgentAction(Enum):
    REFORMULATE = "reformulate"
    ANSWER = "answer"
    NO_MATCH = "no_match"



def load_limits(config_path=None):
    if config_path is None:
        config_path = get_base_dir() / "config" / "harness_limits.json"
    with open(config_path, encoding="utf-8-sig") as f:
        return json.load(f)


def run_query(query, retriever, corrective_check, reasoner, explainer,
              config_path=None):
    limits = load_limits(config_path)
    max_reformulations = limits["max_reformulations"]
    timeout_seconds = limits["timeout_seconds"]

    start = time.time()
    attempts = 0
    current_query = query

    while attempts <= max_reformulations:
        if time.time() - start > timeout_seconds:
            return {"action": AgentAction.NO_MATCH, "reason": "timeout"}

        candidates = retriever(current_query)
        check = corrective_check(candidates, current_query)

        if check.confident:
            best = reasoner(candidates, current_query)
            explanation = explainer(best, current_query)
            return {"action": AgentAction.ANSWER, "result": best, "explanation": explanation}
        else:
            current_query = check.reformulated_query
            attempts += 1

    return {"action": AgentAction.NO_MATCH, "reason": "no confident match after retries"}