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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
import json
from app_paths import get_config_path
from observability.logger import log


_STEP_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="query-step")


class StepDeadlineExceeded(Exception):
    def __init__(self, step_name):
        super().__init__(step_name)
        self.step_name = step_name


class AgentAction(Enum):
    REFORMULATE = "reformulate"
    ANSWER = "answer"
    NO_MATCH = "no_match"



def load_limits(config_path=None):
    if config_path is None:
        config_path = get_config_path("harness_limits.json")
    with open(config_path, encoding="utf-8-sig") as f:
        return json.load(f)


def _run_step(step_name, function, args, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StepDeadlineExceeded(step_name)

    future = _STEP_EXECUTOR.submit(function, *args)
    try:
        return future.result(timeout=remaining)
    except FutureTimeoutError:
        if future.done():
            return future.result()
        future.cancel()
        raise StepDeadlineExceeded(step_name)


def _failure(reason, stage):
    if reason == "timeout":
        message = "Nothing found in time; try rephrasing your search."
    else:
        message = "Search failed unexpectedly; please try again."
    return {
        "action": AgentAction.NO_MATCH,
        "reason": reason,
        "stage": stage,
        "message": message,
    }


def run_query(query, retriever, corrective_check, reasoner, explainer,
              config_path=None):
    start = time.monotonic()
    try:
        limits = load_limits(config_path)
    except Exception as error:
        log(f"Query failed while loading limits: {error}")
        return _failure("error", "load_limits")

    max_reformulations = limits["max_reformulations"]
    timeout_seconds = limits["timeout_seconds"]

    deadline = start + timeout_seconds
    attempts = 0
    current_query = query

    while attempts <= max_reformulations:
        try:
            stage = "retriever"
            candidates = _run_step(
                "retriever", retriever, (current_query,), deadline
            )
            stage = "corrective_check"
            check = _run_step(
                "corrective_check",
                corrective_check,
                (candidates, current_query),
                deadline,
            )

            if check.confident:
                stage = "reasoner"
                best = _run_step(
                    "reasoner", reasoner, (candidates, current_query), deadline
                )
                stage = "explainer"
                explanation = _run_step(
                    "explainer", explainer, (best, current_query), deadline
                )
                return {
                    "action": AgentAction.ANSWER,
                    "result": best,
                    "explanation": explanation,
                }

            current_query = check.reformulated_query
            attempts += 1
        except StepDeadlineExceeded as error:
            return _failure("timeout", error.step_name)
        except Exception as error:
            log(f"Query failed during {stage}: {error}")
            return _failure("error", stage)

    return {"action": AgentAction.NO_MATCH, "reason": "no confident match after retries"}