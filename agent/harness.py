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
from dataclasses import dataclass
import re
import time
import json
from agent.candidate_utils import group_candidates_by_file
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
    OPTIONS = "options"
    NO_MATCH = "no_match"


@dataclass
class CandidateDecision:
    decision: str
    candidates: list


_DEFAULT_LIMITS = {
    "max_reformulations": 2,
    "timeout_seconds": 5,
    "confidence_threshold": 0.55,
    "ambiguity_margin": 0.05,
    "options_floor": 0.45,
    "max_options": 3,
}


def load_limits(config_path=None):
    if config_path is None:
        config_path = get_config_path("harness_limits.json")
    with open(config_path, encoding="utf-8-sig") as f:
        return {**_DEFAULT_LIMITS, **json.load(f)}


def decide_candidates(candidates, confidence_threshold=0.55,
                      ambiguity_margin=0.05, options_floor=0.45,
                      max_options=3, confidence_hint=False):
    """Classify grouped file candidates as confident, options, or no-match."""
    ranked = group_candidates_by_file(candidates)
    if not ranked:
        return CandidateDecision("no_match", [])

    best = ranked[0]
    best_score = getattr(best, "score", None)
    if best_score is None:
        return CandidateDecision(
            "confident" if confidence_hint else "no_match",
            ranked[:1] if confidence_hint else [],
        )

    def score(candidate):
        try:
            return float(candidate.score)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    gap = (
        score(best) - score(ranked[1])
        if len(ranked) > 1 else float("inf")
    )
    if score(best) >= confidence_threshold and gap >= ambiguity_margin:
        return CandidateDecision("confident", ranked[:1])
    if score(best) >= options_floor:
        return CandidateDecision("options", ranked[:max_options])
    return CandidateDecision("no_match", ranked[:max_options])


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
        "decision": "no_match",
    }


def run_query(query, retriever, corrective_check, reasoner, explainer,
              config_path=None):
    start = time.monotonic()
    try:
        limits = load_limits(config_path)
    except Exception as error:
        log(f"Query failed while loading limits: {error}")
        return _failure("error", "load_limits")

    max_reformulations = int(limits["max_reformulations"])
    timeout_seconds = float(limits["timeout_seconds"])

    deadline = start + timeout_seconds
    attempts = 0
    current_query = query
    tried_queries = {query}
    best_candidates = []

    def normalized_query(value):
        return (
            " ".join(re.sub(r"\W+", " ", value.casefold()).split())
            if isinstance(value, str) else None
        )

    while attempts <= max_reformulations:
        try:
            stage = "retriever"
            candidates = _run_step(
                "retriever", retriever, (current_query,), deadline
            )
            candidates = group_candidates_by_file(candidates or [])
            stage = "corrective_check"
            check = _run_step(
                "corrective_check",
                corrective_check,
                (candidates, current_query),
                deadline,
            )

            best_candidates = group_candidates_by_file(best_candidates + candidates)
            decision = decide_candidates(
                candidates,
                confidence_threshold=float(limits["confidence_threshold"]),
                ambiguity_margin=float(limits["ambiguity_margin"]),
                options_floor=float(limits["options_floor"]),
                max_options=int(limits["max_options"]),
                confidence_hint=bool(getattr(check, "confident", False)),
            )

            if decision.decision == "confident":
                stage = "reasoner"
                best = _run_step(
                    "reasoner", reasoner, (decision.candidates, current_query), deadline
                )
                stage = "explainer"
                explanation = _run_step(
                    "explainer", explainer, (best, current_query), deadline
                )
                return {
                    "action": AgentAction.ANSWER,
                    "decision": "confident",
                    "result": best,
                    "explanation": explanation,
                    "candidates": decision.candidates,
                }

            if decision.decision == "options":
                options = []
                for candidate in decision.candidates:
                    stage = "explainer"
                    explanation = _run_step(
                        "explainer", explainer, (candidate, current_query), deadline
                    )
                    options.append({
                        "candidate": candidate,
                        "path": candidate.path,
                        "score": float(candidate.score),
                        "explanation": explanation,
                    })
                return {
                    "action": AgentAction.OPTIONS,
                    "decision": "options",
                    "options": options,
                    "candidates": decision.candidates,
                }

            next_query = check.reformulated_query
            tried = {normalized_query(value) for value in tried_queries}
            normalized_next_query = normalized_query(next_query)
            if not normalized_next_query or normalized_next_query in tried:
                strategy = getattr(check, "strategy", None) or "reformulate"
                return _finish_no_match_or_options(
                    best_candidates, limits,
                    f"no new query from {strategy}", explainer,
                    current_query, deadline,
                )
            current_query = next_query
            tried_queries.add(current_query)
            attempts += 1
        except StepDeadlineExceeded as error:
            return _failure("timeout", error.step_name)
        except Exception as error:
            log(f"Query failed during {stage}: {error}")
            return _failure("error", stage)

    return _finish_no_match_or_options(
        best_candidates, limits, "no confident match after retries",
        explainer, current_query, deadline,
    )


def _finish_no_match_or_options(candidates, limits, reason, explainer,
                                query, deadline):
    decision = decide_candidates(
        candidates,
        confidence_threshold=float(limits["confidence_threshold"]),
        ambiguity_margin=float(limits["ambiguity_margin"]),
        options_floor=float(limits["options_floor"]),
        max_options=int(limits["max_options"]),
    )
    if decision.decision == "options":
        options = []
        try:
            for candidate in decision.candidates:
                explanation = _run_step(
                    "explainer", explainer, (candidate, query), deadline
                )
                options.append({
                    "candidate": candidate,
                    "path": candidate.path,
                    "score": float(candidate.score),
                    "explanation": explanation,
                })
        except StepDeadlineExceeded as error:
            return _failure("timeout", error.step_name)
        except Exception as error:
            log(f"Query failed during explainer: {error}")
            return _failure("error", "explainer")
        return {
            "action": AgentAction.OPTIONS,
            "decision": "options",
            "options": options,
            "candidates": decision.candidates,
        }
    return {
        "action": AgentAction.NO_MATCH,
        "decision": "no_match",
        "reason": reason,
        "candidates": decision.candidates,
    }