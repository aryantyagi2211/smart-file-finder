"""
eval/run_query_cli.py

Simple command-line script to run a query through the full agent harness
(agent/harness.py) using the real planner/corrective/reasoner/explainer
modules -- this is the Phase 2 checkpoint script per TASK.md.

Usage:
    python -m eval.run_query_cli "your query here"
"""

import sys

from agent.harness import run_query, AgentAction
from agent.planner import retrieve
from agent.corrective import check_results
from agent.reasoner import pick_best
from agent.explainer import explain
from indexing.embedder import preload_model
from observability.logger import log
from observability.tracer import (
    start_trace,
    record_attempt,
    record_decision,
    record_failure,
    finalize_trace,
    write_trace,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m eval.run_query_cli \"your query here\"")
        return

    query = " ".join(sys.argv[1:])
    preload_model()
    log(f"Query received: {query}")
    trace = start_trace(query)

    def retriever(q):
        return retrieve(q)

    def corrective_check(candidates, q):
        result = check_results(candidates, q)
        record_attempt(
            trace, q, candidates, result.confident, result.best_score,
            result.reformulated_query, result.strategy
        )
        return result

    def reasoner(candidates, q):
        return pick_best(candidates, q)

    def explainer(best, q):
        return explain(best, q)

    outcome = run_query(query, retriever, corrective_check, reasoner, explainer)

    if outcome["action"] == AgentAction.ANSWER:
        print("\nAction: ANSWER")
        print(f"Best match: {outcome['result'].path} ({outcome['result'].score:.3f})")
        print(f"Explanation: {outcome['explanation']}")
        record_decision(trace, "confident", outcome.get("candidates", [outcome["result"]]))
        finalize_trace(trace, action="answer", result_path=outcome["result"].path, explanation=outcome["explanation"])
        log(f"Query resolved: {outcome['result'].path}")
    elif outcome["action"] == AgentAction.OPTIONS:
        print("\nAction: OPTIONS")
        for index, option in enumerate(outcome["options"], start=1):
            print(f"{index}. {option['path']} ({option['score']:.3f})")
            print(f"   {option['explanation']}")
        record_decision(trace, "options", outcome.get("candidates", []))
        finalize_trace(trace, action="options", reason="uncertain match")
        log("Query returned OPTIONS")
    else:
        print(f"\nAction: NO_MATCH. Reason: {outcome['reason']}")
        for candidate in outcome.get("candidates", []):
            print(f"Candidate: {candidate.path} ({candidate.score:.3f})")
        record_decision(trace, "no_match", outcome.get("candidates", []))
        if outcome["reason"] in {"timeout", "error"}:
            record_failure(
                trace,
                query,
                outcome.get("stage", "query_step"),
                outcome["reason"],
            )
        finalize_trace(trace, action="no_match", reason=outcome["reason"])
        log(f"Query returned NO_MATCH: {outcome['reason']}")

    write_trace(trace)


if __name__ == "__main__":
    main()