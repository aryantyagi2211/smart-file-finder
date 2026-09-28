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
from observability.logger import log
from observability.tracer import start_trace, record_attempt, finalize_trace, write_trace


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m eval.run_query_cli \"your query here\"")
        return

    query = " ".join(sys.argv[1:])
    log(f"Query received: {query}")
    trace = start_trace(query)

    def retriever(q):
        return retrieve(q)

    def corrective_check(candidates, q):
        result = check_results(candidates, q)
        record_attempt(trace, q, candidates, result.confident, result.best_score, result.reformulated_query)
        return result

    def reasoner(candidates, q):
        return pick_best(candidates, q)

    def explainer(best, q):
        return explain(best, q)

    outcome = run_query(query, retriever, corrective_check, reasoner, explainer)

    if outcome["action"] == AgentAction.ANSWER:
        print(f"\nBest match: {outcome['result'].path}")
        print(f"Explanation: {outcome['explanation']}")
        finalize_trace(trace, action="answer", result_path=outcome["result"].path, explanation=outcome["explanation"])
        log(f"Query resolved: {outcome['result'].path}")
    else:
        print(f"\nNo confident match found. Reason: {outcome['reason']}")
        finalize_trace(trace, action="no_match")
        log(f"Query returned NO_MATCH: {outcome['reason']}")

    write_trace(trace)


if __name__ == "__main__":
    main()