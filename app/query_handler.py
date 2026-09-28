"""
app/query_handler.py

Wraps the full Phase 2 harness (agent/harness.py + planner/corrective/
reasoner/explainer) into a single function the UI layer can call. Also
handles tracing/logging, same pattern as eval/run_query_cli.py (Task 2's
checkpoint script) -- this is that same wiring, reused for the real app.
"""

from agent.harness import run_query, AgentAction
from agent.planner import retrieve
from agent.corrective import check_results
from agent.reasoner import pick_best
from agent.explainer import explain
from observability.logger import log
from observability.tracer import start_trace, record_attempt, finalize_trace, write_trace


def handle_query(query_text):
    """
    Run a query through the full harness. Returns a dict:
      {"found": True, "path": ..., "score": ..., "explanation": ...}
      or
      {"found": False, "reason": ...}
    """
    log(f"Query received: {query_text}")
    trace = start_trace(query_text)

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

    outcome = run_query(query_text, retriever, corrective_check, reasoner, explainer)

    if outcome["action"] == AgentAction.ANSWER:
        best = outcome["result"]
        finalize_trace(trace, action="answer", result_path=best.path, explanation=outcome["explanation"])
        write_trace(trace)
        log(f"Query resolved: {best.path}")
        return {
            "found": True,
            "path": best.path,
            "score": best.score,
            "explanation": outcome["explanation"]
        }
    else:
        finalize_trace(trace, action="no_match")
        write_trace(trace)
        log(f"Query returned NO_MATCH: {outcome['reason']}")
        return {"found": False, "reason": outcome["reason"]}