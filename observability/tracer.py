"""
observability/tracer.py

Structured per-query trace, written as JSON Lines (one JSON object per
line) to observability/traces.jsonl. Captures the full decision path for
a single query: the query itself, retrieved candidates, the corrective
check outcome, any reformulation attempts, and the final result. This is
what Phase 3's golden-set eval (skills/golden-set-validation/) and future
debugging of bad matches will read from -- richer than the plain-English
activity log in logger.py.
"""

import json
from pathlib import Path
from datetime import datetime

from app_paths import get_base_dir

TRACE_PATH = get_base_dir() / "observability" / "traces.jsonl"

def start_trace(query):
    """
    Create a new trace dict for a single query. Call this once per query,
    then pass the returned dict into record_attempt() for each retrieval
    attempt, and finally write_trace() once the query is fully resolved.
    """
    return {
        "query": query,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "attempts": [],
        "final_action": None,
        "final_result": None
    }


def record_attempt(trace, query_used, candidates, confident, best_score,
                   reformulated_query=None, strategy=None):
    """Append one retrieval attempt's details to the trace."""
    trace["attempts"].append({
        "query_used": query_used,
        "num_candidates": len(candidates),
        "top_candidate_paths": [c.path for c in candidates[:3]],
        "candidate_scores": [
            {"path": c.path, "score": float(c.score)}
            for c in candidates[:3]
        ],
        "confident": confident,
        "best_score": best_score,
        "reformulated_query": reformulated_query,
        "strategy": strategy,
    })


def record_decision(trace, decision, candidates):
    """Record the final confidence decision and its ranked file scores."""
    trace["decision"] = decision
    trace["decision_files"] = [
        {"path": candidate.path, "score": float(candidate.score)}
        for candidate in candidates
    ]


def record_failure(trace, query_used, stage, reason):
    """Record a timed-out or failed harness step as a trace attempt."""
    trace["attempts"].append({
        "query_used": query_used,
        "stage": stage,
        "outcome": reason,
        "confident": False,
    })


def finalize_trace(trace, action, result_path=None, explanation=None, reason=None):
    """Mark the trace as complete with its final outcome."""
    trace["final_action"] = action
    trace["final_result"] = result_path
    trace["explanation"] = explanation
    trace["reason"] = reason
    trace["ended_at"] = datetime.now().isoformat(timespec="seconds")


def write_trace(trace, trace_path=TRACE_PATH):
    """Append the completed trace as one JSON line."""
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")