"""
eval/run_eval.py

Loads eval/golden_set.json, runs each query through the real harness (via
app/query_handler.py), and reports a hit rate -- per
skills/golden-set-validation/. Also used to empirically tune
confidence_threshold in config/harness_limits.json rather than guessing.

Usage:
    python -m eval.run_eval
"""

import json
from pathlib import Path

from app.query_handler import handle_query

GOLDEN_SET_PATH = Path("eval/golden_set.json")


def run_eval(golden_set_path=GOLDEN_SET_PATH):
    with open(golden_set_path, encoding="utf-8-sig") as f:
        cases = json.load(f)

    results = []

    for case in cases:
        query = case["query"]
        expected = case["expected_file"]
        outcome = handle_query(query)

        if expected is None:
            # This case expects NO_MATCH -- a "hit" here means the system
            # correctly found nothing, not that it found the right file.
            hit = not outcome["found"]
            actual = outcome.get("path") if outcome["found"] else "NO_MATCH"
        else:
            hit = outcome["found"] and outcome["path"] == expected
            actual = outcome.get("path") if outcome["found"] else "NO_MATCH"

        results.append({
            "query": query,
            "expected": expected or "NO_MATCH (expected)",
            "actual": actual,
            "hit": hit,
            "score": outcome.get("score")
        })

        status = "PASS" if hit else "FAIL"
        print(f"[{status}] \"{query}\"")
        print(f"        expected: {expected or 'NO_MATCH'}")
        print(f"        actual:   {actual}")
        if outcome.get("score") is not None:
            print(f"        score:    {outcome['score']:.3f}")
        print()

    hit_count = sum(r["hit"] for r in results)
    hit_rate = hit_count / len(results)
    print(f"Hit rate: {hit_rate:.0%} ({hit_count}/{len(results)})")

    return results, hit_rate


if __name__ == "__main__":
    run_eval()