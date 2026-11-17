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
from datetime import datetime
from pathlib import Path

from app.query_handler import handle_query
from app_paths import normalize_path
from indexing.backfill_index import backfill
from indexing.embedder import preload_model
from storage.vector_db.vector_store import count_items, list_indexed_paths

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"
REPORT_PATH = Path(__file__).resolve().parent / "eval_report.md"


def run_eval(golden_set_path=GOLDEN_SET_PATH):
    golden_dir = Path(normalize_path("test_data"))
    with open(golden_set_path, encoding="utf-8-sig") as f:
        cases = json.load(f)

    backfill(indexed_folders=[str(golden_dir)])
    indexed_paths = list_indexed_paths()
    expected_paths = {
        normalize_path(case["expected_file"])
        for case in cases
        if case["expected_file"] is not None
    }
    found_paths = expected_paths & indexed_paths
    print(f"Vector entries: {count_items()}")
    print(f"Golden expected files found in index: {len(found_paths)}/{len(expected_paths)}")
    for missing_path in sorted(expected_paths - indexed_paths):
        print(f"WARNING: Golden expected file is not indexed: {missing_path}")
    print()

    preload_model()

    results = []

    for case in cases:
        query = case["query"]
        expected = case["expected_file"]
        expected_path = normalize_path(expected) if expected is not None else None
        outcome = handle_query(query)
        actual_path = normalize_path(outcome["path"]) if outcome.get("found") else None

        if expected is None:
            hit = not outcome["found"]
        else:
            hit = outcome["found"] and actual_path.casefold() == expected_path.casefold()
        actual = actual_path if outcome["found"] else f"NO_MATCH ({outcome.get('reason')})"

        results.append({
            "query": query,
            "expected": expected_path or "NO_MATCH (expected)",
            "actual": actual,
            "hit": hit,
            "score": outcome.get("score", outcome.get("best_score", 0.0))
        })

        status = "PASS" if hit else "FAIL"
        print(f"[{status}] \"{query}\"")
        print(f"        expected: {expected_path or 'NO_MATCH'}")
        print(f"        actual:   {actual}")
        print(f"        best score: {results[-1]['score']:.3f}")
        print()

    hit_count = sum(r["hit"] for r in results)
    hit_rate = hit_count / len(results)
    print(f"Hit rate: {hit_rate:.0%} ({hit_count}/{len(results)})")
    _append_report(results, hit_rate)

    return results, hit_rate


def _append_report(results, hit_rate):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(REPORT_PATH, "a", encoding="utf-8") as report:
        report.write(f"\n## Evaluation — {timestamp}\n\n")
        report.write(f"Hit rate: **{hit_rate:.0%} ({sum(row['hit'] for row in results)}/{len(results)})**\n\n")
        report.write("| Result | Query | Expected | Actual | Best score |\n")
        report.write("|---|---|---|---|---:|\n")
        for row in results:
            report.write(
                f"| {'PASS' if row['hit'] else 'FAIL'} | {row['query']} "
                f"| {row['expected']} | {row['actual']} | {row['score']:.3f} |\n"
            )


if __name__ == "__main__":
    run_eval()