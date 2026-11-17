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

from app import query_handler
from app_paths import normalize_path
from indexing.backfill_index import backfill
from indexing.embedder import preload_model
from storage.vector_db.vector_store import count_items, list_indexed_paths

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"
REPORT_PATH = Path(__file__).resolve().parent / "eval_report.md"
THRESHOLDS = tuple(round(value / 100, 2) for value in range(40, 71, 5))


def _resolve_golden_path(file_name, golden_dir):
    return normalize_path(Path(golden_dir) / Path(file_name).name)


def _acceptable_paths(case, golden_dir):
    file_names = case.get("acceptable_files")
    if file_names is None:
        file_name = case.get("expected_file")
        file_names = [] if file_name is None else [file_name]
    return {
        _resolve_golden_path(file_name, golden_dir).casefold()
        for file_name in file_names
    }


def _run_threshold_sweep(results, thresholds=THRESHOLDS):
    positive_cases = [row for row in results if row["acceptable_paths"]]
    negative_cases = [row for row in results if not row["acceptable_paths"]]
    sweep = []
    for threshold in thresholds:
        predictions = [
            row["best_path"] if row["best_score"] >= threshold else None
            for row in results
        ]
        positive_hits = sum(
            prediction is not None
            and prediction.casefold() in row["acceptable_paths"]
            for row, prediction in zip(results, predictions)
            if row["acceptable_paths"]
        )
        false_positives = sum(
            prediction is not None
            for row, prediction in zip(results, predictions)
            if not row["acceptable_paths"]
        )
        top1_correct = sum(
            (
                prediction is not None
                and prediction.casefold() in row["acceptable_paths"]
            ) if row["acceptable_paths"] else prediction is None
            for row, prediction in zip(results, predictions)
        )
        top3_correct = sum(
            (
                row["best_score"] >= threshold
                and bool(row["acceptable_paths"] & {
                    path.casefold() for path in row["top3_paths"]
                })
            ) if row["acceptable_paths"] else prediction is None
            for row, prediction in zip(results, predictions)
        )
        sweep.append({
            "threshold": threshold,
            "positive_hit_rate": positive_hits / len(positive_cases) if positive_cases else 0,
            "false_positive_rate": false_positives / len(negative_cases) if negative_cases else 0,
            "top1_accuracy": top1_correct / len(results) if results else 0,
            "top3_accuracy": top3_correct / len(results) if results else 0,
        })
    return sweep


def run_eval(golden_set_path=GOLDEN_SET_PATH):
    golden_dir = Path(normalize_path("test_data"))
    with open(golden_set_path, encoding="utf-8-sig") as f:
        cases = json.load(f)

    backfill(indexed_folders=[str(golden_dir)])
    indexed_paths = {
        normalize_path(path).casefold()
        for path in list_indexed_paths()
    }
    expected_paths = set().union(*(
        _acceptable_paths(case, golden_dir) for case in cases
    ))
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
        acceptable_paths = _acceptable_paths(case, golden_dir)
        attempts = []
        original_retrieve = query_handler.retrieve

        def capture_retrieve(reformulated_query):
            candidates = original_retrieve(reformulated_query)
            attempts.append(candidates)
            return candidates

        query_handler.retrieve = capture_retrieve
        try:
            outcome = query_handler.handle_query(query)
        finally:
            query_handler.retrieve = original_retrieve

        first_candidates = attempts[0] if attempts else []
        top3_paths = [
            normalize_path(candidate.path)
            for candidate in first_candidates[:3]
        ]
        all_candidates = [candidate for batch in attempts for candidate in batch]
        best_candidate = max(all_candidates, key=lambda candidate: candidate.score) if all_candidates else None
        best_path = normalize_path(best_candidate.path) if best_candidate else None
        best_score = best_candidate.score if best_candidate else 0.0
        actual_path = normalize_path(outcome["path"]) if outcome.get("found") else None
        hit = (
            actual_path.casefold() in acceptable_paths
            if acceptable_paths else not outcome.get("found")
        )
        expected = (
            " | ".join(sorted(Path(path).name for path in acceptable_paths))
            if acceptable_paths else "NO_MATCH"
        )
        actual = actual_path or f"NO_MATCH ({outcome.get('reason')})"
        in_top3 = (
            bool(acceptable_paths & {path.casefold() for path in top3_paths})
            if acceptable_paths else None
        )

        results.append({
            "query": query,
            "expected": expected,
            "actual": actual,
            "hit": hit,
            "score": best_score,
            "acceptable_paths": acceptable_paths,
            "best_path": best_path,
            "best_score": best_score,
            "top3_paths": top3_paths,
            "in_top3": in_top3,
        })

        status = "PASS" if hit else "FAIL"
        print(f"[{status}] {query} | {expected} | {actual} | {best_score:.3f} | {in_top3 if in_top3 is not None else 'N/A'}")

    hit_count = sum(r["hit"] for r in results)
    hit_rate = hit_count / len(results)
    print(f"Hit rate: {hit_rate:.0%} ({hit_count}/{len(results)})")
    sweep = _run_threshold_sweep(results)
    _print_threshold_sweep(sweep)
    _append_report(results, hit_rate, sweep)

    return results, hit_rate


def _print_threshold_sweep(sweep):
    print("\nThreshold sweep (top candidate per query; no config changes):")
    print("threshold | match hit rate | no-match false-positive rate | top-1 accuracy | top-3 accuracy")
    for row in sweep:
        print(
            f"{row['threshold']:.2f}      | "
            f"{row['positive_hit_rate']:.0%}            | "
            f"{row['false_positive_rate']:.0%}                          | "
            f"{row['top1_accuracy']:.0%}             | "
            f"{row['top3_accuracy']:.0%}"
        )


def _append_report(results, hit_rate, sweep):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(REPORT_PATH, "a", encoding="utf-8") as report:
        report.write(f"\n## Evaluation — {timestamp}\n\n")
        report.write(f"Hit rate: **{hit_rate:.0%} ({sum(row['hit'] for row in results)}/{len(results)})**\n\n")
        report.write("| Result | Query | Expected | Returned | Best score | Expected in top 3 |\n")
        report.write("|---|---|---|---|---:|:---:|\n")
        for row in results:
            report.write(
                f"| {'PASS' if row['hit'] else 'FAIL'} | {row['query']} "
                f"| {row['expected']} | {row['actual']} | {row['score']:.3f} "
                f"| {row['in_top3'] if row['in_top3'] is not None else 'N/A'} |\n"
            )
        report.write("\nThreshold sweep (top candidate score cutoff):\n\n")
        report.write("| Threshold | Match hit rate | No-match false-positive rate | Top-1 accuracy | Top-3 accuracy |\n")
        report.write("|---:|---:|---:|---:|---:|\n")
        for row in sweep:
            report.write(
                f"| {row['threshold']:.2f} | {row['positive_hit_rate']:.0%} "
                f"| {row['false_positive_rate']:.0%} | {row['top1_accuracy']:.0%} "
                f"| {row['top3_accuracy']:.0%} |\n"
            )


if __name__ == "__main__":
    run_eval()