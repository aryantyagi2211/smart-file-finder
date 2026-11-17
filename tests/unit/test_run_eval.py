from eval import run_eval


def test_acceptable_files_are_normalized_relative_to_golden_folder(tmp_path):
    case = {
        "query": "ambiguous query",
        "acceptable_files": ["first.txt", "folder/second.txt"],
    }

    accepted = run_eval._acceptable_paths(case, tmp_path)

    assert accepted == {
        (tmp_path / "first.txt").resolve().as_posix().casefold(),
        (tmp_path / "second.txt").resolve().as_posix().casefold(),
    }


def test_threshold_sweep_uses_cached_scores_and_top_three():
    correct = "C:/corpus/correct.txt"
    wrong = "C:/corpus/wrong.txt"
    no_match_fp = "C:/corpus/near_match.txt"
    results = [
        {
            "acceptable_paths": {correct.casefold()},
            "best_path": correct,
            "best_score": 0.65,
            "top3_paths": [wrong, correct],
        },
        {
            "acceptable_paths": set(),
            "best_path": no_match_fp,
            "best_score": 0.52,
            "top3_paths": [no_match_fp],
        },
    ]

    sweep = run_eval._run_threshold_sweep(results, thresholds=(0.5, 0.7))

    assert sweep == [
        {
            "threshold": 0.5,
            "positive_hit_rate": 1.0,
            "false_positive_rate": 1.0,
            "top1_accuracy": 0.5,
            "top3_accuracy": 0.5,
        },
        {
            "threshold": 0.7,
            "positive_hit_rate": 0.0,
            "false_positive_rate": 0.0,
            "top1_accuracy": 0.5,
            "top3_accuracy": 0.5,
        },
    ]
