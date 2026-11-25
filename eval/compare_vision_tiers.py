"""Compare local vision models on golden-set images without touching the real index."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"


def _read_registry():
    with open(PROJECT_ROOT / "config" / "model_tiers.json", encoding="utf-8-sig") as file:
        registry = json.load(file)
    return registry["models"] + registry.get("optional_models", [])


def _progress_printer(name):
    last_percent = [-1]

    def report(done, total, _speed):
        percent = int(done * 100 / total) if total else 100
        if percent != last_percent[0]:
            last_percent[0] = percent
            print(f"Downloading {name}: {percent}% ({done}/{total} bytes)", flush=True)

    return report


def _ensure_model(model):
    from app_paths import find_model
    from indexing.model_downloader import download_model, set_huggingface_offline

    if find_model(model["repo_id"]) is not None:
        return
    set_huggingface_offline(False)
    print(f"Downloading comparison model: {model['repo_id']}", flush=True)
    download_model(
        model,
        progress_cb=_progress_printer(model["name"]),
        download_source="vision_comparison",
    )


def _worker(args):
    from app.hardware_profile import set_current_tier
    from app_paths import normalize_path
    from indexing.backfill_index import index_one_file
    from indexing.embedder import preload_model
    from indexing.vision_pipeline import VISION_MODELS, unload_vision_model
    from indexing import vision_pipeline
    from storage.metadata_store import get_file_metadata
    from storage.vector_db.vector_store import _get_collection

    set_current_tier(args.tier)
    selected_model = next(
        model for model in VISION_MODELS
        if model.get("repo_id") == args.repo
    )
    if args.big_vision_model:
        vision_pipeline.get_big_vision_model_setting = (
            lambda: args.big_vision_model
        )
    if args.tier == "small":
        for model in VISION_MODELS:
            if (
                model.get("candidate_tier") == "small"
                or "small" in model.get("tiers", [])
            ):
                model["tiers"] = (
                    ["small"] if model["repo_id"] == args.repo else []
                )

    with open(GOLDEN_SET_PATH, encoding="utf-8-sig") as file:
        cases = json.load(file)
    image_names = sorted({
        name
        for case in cases
        for name in (
            case.get("acceptable_files")
            or ([case["expected_file"]] if case.get("expected_file") else [])
        )
        if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    })
    image_paths = [
        Path(normalize_path(PROJECT_ROOT / "test_data" / name))
        for name in image_names
    ]

    import psutil

    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop_sampling = threading.Event()

    def sample_memory():
        while not stop_sampling.wait(0.1):
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    preload_model()
    durations = {}
    try:
        for image_path in image_paths:
            started = time.perf_counter()
            status = index_one_file(
                image_path, force=True, retain_vision_model=True
            )
            durations[image_path.name] = time.perf_counter() - started
            if status.startswith("[ERROR]"):
                raise RuntimeError(status)
        unload_vision_model()

        descriptions = {}
        collection = _get_collection()
        for image_path in image_paths:
            stored_path = normalize_path(image_path)
            stored = collection.get(
                ids=[stored_path], include=["metadatas"]
            )
            metadata = (stored.get("metadatas") or [None])[0]
            model_used = (
                (get_file_metadata(image_path) or {}).get("vision_model")
            )
            if (
                args.repo == "google/gemma-4-E4B-it"
                and model_used != args.repo
            ):
                raise RuntimeError(
                    f"Gemma E4B was not used for {image_path.name}; "
                    f"active vision model was {model_used or 'filename-only'}."
                )
            descriptions[image_path.name] = {
                "text": metadata.get("description", "[filename-only]")
                if metadata else "[filename-only]",
                "vision_model": model_used,
            }

        from app import query_handler

        image_cases = []
        for case in cases:
            expected_names = case.get("acceptable_files")
            if expected_names is None:
                expected_names = [case.get("expected_file")] if case.get(
                    "expected_file"
                ) else []
            if any(
                Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                for name in expected_names
            ):
                image_cases.append(case)
        hit_count = 0
        shown_hit_count = 0
        for case in image_cases:
            attempts = []
            original_retrieve = query_handler.retrieve

            def capture_retrieve(query):
                result = original_retrieve(query)
                attempts.append(result)
                return result

            query_handler.retrieve = capture_retrieve
            try:
                outcome = query_handler.handle_query(case["query"])
            finally:
                query_handler.retrieve = original_retrieve

            acceptable = {
                normalize_path(PROJECT_ROOT / "test_data" / name).casefold()
                for name in (
                    case.get("acceptable_files")
                    or [case["expected_file"]]
                )
            }
            options = outcome.get("options", [])
            returned = (
                [normalize_path(option["path"]) for option in options]
                if outcome.get("action") == "options"
                else (
                    [normalize_path(outcome["path"])]
                    if outcome.get("found") else []
                )
            )
            first = returned[0].casefold() if returned else None
            hit_count += first in acceptable
            shown_hit_count += bool(
                acceptable & {path.casefold() for path in returned}
            )

        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        result = {
            "model": selected_model["name"],
            "repo_id": args.repo,
            "tier": args.tier,
            "seconds_per_image": (
                sum(durations.values()) / len(durations) if durations else 0
            ),
            "seconds_by_image": durations,
            "peak_rss_mb": peak_rss[0] / 1_000_000,
            "image_cases": len(image_cases),
            "top1_hit": hit_count / len(image_cases) if image_cases else 0,
            "shown_hit": shown_hit_count / len(image_cases) if image_cases else 0,
            "descriptions": descriptions,
        }
        Path(args.output).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    finally:
        stop_sampling.set()
        sampler.join(timeout=2)
        unload_vision_model()


def _build_comparisons(registry, include_e4b=False):
    comparisons = [
        (model, "small")
        for model in registry
        if model.get("purpose") == "vision"
        and model.get("candidate_tier") == "small"
        and model.get("family") == "smolvlm"
    ]
    selected_small = next(
        model for model in registry
        if model.get("purpose") == "vision"
        and "small" in model.get("tiers", [])
    )
    if all(model["repo_id"] != selected_small["repo_id"] for model, _ in comparisons):
        comparisons.insert(0, (selected_small, "small"))
    mid_model = next(
        model for model in registry
        if model.get("purpose") == "vision" and "mid" in model.get("tiers", [])
    )
    comparisons.append((mid_model, "mid"))
    if include_e4b:
        e4b_model = next(
            (
                model for model in registry
                if model.get("purpose") == "vision"
                and model.get("setting_value") == "gemma-4-E4B-it"
                and model.get("status") in {"ready", "untested"}
            ),
            None,
        )
        if e4b_model is None:
            raise RuntimeError(
                "The model registry does not contain a usable Gemma E4B entry."
            )
        comparisons.append((e4b_model, "big"))
    return comparisons, mid_model


def _require_e4b_hardware():
    from app.hardware_profile import detect_hardware

    profile = detect_hardware()
    requirements = {
        "ram_gb": (24, "GB RAM"),
        "vram_gb": (12, "GB NVIDIA VRAM"),
        "free_disk_gb": (28, "GB free disk"),
    }
    unmet = [
        f"{minimum} {label} required (detected {profile.get(key, 0) or 0})"
        for key, (minimum, label) in requirements.items()
        if float(profile.get(key) or 0) < minimum
    ]
    if unmet:
        raise RuntimeError(
            "Gemma E4B comparison requires a compatible test PC: "
            + "; ".join(unmet)
        )


def _run_parent(include_e4b=False):
    from app_paths import find_model
    registry = _read_registry()
    if include_e4b:
        _require_e4b_hardware()
    comparisons, mid_model = _build_comparisons(registry, include_e4b)

    for model, _tier in comparisons:
        _ensure_model(model)
    if find_model(mid_model["repo_id"]) is None:
        raise RuntimeError("The cached mid-tier Gemma E2B model is incomplete.")

    with tempfile.TemporaryDirectory(prefix="sff-vision-eval-") as temp_root:
        outputs = []
        for index, (model, tier) in enumerate(comparisons):
            data_dir = Path(temp_root) / f"data-{index}"
            output_path = Path(temp_root) / f"result-{index}.json"
            environment = os.environ.copy()
            environment["SMART_FILE_FINDER_EVAL_DATA_DIR"] = str(data_dir)
            environment["HF_HUB_OFFLINE"] = "1"
            if tier == "small":
                environment["CUDA_VISIBLE_DEVICES"] = ""
            else:
                environment.pop("CUDA_VISIBLE_DEVICES", None)
            command = [
                sys.executable,
                "-m",
                "eval.compare_vision_tiers",
                "--worker",
                "--repo",
                model["repo_id"],
                "--tier",
                tier,
                "--output",
                str(output_path),
            ]
            if tier == "big":
                command.extend([
                    "--big-vision-model",
                    "gemma-4-E4B-it",
                ])
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
            )
            outputs.append(json.loads(output_path.read_text(encoding="utf-8")))

    print("\nImage-set comparison")
    print("| Model | Seconds/image | Peak RAM (MB) | Top-1 hit | Shown hit |")
    print("|---|---:|---:|---:|---:|")
    for result in outputs:
        print(
            f"| {result['model']} | {result['seconds_per_image']:.2f} | "
            f"{result['peak_rss_mb']:.0f} | {result['top1_hit']:.0%} | "
            f"{result['shown_hit']:.0%} |"
        )
    for result in outputs:
        print(f"\n{result['model']} descriptions:")
        for name, description in result["descriptions"].items():
            print(
                f"- {name} [{description['vision_model'] or 'filename-only'}]: "
                f"{description['text']}"
            )
    if not include_e4b:
        print(
            "\nGemma E4B was not included. On a compatible PC, use "
            "--include-e4b to benchmark it."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--tier")
    parser.add_argument("--output")
    parser.add_argument(
        "--include-e4b",
        action="store_true",
        help=(
            "Download and benchmark optional Gemma E4B; requires at least "
            "24 GB RAM, 12 GB NVIDIA VRAM, and 28 GB free disk."
        ),
    )
    parser.add_argument("--big-vision-model")
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        _run_parent(include_e4b=args.include_e4b)


if __name__ == "__main__":
    main()
