"""
app_paths.py

Resolves where the app should store its working data (databases, logs,
indexes) -- always in a user-writable folder, never inside the app's own
installation directory. This matters specifically once installed via the
Inno Setup installer to Program Files, which is read-only for normal
(non-admin) processes -- writing there fails with "attempt to write a
readonly database" errors, confirmed during real installer testing.
"""

import sys
import os
import json
from pathlib import Path

APP_FOLDER_NAME = "SmartFileFinder"
MODEL_TIER_DEFAULTS = {
    "tiers": {
        "big": {
            "min_ram_gb": 24,
            "min_vram_gb": 12,
            "min_free_disk_gb": 42,
            "vision_model": "google/gemma-4-E4B-it (4-bit, untested)",
            "whisper_model_size": "small/medium",
            "summarizer": "yes",
        },
        "mid": {
            "min_ram_gb": 12,
            "min_vram_gb": 4,
            "min_free_disk_gb": 18,
            "vision_model": "google/gemma-4-E2B-it (4-bit)",
            "whisper_model_size": "base",
            "summarizer": "yes",
        },
        "small": {
            "min_ram_gb": 0,
            "min_vram_gb": 0,
            "min_free_disk_gb": 3,
            "vision_model": "HuggingFaceTB/SmolVLM-500M-Instruct",
            "whisper_model_size": "tiny",
            "summarizer": "no",
        },
    },
    "models": [
        {
            "name": "all-mpnet-base-v2",
            "purpose": "embedding",
            "repo_id": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "download_size_gb": 0.43867131,
            "download_size_bytes": 438671310,
            "repository_size_gb": 3.824846935,
            "repository_size_bytes": 3824846935,
            "repository_file_count": 28,
            "tiers": ["small", "mid", "big"],
            "status": "ready",
            "files": {
                "1_Pooling/config.json": 190,
                "config.json": 571,
                "config_sentence_transformers.json": 116,
                "model.safetensors": 437971872,
                "modules.json": 349,
                "sentence_bert_config.json": 53,
                "special_tokens_map.json": 239,
                "tokenizer.json": 466021,
                "tokenizer_config.json": 363,
                "vocab.txt": 231536,
            },
        },
        {
            "name": "Gemma 4 E2B-it",
            "purpose": "vision",
            "repo_id": "google/gemma-4-E2B-it",
            "revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
            "download_size_gb": 10.278806153,
            "download_size_bytes": 10278806153,
            "repository_size_gb": 10.278835678,
            "repository_size_bytes": 10278835678,
            "repository_file_count": 9,
            "family": "gemma4",
            "requires_cuda": True,
            "tiers": ["mid", "big"],
            "status": "ready",
            "prompt": "Describe this image in one or two sentences, focusing on what it depicts.",
            "max_new_tokens": 60,
            "license": "apache-2.0",
            "gated": False,
            "files": {
                "chat_template.jinja": 18569,
                "config.json": 4954,
                "generation_config.json": 208,
                "model.safetensors": 10246621918,
                "processor_config.json": 1689,
                "tokenizer.json": 32155733,
                "tokenizer_config.json": 3082,
            },
        },
        {
            "name": "SmolVLM-500M-Instruct",
            "purpose": "vision",
            "repo_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
            "revision": "753c1752602b5e739af7a7fa6947a635051897a3",
            "download_size_gb": 1.01988383,
            "download_size_bytes": 1019883830,
            "repository_size_gb": 6.902685746,
            "repository_size_bytes": 6902685746,
            "repository_file_count": 38,
            "family": "smolvlm",
            "requires_cuda": False,
            "tiers": ["small", "mid", "big"],
            "status": "ready",
            "prompt": "Describe only the visible subject, objects, text, and setting in one concise sentence. Do not guess or add details that are not clearly visible.",
            "max_new_tokens": 60,
            "license": "apache-2.0",
            "gated": False,
            "files": {
                "added_tokens.json": 4739,
                "chat_template.json": 516,
                "config.json": 7339,
                "generation_config.json": 136,
                "merges.txt": 466391,
                "model.safetensors": 1015025832,
                "preprocessor_config.json": 486,
                "processor_config.json": 68,
                "special_tokens_map.json": 1069,
                "tokenizer.json": 3548256,
                "tokenizer_config.json": 28336,
                "vocab.json": 800662,
            },
        },
        {
            "name": "SmolVLM-256M-Instruct (benchmark candidate)",
            "purpose": "vision",
            "repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "revision": "d039ab0d4b9311b617e8ffec1d06c8464364aeb3",
            "download_size_gb": 0.51788682,
            "download_size_bytes": 517886820,
            "repository_size_gb": 3.542912443,
            "repository_size_bytes": 3542912443,
            "repository_file_count": 38,
            "family": "smolvlm",
            "requires_cuda": False,
            "tiers": [],
            "candidate_tier": "small",
            "benchmark_only": True,
            "status": "ready",
            "prompt": "Describe only the visible subject, objects, text, and setting in one concise sentence. Do not guess or add details that are not clearly visible.",
            "max_new_tokens": 60,
            "license": "apache-2.0",
            "gated": False,
            "files": {
                "added_tokens.json": 4739,
                "chat_template.json": 516,
                "config.json": 7353,
                "generation_config.json": 136,
                "merges.txt": 466391,
                "model.safetensors": 513028808,
                "preprocessor_config.json": 486,
                "processor_config.json": 68,
                "special_tokens_map.json": 1069,
                "tokenizer.json": 3548256,
                "tokenizer_config.json": 28336,
                "vocab.json": 800662,
            },
        },
        {
            "name": "Gemma 4 E4B-it",
            "purpose": "vision",
            "repo_id": "google/gemma-4-E4B-it",
            "revision": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
            "download_size_gb": 16.024781005,
            "download_size_bytes": 16024781005,
            "repository_size_gb": 16.024810531,
            "repository_size_bytes": 16024810531,
            "repository_file_count": 9,
            "family": "gemma4",
            "requires_cuda": True,
            "tiers": ["big"],
            "status": "untested",
            "prompt": "Describe this image in one or two sentences, focusing on what it depicts.",
            "max_new_tokens": 60,
            "license": "apache-2.0",
            "gated": False,
            "files": {
                "chat_template.jinja": 18569,
                "config.json": 5145,
                "generation_config.json": 208,
                "model.safetensors": 15992595884,
                "processor_config.json": 1689,
                "tokenizer.json": 32156428,
                "tokenizer_config.json": 3082,
            },
        },
    ],
}


def get_app_root_dir():
    """Return the base for resolving app-relative paths."""
    if getattr(sys, "frozen", False):
        return get_base_dir()
    return Path(__file__).resolve().parent


def normalize_path(path, relative_to=None):
    """Return an absolute path using consistent forward slashes."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (Path(relative_to) if relative_to is not None
                    else get_app_root_dir()) / resolved
    return resolved.resolve().as_posix()


def get_base_dir():
    """
    Returns a writable, per-user folder for the app's working data
    (%LOCALAPPDATA%\\SmartFileFinder on Windows), regardless of whether
    running from source or as a packaged/installed .exe -- since the
    installed .exe's own folder (Program Files) is not writable.
    """
    override = os.environ.get("SMART_FILE_FINDER_EVAL_DATA_DIR")
    if override:
        base_dir = Path(override).expanduser()
    else:
        base_dir = _default_base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _default_base_dir():
    app_data = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    base_dir = app_data / APP_FOLDER_NAME
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_hardware_profile_path():
    """Return the writable path used to persist the latest hardware report."""
    return get_base_dir() / "hardware_profile.json"


def get_models_dir():
    """Return the writable app-local Hugging Face model cache directory."""
    models_dir = _default_base_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _model_cache_path(cache_dir, repo_id, revision):
    repo_folder = "models--" + repo_id.replace("/", "--")
    return Path(cache_dir) / repo_folder / "snapshots" / revision


def find_model(repo_id):
    """Find a complete pinned model snapshot in app storage or the HF cache."""
    config_path = get_config_path("model_tiers.json")
    with open(config_path, encoding="utf-8-sig") as config_file:
        config = json.load(config_file)
    model = next(
        (item for item in config.get("models", []) if item.get("repo_id") == repo_id),
        None,
    )
    if model is None or model.get("status") not in {"ready", "untested"}:
        return None

    revision = model.get("revision")
    expected_files = model.get("files", {})
    if not revision or not expected_files:
        return None

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    default_cache = Path(
        os.environ.get("HF_HUB_CACHE", hf_home / "hub")
    )
    app_models_dir = get_models_dir()
    for cache_dir in (app_models_dir, default_cache):
        snapshot_dir = _model_cache_path(cache_dir, repo_id, revision)
        try:
            is_complete = all(
                (snapshot_dir / name).is_file()
                and (snapshot_dir / name).stat().st_size == size
                for name, size in expected_files.items()
            )
        except OSError:
            is_complete = False
        if not is_complete:
            continue
        if cache_dir == app_models_dir:
            marker_path = snapshot_dir / ".smart-file-finder-complete.json"
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if marker.get("repo_id") != repo_id or marker.get("revision") != revision:
                continue
        return snapshot_dir
    return None


def get_config_path(filename):
    """Return a config file path independent of the current working directory."""
    if getattr(sys, "frozen", False):
        config_dir = Path(sys.executable).resolve().parent / "config"
    else:
        config_dir = Path(__file__).resolve().parent / "config"

    config_path = config_dir / filename
    defaults = {
        "settings.json": {
            "indexed_folders": [],
            "performance_tier": "auto",
        },
        "model_tiers.json": MODEL_TIER_DEFAULTS,
        "harness_limits.json": {
            "max_reformulations": 2,
            "timeout_seconds": 5,
            "confidence_threshold": 0.55,
            "ambiguity_margin": 0.07,
            "options_floor": 0.48,
            "min_option_score": 0.50,
            "option_score_window": 0.03,
            "max_options": 3,
        },
    }
    if not config_path.exists() and filename in defaults:
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(defaults[filename], indent=2) + "\n",
            encoding="utf-8",
        )
        from observability.logger import log

        log(f"Created missing config file with defaults: {config_path}")

    return config_path


def ensure_config_exists():
    """Ensure the runtime configuration files exist with usable defaults."""
    get_config_path("settings.json")
    get_config_path("harness_limits.json")
    get_config_path("model_tiers.json")


def get_resource_path(filename):
    """
    Returns the path to a bundled static resource (icons, etc.) that
    ships with the app -- resolved relative to the .exe's own folder
    when frozen, or the project root when running from source.
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / filename
    else:
        return Path(__file__).parent / filename