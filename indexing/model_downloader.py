"""Registry-backed, resumable downloads for the public model snapshots."""

import json
import os
import shutil
import threading
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download
from huggingface_hub import constants as _hub_constants
from huggingface_hub.utils import tqdm as _hub_tqdm

from app_paths import (
    find_model,
    get_config_path,
    get_models_dir,
    get_required_models_for_tier,
)


class ModelDownloadError(RuntimeError):
    """A model could not be downloaded or verified."""


class InsufficientDiskSpaceError(ModelDownloadError):
    """The model will not fit on the app-data drive."""


class ModelDownloadCancelled(ModelDownloadError):
    """The user cancelled a model download."""


def set_huggingface_offline(enabled):
    """Keep the environment and already-imported Hub runtime in sync."""
    if enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
    _hub_constants.HF_HUB_OFFLINE = bool(enabled)


def _registry_models():
    with open(get_config_path("model_tiers.json"), encoding="utf-8-sig") as file:
        return json.load(file).get("models", [])


def models_for_tier(tier):
    """Return exactly the embedding and primary vision model for a tier."""
    config = json.loads(
        Path(get_config_path("model_tiers.json")).read_text(encoding="utf-8-sig")
    )
    return get_required_models_for_tier(tier, config)


def missing_models_for_tier(tier):
    """Return ready tier models whose pinned snapshots are incomplete."""
    missing = []
    for model in models_for_tier(tier):
        if find_model(model["repo_id"]) is None:
            missing.append(model)
    return missing


def _snapshot_path(cache_dir, repo_id, revision):
    repo_dir = "models--" + repo_id.replace("/", "--")
    return Path(cache_dir) / repo_dir / "snapshots" / revision


def _existing_complete_bytes(snapshot_dir, files):
    completed = 0
    for relative_path, expected_size in files.items():
        path = snapshot_dir / relative_path
        try:
            if path.is_file() and path.stat().st_size == expected_size:
                completed += expected_size
        except OSError:
            continue
    return completed


def download_model(model, progress_cb=None, cancel_event=None):
    """Download and verify one pinned snapshot; Hub cache files allow resume."""
    if model.get("status") not in {"ready", "untested"} or not model.get("repo_id"):
        raise ModelDownloadError("This model is planned and cannot be downloaded yet.")

    repo_id = model["repo_id"]
    revision = model.get("revision")
    files = model.get("files") or {}
    total_bytes = int(model.get("download_size_bytes") or sum(files.values()))
    if not revision or not files or total_bytes <= 0:
        raise ModelDownloadError(f"Model registry entry is incomplete: {repo_id}")

    models_dir = get_models_dir()
    snapshot_dir = _snapshot_path(models_dir, repo_id, revision)
    completed_bytes = _existing_complete_bytes(snapshot_dir, files)
    required_bytes = max(0, total_bytes - completed_bytes)
    free_bytes = shutil.disk_usage(models_dir).free
    if free_bytes < required_bytes:
        raise InsufficientDiskSpaceError(
            f"Not enough free disk space for {model['name']}: "
            f"{required_bytes / 1_000_000:.0f} MB required, "
            f"{free_bytes / 1_000_000:.0f} MB available."
        )

    if cancel_event is not None and cancel_event.is_set():
        raise ModelDownloadCancelled("Model download cancelled.")

    marker_path = snapshot_dir / ".smart-file-finder-complete.json"
    marker_path.unlink(missing_ok=True)

    started_at = time.monotonic()
    last_reported = completed_bytes
    downloaded_bytes = 0
    progress_lock = threading.Lock()

    def report_progress(done):
        nonlocal last_reported
        done = max(last_reported, min(total_bytes, int(done)))
        last_reported = done
        if progress_cb is not None:
            elapsed = max(time.monotonic() - started_at, 0.25)
            progress_cb(done, total_bytes, downloaded_bytes / elapsed)

    report_progress(completed_bytes)

    class ProgressTqdm(_hub_tqdm):
        def __init__(self, *args, **kwargs):
            unit = str(kwargs.get("unit", "")).casefold()
            description = str(kwargs.get("desc", "")).casefold()
            self._tracks_bytes = (
                unit in {"b", "byte", "bytes"}
                and "reconstruct" not in description
            )
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if cancel_event is not None and cancel_event.is_set():
                raise ModelDownloadCancelled("Model download cancelled.")
            result = super().update(n)
            if self._tracks_bytes:
                delta = max(0, int(n or 0))
                with progress_lock:
                    nonlocal downloaded_bytes
                    downloaded_bytes += delta
                    current = completed_bytes + downloaded_bytes
                report_progress(current)
            if cancel_event is not None and cancel_event.is_set():
                raise ModelDownloadCancelled("Model download cancelled.")
            return result

    set_huggingface_offline(False)
    try:
        downloaded_path = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(models_dir),
            token=False,
            allow_patterns=list(files),
            tqdm_class=ProgressTqdm,
        ))
        for relative_path, expected_size in files.items():
            path = downloaded_path / relative_path
            if not path.is_file() or path.stat().st_size != expected_size:
                raise ModelDownloadError(
                    f"Downloaded model is incomplete: {repo_id} is missing "
                    f"{relative_path} or its size does not match the pinned revision."
                )
        if cancel_event is not None and cancel_event.is_set():
            raise ModelDownloadCancelled("Model download cancelled.")
        marker_path = downloaded_path / ".smart-file-finder-complete.json"
        marker_path.write_text(
            json.dumps({"repo_id": repo_id, "revision": revision}) + "\n",
            encoding="utf-8",
        )
        report_progress(total_bytes)
        return downloaded_path
    except ModelDownloadCancelled:
        marker_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        marker_path.unlink(missing_ok=True)
        if cancel_event is not None and cancel_event.is_set():
            raise ModelDownloadCancelled("Model download cancelled.") from error
        if isinstance(error, ModelDownloadError):
            raise
        raise ModelDownloadError(
            f"Could not download {model['name']} from Hugging Face. "
            f"Check your internet connection and the repository/revision: {error}"
        ) from error
