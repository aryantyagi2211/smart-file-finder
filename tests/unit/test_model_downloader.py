import json
import io
import os
import time
from pathlib import Path

import pytest

import app_paths
from app import setup_window
from indexing import model_downloader


@pytest.fixture
def tiny_registry(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = {
        "name": "tiny-test-model",
        "purpose": "embedding",
        "repo_id": "test/tiny-model",
        "revision": "0123456789abcdef",
        "download_size_gb": 0.000000008,
        "download_size_bytes": 8,
        "tiers": ["small", "mid", "big"],
        "status": "ready",
        "files": {"config.json": 4, "weights.bin": 4},
    }
    vision_model = {
        "name": "tiny-test-vision",
        "purpose": "vision",
        "repo_id": "test/tiny-vision",
        "revision": "abcdef0123456789",
        "download_size_gb": 0.000000008,
        "download_size_bytes": 8,
        "tiers": ["small"],
        "status": "ready",
        "files": {"config.json": 4, "weights.bin": 4},
    }
    voice_model = {
        "name": "tiny-test-voice",
        "purpose": "voice",
        "repo_id": "test/tiny-voice",
        "revision": "fedcba9876543210",
        "download_size_gb": 0.000000008,
        "download_size_bytes": 8,
        "tiers": ["small"],
        "status": "ready",
        "files": {"config.json": 4, "weights.bin": 4},
    }
    config_path = tmp_path / "model_tiers.json"
    config_path.write_text(json.dumps({
        "tiers": {"small": {"vision_model_repo_id": vision_model["repo_id"]}},
        "models": [model, vision_model, voice_model],
        "optional_models": [],
    }), encoding="utf-8")
    monkeypatch.setattr(app_paths, "get_config_path", lambda name: config_path)
    monkeypatch.setattr(model_downloader, "get_config_path", lambda name: config_path)
    monkeypatch.setattr(app_paths, "get_models_dir", lambda: models_dir)
    monkeypatch.setattr(model_downloader, "get_models_dir", lambda: models_dir)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "default_hub_cache"))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return model, vision_model, models_dir, config_path


def _write_snapshot(cache_dir, model):
    snapshot = (
        Path(cache_dir)
        / ("models--" + model["repo_id"].replace("/", "--"))
        / "snapshots"
        / model["revision"]
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_bytes(b"conf")
    (snapshot / "weights.bin").write_bytes(b"data")
    return snapshot


def test_missing_models_lists_only_models_without_complete_pinned_files(
    tiny_registry, monkeypatch
):
    model, vision_model, models_dir, config_path = tiny_registry
    voice_model = json.loads(
        config_path.read_text(encoding="utf-8")
    )["models"][2]

    assert model_downloader.missing_models_for_tier("small") == [
        model, vision_model, voice_model
    ]
    snapshot = _write_snapshot(models_dir, model)
    (snapshot / ".smart-file-finder-complete.json").write_text(
        json.dumps({"repo_id": model["repo_id"], "revision": model["revision"]}),
        encoding="utf-8",
    )

    assert app_paths.find_model(model["repo_id"]) == snapshot
    assert model_downloader.missing_models_for_tier("small") == [
        vision_model, voice_model
    ]
    vision_snapshot = _write_snapshot(models_dir, vision_model)
    (vision_snapshot / ".smart-file-finder-complete.json").write_text(
        json.dumps({
            "repo_id": vision_model["repo_id"],
            "revision": vision_model["revision"],
        }),
        encoding="utf-8",
    )
    assert model_downloader.missing_models_for_tier("small") == [voice_model]
    voice_snapshot = _write_snapshot(models_dir, voice_model)
    (voice_snapshot / ".smart-file-finder-complete.json").write_text(
        json.dumps({
            "repo_id": voice_model["repo_id"],
            "revision": voice_model["revision"],
        }),
        encoding="utf-8",
    )
    assert model_downloader.missing_models_for_tier("small") == []


def test_incomplete_model_folder_is_not_reported_present(tiny_registry):
    model, _, models_dir, _ = tiny_registry
    snapshot = (
        models_dir
        / ("models--" + model["repo_id"].replace("/", "--"))
        / "snapshots"
        / model["revision"]
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"conf")
    (snapshot / ".smart-file-finder-complete.json").write_text(
        json.dumps({"repo_id": model["repo_id"], "revision": model["revision"]}),
        encoding="utf-8",
    )

    assert app_paths.find_model(model["repo_id"]) is None


def test_low_disk_space_refuses_before_writing_model(
    tiny_registry, monkeypatch
):
    model, _, models_dir, _ = tiny_registry
    monkeypatch.setattr(
        model_downloader.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 7})(),
    )
    called = []
    monkeypatch.setattr(
        model_downloader,
        "snapshot_download",
        lambda **kwargs: called.append(kwargs),
    )

    with pytest.raises(model_downloader.InsufficientDiskSpaceError, match="Not enough free"):
        model_downloader.download_model(model)

    assert called == []
    assert list(models_dir.iterdir()) == []


def test_failed_download_can_be_retried_and_progress_is_monotonic(
    tiny_registry, monkeypatch
):
    model, _, models_dir, _ = tiny_registry
    attempts = []

    def fake_snapshot_download(
        repo_id, revision, cache_dir, tqdm_class, token, allow_patterns
    ):
        attempts.append(1)
        assert "HF_HUB_OFFLINE" not in os.environ
        if len(attempts) == 1:
            snapshot = (
                Path(cache_dir)
                / "models--test--tiny-model"
                / "snapshots"
                / revision
            )
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "weights.bin.incomplete").write_bytes(b"da")
            raise OSError("temporary network interruption")
        snapshot = _write_snapshot(cache_dir, model)
        with tqdm_class(total=8, unit="B", disable=True) as progress:
            progress.update(4)
            progress.update(4)
        return str(snapshot)

    monkeypatch.setattr(model_downloader, "snapshot_download", fake_snapshot_download)
    updates = []
    with pytest.raises(model_downloader.ModelDownloadError, match="internet connection"):
        model_downloader.download_model(model)
    result = model_downloader.download_model(
        model,
        progress_cb=lambda done, total, speed: updates.append((done, total, speed)),
    )

    assert len(attempts) == 2
    assert result == app_paths.find_model(model["repo_id"])
    assert [done for done, _, _ in updates] == sorted(
        done for done, _, _ in updates
    )
    assert updates[-1][0] == updates[-1][1] == 8
    assert updates[-1][2] >= 0


def test_cancelled_download_is_not_reported_present(tiny_registry, monkeypatch):
    model, _, _, _ = tiny_registry
    cancel_event = __import__("threading").Event()

    def fake_snapshot_download(
        repo_id, revision, cache_dir, tqdm_class, token, allow_patterns
    ):
        with tqdm_class(
            total=8, unit="B", file=io.StringIO(), disable=False
        ) as progress:
            progress.update(4)
        return str(_write_snapshot(cache_dir, model))

    monkeypatch.setattr(model_downloader, "snapshot_download", fake_snapshot_download)

    def cancel_after_progress(done, total, speed):
        if done:
            cancel_event.set()

    with pytest.raises(model_downloader.ModelDownloadCancelled):
        model_downloader.download_model(
            model, cancel_after_progress, cancel_event
        )

    assert app_paths.find_model(model["repo_id"]) is None


def test_download_requires_explicit_click_and_retries_missing_model(
    tiny_registry, monkeypatch
):
    from PyQt6.QtWidgets import QApplication

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    model, vision_model, _, config_path = tiny_registry
    voice_model = json.loads(
        config_path.read_text(encoding="utf-8")
    )["models"][2]
    monkeypatch.setattr(setup_window, "find_model", lambda repo_id: None)
    calls = []
    statuses = []

    def fake_download(model_arg, progress_cb, cancel_event):
        calls.append(model_arg["repo_id"])
        progress_cb(8, 8, 1000)

    monkeypatch.setattr(setup_window, "download_model", fake_download)
    window = setup_window.SetupWindow(
        "small", [model, vision_model, voice_model], 10
    )
    window.status_changed.connect(statuses.append)
    assert calls == []

    window.download_button.click()
    deadline = time.monotonic() + 3
    while window._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()

    assert calls == [
        model["repo_id"],
        vision_model["repo_id"],
        voice_model["repo_id"],
    ]
    assert statuses == [
        "Downloading embedding model: tiny-test-model...",
        "Downloading vision model: tiny-test-vision...",
        "Downloading voice model: tiny-test-voice...",
        "Model downloads complete; preparing to load...",
    ]
    assert window.total_bar.value() == 1000
    window.close()


def test_startup_opens_setup_only_when_a_required_model_is_missing(
    tiny_registry, monkeypatch
):
    from app import main as app_main

    model, _, _, _ = tiny_registry

    class FakeSignal:
        def connect(self, callback):
            pass

    class FakeAction:
        def setText(self, text):
            self.text = text

        def setToolTip(self, text):
            self.tooltip = text

    class FakeTierStatus:
        def set_performance_tier(self, tier, mode):
            self.selection = (tier, mode)

        def show_status(self, message):
            self.status = message

        def show(self):
            self.shown = True

    class FakeTrayIcon:
        def showMessage(self, *args):
            self.message = args

    class FakeApp:
        def quit(self):
            pass

    class FakeSetupWindow:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.ready = FakeSignal()
            self.cancelled = FakeSignal()
            self.status_changed = FakeSignal()
            self.shown = False

        def show(self):
            self.shown = True

    instance = object.__new__(app_main.SmartFileFinderApp)
    instance.performance_tier_action = FakeAction()
    instance.status_window = FakeTierStatus()
    instance.tray_icon = FakeTrayIcon()
    instance.app = FakeApp()
    instance.setup_window = None
    instance.voice_enabled = False
    instance.voice_available = True
    ready = []
    created = []
    monkeypatch.setattr(app_main, "set_current_tier", lambda tier: None)
    monkeypatch.setattr(app_main, "missing_models_for_tier", lambda tier: [])
    monkeypatch.setattr(
        app_main.SmartFileFinderApp,
        "_on_models_ready",
        lambda self: ready.append(True),
    )
    monkeypatch.setattr(
        app_main,
        "SetupWindow",
        lambda *args, **kwargs: created.append(
            FakeSetupWindow(*args, **kwargs)
        ) or created[-1],
    )

    instance._on_hardware_tier_ready("small", "auto", 10)

    assert ready == [True]
    assert created == []

    monkeypatch.setattr(
        app_main, "missing_models_for_tier", lambda tier: [model]
    )
    instance._on_hardware_tier_ready("small", "auto", 10)

    assert len(created) == 1
    assert created[0].shown
    assert created[0].args[1] == [model]

    voice_model = {
        "name": "Whisper tiny.en",
        "purpose": "voice",
        "repo_id": "test/voice",
    }
    instance.voice_enabled = True
    monkeypatch.setattr(
        app_main,
        "missing_voice_model_for_tier",
        lambda tier: voice_model,
    )
    created.clear()
    instance._on_hardware_tier_ready("small", "auto", 10)

    assert created[0].args[1] == [model]


def test_model_registry_defaults_keep_untested_big_model_optional(
    tmp_path, monkeypatch
):
    project = tmp_path / "source"
    project.mkdir()
    monkeypatch.setattr(app_paths, "__file__", str(project / "app_paths.py"))

    config_path = app_paths.get_config_path("model_tiers.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert len([model for model in config["models"] if model["status"] == "ready"]) == 8
    assert any(
        model["status"] == "untested"
        and model["repo_id"] == "google/gemma-4-E4B-it"
        for model in config["optional_models"]
    )
    assert config["tiers"]["small"]["min_free_disk_gb"] == 3
    assert config["tiers"]["mid"]["min_free_disk_gb"] == 19
    assert config["tiers"]["big"]["min_free_disk_gb"] == 19


def test_each_tier_requires_embedding_primary_vision_and_voice_models():
    small = model_downloader.models_for_tier("small")
    mid = model_downloader.models_for_tier("mid")
    big = model_downloader.models_for_tier("big")

    assert {model["repo_id"] for model in small} == {
        "sentence-transformers/all-mpnet-base-v2",
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "Systran/faster-whisper-small.en",
    }
    assert {model["repo_id"] for model in mid} == {
        "sentence-transformers/all-mpnet-base-v2",
        "google/gemma-4-E2B-it",
        "dropbox-dash/faster-whisper-large-v3-turbo",
    }
    assert {model["repo_id"] for model in big} == {
        "sentence-transformers/all-mpnet-base-v2",
        "google/gemma-4-E2B-it",
        "dropbox-dash/faster-whisper-large-v3-turbo",
    }
    assert "HuggingFaceTB/SmolVLM-500M-Instruct" not in {
        model["repo_id"] for model in mid + big
    }
    assert "google/gemma-4-E4B-it" not in {
        model["repo_id"] for model in big
    }


def test_optional_e4b_setting_replaces_big_primary_model(
    tiny_registry, monkeypatch
):
    _, _, _, config_path = tiny_registry
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tiers"]["big"] = {
        "vision_model_repo_id": "google/gemma-4-E2B-it"
    }
    config["models"].append({
        "name": "Gemma E2B",
        "purpose": "vision",
        "repo_id": "google/gemma-4-E2B-it",
        "revision": "e2b",
        "tiers": ["mid", "big"],
        "status": "ready",
        "files": {"config.json": 1},
    })
    config["models"].append({
        "name": "Whisper small.en",
        "purpose": "voice",
        "repo_id": "test/voice",
        "revision": "voice",
        "tiers": ["big"],
        "status": "ready",
        "files": {"config.json": 1},
    })
    config["optional_models"] = [{
        "name": "Gemma E4B",
        "purpose": "vision",
        "repo_id": "google/gemma-4-E4B-it",
        "revision": "e4b",
        "for_tier": "big",
        "setting_value": "gemma-4-E4B-it",
        "status": "untested",
        "files": {"config.json": 1},
    }]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        app_paths, "get_big_vision_model_setting",
        lambda: "gemma-4-E4B-it",
    )

    required = model_downloader.models_for_tier("big")

    assert {model["repo_id"] for model in required} == {
        "test/tiny-model",
        "google/gemma-4-E4B-it",
        "test/voice",
    }


def test_extra_cached_models_are_not_removed(tiny_registry):
    _, _, models_dir, _ = tiny_registry
    extra = models_dir / "models--cached--extra"
    extra.mkdir()
    marker = extra / "keep.txt"
    marker.write_text("cached model", encoding="utf-8")

    model_downloader.missing_models_for_tier("small")

    assert marker.read_text(encoding="utf-8") == "cached model"


def test_embedder_sets_offline_only_after_finding_local_model(monkeypatch, tmp_path):
    from indexing import embedder
    from indexing import model_downloader

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(model_downloader._hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(embedder, "_model", None)
    monkeypatch.setattr(embedder, "find_model", lambda repo_id: None)
    with pytest.raises(RuntimeError, match="not fully installed"):
        embedder.get_model()
    assert "HF_HUB_OFFLINE" not in os.environ

    calls = []

    class FakeSentenceTransformer:
        def __init__(self, path, local_files_only):
            calls.append((path, local_files_only))

    monkeypatch.setattr(embedder, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embedder, "find_model", lambda repo_id: tmp_path / "model")
    embedder.get_model()

    assert calls == [(str(tmp_path / "model"), True)]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert model_downloader._hub_constants.HF_HUB_OFFLINE is True
