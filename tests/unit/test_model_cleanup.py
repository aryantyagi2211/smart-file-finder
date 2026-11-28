import json
import os
from pathlib import Path

from app import model_cleanup


def _entry(repo_id, purpose, tiers, revision="rev-1"):
    return {
        "repo_id": repo_id,
        "name": repo_id.rsplit("/", 1)[-1],
        "purpose": purpose,
        "tiers": tiers,
        "revision": revision,
        "status": "ready",
        "files": {"config.json": 4, "weights.bin": 8},
    }


def _registry():
    embedding = _entry("test/embedding", "embedding", ["small", "mid", "big"])
    vision = _entry("test/vision", "vision", ["small"])
    voice = _entry("test/voice", "voice", ["small"])
    candidate = _entry("test/voice-lab", "voice", [])
    superseded = _entry("test/old-voice", "voice", [])
    external = _entry("test/external-candidate", "voice", [])
    other_tier = _entry("test/other-tier", "voice", ["big"])
    return {
        "tiers": {
            "small": {"vision_model_repo_id": vision["repo_id"]},
            "mid": {"vision_model_repo_id": vision["repo_id"]},
            "big": {"vision_model_repo_id": vision["repo_id"]},
        },
        "models": [
            embedding, vision, voice, candidate, superseded, external,
            other_tier,
        ],
        "optional_models": [],
        "superseded": [{
            "repo_id": superseded["repo_id"],
            "reason": "Replaced by the current tier voice model.",
        }],
    }


def _write_model(root, model, source=None):
    repo_folder = Path(root) / (
        "models--" + model["repo_id"].replace("/", "--")
    )
    snapshot = repo_folder / "snapshots" / model["revision"]
    snapshot.mkdir(parents=True)
    for name, size in model["files"].items():
        (snapshot / name).write_bytes(b"x" * size)
    if source:
        (snapshot / model_cleanup._COMPLETE_MARKER).write_text(
            json.dumps({
                "repo_id": model["repo_id"],
                "revision": model["revision"],
                "download_source": source,
                "files": model["files"],
            }),
            encoding="utf-8",
        )
    return repo_folder


def _write_hf_snapshot(cache_root, model, revision, file_contents, refs=()):
    import hashlib

    repo_folder = Path(cache_root) / (
        "models--" + model["repo_id"].replace("/", "--")
    )
    blobs = repo_folder / "blobs"
    snapshot = repo_folder / "snapshots" / revision
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, content in file_contents.items():
        blob_name = hashlib.sha256(content).hexdigest()
        blob = blobs / blob_name
        blob.write_bytes(content)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(blob, link)
    for ref in refs:
        refs_dir = repo_folder / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / ref).write_text(revision, encoding="utf-8")
    return repo_folder, snapshot


def _model(
    tmp_path,
    status="CANDIDATE",
    underlying_status=None,
    protected=False,
    complete=True,
    is_external=False,
):
    return model_cleanup.ModelCache(
        repo_id="test/model",
        path=tmp_path / "models--test--model",
        cache_root=tmp_path,
        size_bytes=12,
        status=status,
        underlying_status=underlying_status or status,
        protected=protected,
        complete=complete,
        reason="test",
        is_external=is_external,
    )


def test_scan_classifies_in_use_candidate_superseded_and_external(tmp_path):
    registry = _registry()
    app_models = tmp_path / "app" / "models"
    shared_cache = tmp_path / "shared-hub"
    entries = {
        model["repo_id"]: model
        for model in registry["models"]
    }
    _write_model(app_models, entries["test/embedding"])
    _write_model(app_models, entries["test/vision"])
    _write_model(app_models, entries["test/voice"])
    _write_model(app_models, entries["test/voice-lab"])
    _write_model(app_models, entries["test/old-voice"])
    _write_model(shared_cache, entries["test/external-candidate"])

    scanned = model_cleanup.scan_model_folders(
        app_data_dir=tmp_path / "app",
        app_models_dir=app_models,
        hub_cache_dirs=[shared_cache],
        registry=registry,
        settings={"big_vision_model": "gemma-4-E2B-it"},
        current_tier="small",
    )
    by_repo = {model.repo_id: model for model in scanned}

    assert by_repo["test/embedding"].status == "IN USE"
    assert by_repo["test/voice-lab"].status == "CANDIDATE"
    assert by_repo["test/old-voice"].status == "SUPERSEDED"
    assert by_repo["test/external-candidate"].status == "EXTERNAL"
    assert by_repo["test/external-candidate"].underlying_status == "CANDIDATE"
    assert by_repo["test/external-candidate"].is_external


def test_voice_lab_download_source_overrides_superseded_classification(tmp_path):
    registry = _registry()
    app_models = tmp_path / "app" / "models"
    old_model = next(
        item for item in registry["models"]
        if item["repo_id"] == "test/old-voice"
    )
    _write_model(app_models, old_model, source="voice_lab")

    scanned = model_cleanup.scan_model_folders(
        app_data_dir=tmp_path / "app",
        app_models_dir=app_models,
        hub_cache_dirs=[],
        registry=registry,
        settings={},
        current_tier="small",
    )

    assert scanned[0].status == "CANDIDATE"


def test_dry_run_never_prompts_or_deletes(tmp_path):
    calls = []
    model = _model(tmp_path)
    output = []

    deleted = model_cleanup.cleanup_models(
        [model],
        apply=False,
        input_fn=lambda _prompt: calls.append("prompt"),
        output_fn=output.append,
        remove_model_folder=lambda item: calls.append(item),
    )

    assert deleted == []
    assert calls == []
    assert any("DRY RUN" in line for line in output)


def test_in_use_model_is_never_prompted_or_deleted(tmp_path):
    calls = []
    model = _model(
        tmp_path,
        status="IN USE",
        protected=True,
    )

    deleted = model_cleanup.cleanup_models(
        [model],
        apply=True,
        input_fn=lambda _prompt: calls.append("prompt"),
        output_fn=lambda _line: None,
        remove_model_folder=lambda item: calls.append(item),
    )

    assert deleted == []
    assert calls == []


def test_apply_confirms_each_model_individually(tmp_path):
    models = [
        _model(tmp_path / "one"),
        _model(tmp_path / "two", status="SUPERSEDED"),
    ]
    answers = iter(["y", "n"])
    prompts = []
    removed = []

    deleted = model_cleanup.cleanup_models(
        models,
        apply=True,
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        output_fn=lambda _line: None,
        remove_model_folder=removed.append,
    )

    assert len(prompts) == 2
    assert all("? [y/N]:" in prompt for prompt in prompts)
    assert deleted == [models[0]]
    assert removed == [models[0]]


def test_external_cache_warns_and_empty_confirmation_defaults_to_no(tmp_path):
    external = _model(
        tmp_path,
        status="EXTERNAL",
        underlying_status="CANDIDATE",
        is_external=True,
    )
    prompts = []
    output = []
    removed = []

    deleted = model_cleanup.cleanup_models(
        [external],
        apply=True,
        input_fn=lambda prompt: prompts.append(prompt) or "",
        output_fn=output.append,
        remove_model_folder=removed.append,
    )

    assert deleted == []
    assert removed == []
    assert len(prompts) == 1
    assert "[y/N]:" in prompts[0]
    assert any("WARNING" in line for line in output)
    assert any("other projects" in line for line in output)


def test_recordings_option_is_dry_run_then_requires_confirmation(tmp_path):
    recordings = model_cleanup.voice_lab_recordings_dir(tmp_path)
    recordings.mkdir(parents=True)
    (recordings / "phrase.wav").write_bytes(b"private recording")
    removals = []
    prompts = []
    output = []

    dry_run = model_cleanup.cleanup_voice_lab_recordings(
        app_data_dir=tmp_path,
        apply=False,
        input_fn=lambda prompt: prompts.append(prompt) or "y",
        output_fn=output.append,
        remove_recordings=lambda path, root: removals.append((path, root)),
    )
    assert not dry_run
    assert prompts == []
    assert removals == []
    assert any(str(recordings) in line for line in output)

    applied = model_cleanup.cleanup_voice_lab_recordings(
        app_data_dir=tmp_path,
        apply=True,
        input_fn=lambda prompt: prompts.append(prompt) or "y",
        output_fn=output.append,
        remove_recordings=lambda path, root: removals.append((path, root)),
    )
    assert applied
    assert len(prompts) == 1
    assert removals == [(recordings, tmp_path)]
    assert recordings.exists()


def test_missing_model_and_recording_folders_are_safe(tmp_path):
    scanned = model_cleanup.scan_model_folders(
        app_data_dir=tmp_path / "missing",
        app_models_dir=tmp_path / "missing" / "models",
        hub_cache_dirs=[tmp_path / "missing" / "hub"],
        registry=_registry(),
        settings={},
        current_tier="small",
    )
    output = []

    deleted = model_cleanup.cleanup_voice_lab_recordings(
        app_data_dir=tmp_path / "missing",
        apply=True,
        input_fn=lambda _prompt: (_ for _ in ()).throw(
            AssertionError("missing folders must not prompt")
        ),
        output_fn=output.append,
    )

    assert scanned == []
    assert not deleted
    assert any("No Voice Lab recordings folder" in line for line in output)


def test_cache_roots_include_app_hf_home_and_default_hub(tmp_path):
    app_models = tmp_path / "app" / "models"
    hf_home = tmp_path / "custom-hf"
    default_home = tmp_path / "user"
    roots = model_cleanup._cache_roots(
        app_models,
        env={
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(tmp_path / "explicit-hub"),
        },
        home=default_home,
    )

    assert roots == [
        app_models.resolve(),
        (tmp_path / "explicit-hub").resolve(),
        (hf_home / "hub").resolve(),
        hf_home.resolve(),
        (default_home / ".cache" / "huggingface" / "hub").resolve(),
        (default_home / ".cache" / "huggingface").resolve(),
    ]


def test_hf_cache_size_counts_blob_storage_behind_snapshot_symlinks(
    tmp_path,
):
    registry = _registry()
    model = next(
        entry for entry in registry["models"]
        if entry["repo_id"] == "test/voice-lab"
    )
    app_models = tmp_path / "app" / "models"
    _write_hf_snapshot(
        app_models,
        model,
        "rev-1",
        {"config.json": b"conf", "weights.bin": b"weights!"},
        refs=("main",),
    )

    scanned = model_cleanup.scan_model_folders(
        app_data_dir=tmp_path / "app",
        app_models_dir=app_models,
        hub_cache_dirs=[],
        registry=registry,
        settings={},
        current_tier="small",
    )

    assert len(scanned) == 1
    assert scanned[0].size_bytes == len(b"conf") + len(b"weights!")
    assert scanned[0].registry_size_bytes == 12
    assert not scanned[0].size_mismatch
    assert scanned[0].complete


def test_shared_hf_blob_is_counted_once_across_revisions(tmp_path):
    registry = _registry()
    model = next(
        entry for entry in registry["models"]
        if entry["repo_id"] == "test/voice-lab"
    )
    app_models = tmp_path / "app" / "models"
    shared = {"config.json": b"conf", "weights.bin": b"weights!"}
    _write_hf_snapshot(app_models, model, "rev-1", shared)
    _write_hf_snapshot(
        app_models,
        model,
        "rev-2",
        shared,
        refs=("main",),
    )
    cache_info = model_cleanup._scan_cache(app_models)
    repo_info = next(
        repo for repo in cache_info.repos
        if repo.repo_id == model["repo_id"]
    )

    assert repo_info.size_on_disk == len(b"conf") + len(b"weights!")
    assert len(repo_info.revisions) == 2
    assert cache_info.delete_revisions("rev-1").expected_freed_size == 0
    assert cache_info.delete_revisions(
        "rev-1", "rev-2"
    ).expected_freed_size == len(b"conf") + len(b"weights!")


def test_cache_delete_strategy_shows_estimate_and_verifies_freed_space(
    tmp_path,
):
    registry = _registry()
    model = next(
        entry for entry in registry["models"]
        if entry["repo_id"] == "test/voice-lab"
    )
    app_models = tmp_path / "app" / "models"
    _write_hf_snapshot(
        app_models,
        model,
        "rev-1",
        {"config.json": b"conf", "weights.bin": b"weights!"},
        refs=("main",),
    )
    scanned = model_cleanup.scan_model_folders(
        app_data_dir=tmp_path / "app",
        app_models_dir=app_models,
        hub_cache_dirs=[],
        registry=registry,
        settings={},
        current_tier="small",
    )
    output = []
    deleted = model_cleanup.cleanup_models(
        scanned,
        apply=True,
        input_fn=lambda _prompt: "y",
        output_fn=output.append,
    )

    assert len(deleted) == 1
    assert any("expects to free 12 B" in line for line in output)
    assert any("Verified after deletion" in line for line in output)
    assert any("decreased by 12 B" in line for line in output)
    after = model_cleanup._scan_cache(app_models)
    assert after is not None
    assert all(repo.repo_id != model["repo_id"] for repo in after.repos)


def test_cache_delete_rejects_linked_model_folder(tmp_path, monkeypatch):
    registry = _registry()
    model = next(
        entry for entry in registry["models"]
        if entry["repo_id"] == "test/voice-lab"
    )
    app_models = tmp_path / "app" / "models"
    _write_hf_snapshot(
        app_models,
        model,
        "rev-1",
        {"config.json": b"conf", "weights.bin": b"weights!"},
        refs=("main",),
    )
    scanned = model_cleanup.scan_model_folders(
        app_data_dir=tmp_path / "app",
        app_models_dir=app_models,
        hub_cache_dirs=[],
        registry=registry,
        settings={},
        current_tier="small",
    )
    monkeypatch.setattr(
        model_cleanup,
        "_is_reparse_point",
        lambda _path: True,
    )
    monkeypatch.setattr(
        model_cleanup,
        "_delete_strategy_for_revisions",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("linked cache folders must not be deleted")
        ),
    )

    assert model_cleanup._prepare_cache_delete(scanned[0]) is None


def test_incomplete_hf_revision_requires_confirmation_and_is_deletable(
    tmp_path,
):
    registry = _registry()
    model = next(
        entry for entry in registry["models"]
        if entry["repo_id"] == "test/voice-lab"
    )
    app_models = tmp_path / "app" / "models"
    _write_hf_snapshot(
        app_models,
        model,
        "rev-1",
        {"config.json": b"conf"},
        refs=("main",),
    )
    scanned = model_cleanup.scan_model_folders(
        app_data_dir=tmp_path / "app",
        app_models_dir=app_models,
        hub_cache_dirs=[],
        registry=registry,
        settings={},
        current_tier="small",
    )
    assert not scanned[0].complete
    output = []

    deleted = model_cleanup.cleanup_models(
        scanned,
        apply=True,
        input_fn=lambda prompt: output.append(prompt) or "y",
        output_fn=output.append,
    )

    assert len(deleted) == 1
    assert any("[INCOMPLETE]" in line for line in output)
    assert any("Hugging Face cache strategy expects" in line for line in output)
    assert any("Verified after deletion" in line for line in output)


def test_non_cache_size_deduplicates_hardlinks_and_internal_symlinks(
    tmp_path,
):
    model_folder = tmp_path / "model"
    model_folder.mkdir()
    blob = model_folder / "blob.bin"
    blob.write_bytes(b"shared-data")
    os.link(blob, model_folder / "hardlink.bin")
    os.symlink(blob, model_folder / "symlink.bin")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must-not-be-counted")
    os.symlink(outside, model_folder / "outside-link.bin")

    size, errors = model_cleanup._size_on_disk(model_folder)

    assert size == len(b"shared-data")
    assert errors == []


def test_missing_hf_cache_directory_does_not_crash(tmp_path):
    assert model_cleanup._scan_cache(tmp_path / "missing-cache") is None


def test_app_data_vision_model_gemma_e2b_is_always_protected(tmp_path):
    gemma = _entry("google/gemma-4-E2B-it", "vision", ["mid", "big"])
    embedding = _entry("test/embedding", "embedding", ["small"])
    vision = _entry("test/small-vision", "vision", ["small"])
    voice = _entry("test/small-voice", "voice", ["small"])
    registry = {
        "tiers": {
            "small": {"vision_model_repo_id": vision["repo_id"]},
        },
        "models": [embedding, vision, voice, gemma],
        "optional_models": [],
        "superseded": [{"repo_id": gemma["repo_id"]}],
    }
    app_models = tmp_path / "app" / "models"
    _write_model(app_models, gemma)

    scanned = model_cleanup.scan_model_folders(
        app_data_dir=tmp_path / "app",
        app_models_dir=app_models,
        hub_cache_dirs=[],
        registry=registry,
        settings={},
        current_tier="small",
    )

    assert scanned[0].status == "IN USE"
    assert scanned[0].protected


def test_selected_optional_model_is_in_use_for_cleanup():
    registry = _registry()
    optional = _entry("test/optional-vision", "vision", [])
    optional["setting_value"] = "gemma-4-E4B-it"
    registry["optional_models"] = [optional]

    required = model_cleanup._required_repo_ids(
        registry,
        {"big_vision_model": "gemma-4-E4B-it"},
        current_tier="small",
    )

    assert optional["repo_id"] in required


def test_safe_delete_targets_only_the_model_repo_folder(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    model_folder = cache_root / "models--test--model"
    model_folder.mkdir(parents=True)
    model = _model(cache_root)
    model.path = model_folder
    model.cache_root = cache_root
    removals = []
    monkeypatch.setattr(
        model_cleanup.shutil,
        "rmtree",
        lambda path: removals.append(Path(path)),
    )

    model_cleanup._safe_remove_model_folder(model)

    assert removals == [model_folder]
    assert model_folder.is_dir()


def test_incomplete_folder_requires_explicit_confirmation(tmp_path):
    incomplete = _model(tmp_path, complete=False)
    calls = []
    output = []

    declined = model_cleanup.cleanup_models(
        [incomplete],
        apply=True,
        input_fn=lambda _prompt: "n",
        output_fn=output.append,
        remove_model_folder=calls.append,
    )

    assert declined == []
    assert calls == []
    assert any("INCOMPLETE" in line for line in output)

    accepted = model_cleanup.cleanup_models(
        [incomplete],
        apply=True,
        input_fn=lambda _prompt: "y",
        output_fn=lambda _line: None,
        remove_model_folder=calls.append,
    )
    assert accepted == [incomplete]
    assert calls == [incomplete]


def test_cleanup_default_registry_lists_the_replaced_voice_models():
    registry = load_registry()
    superseded = {
        entry["repo_id"] for entry in registry["superseded"]
    }

    assert superseded == {
        "Systran/faster-whisper-tiny.en",
        "Systran/faster-whisper-base.en",
    }


def load_registry():
    from app_paths import load_model_registry

    return load_model_registry()
