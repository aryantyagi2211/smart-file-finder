import json
import sys

import app_paths
from agent.harness import load_limits
from indexing.backfill_index import load_indexed_folders


def _set_config_root(monkeypatch, root):
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_paths, "__file__", str(root / "app_paths.py"))
    monkeypatch.setattr(sys, "frozen", False, raising=False)


def test_get_config_path_uses_project_config_in_source_mode(tmp_path, monkeypatch):
    project_root = tmp_path / "source"
    _set_config_root(monkeypatch, project_root)
    (project_root / "config").mkdir()

    config_path = app_paths.get_config_path("settings.json")

    assert config_path == project_root / "config" / "settings.json"


def test_get_config_path_uses_config_next_to_executable_when_frozen(
    tmp_path, monkeypatch
):
    exe_dir = tmp_path / "installed"
    exe_dir.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "SmartFileFinder.exe"))

    config_path = app_paths.get_config_path("settings.json")

    assert config_path == exe_dir / "config" / "settings.json"


def test_missing_settings_file_is_created_with_empty_folder_defaults(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "source"
    _set_config_root(monkeypatch, project_root)
    messages = []
    import observability.logger

    monkeypatch.setattr(observability.logger, "log", messages.append)

    config_path = app_paths.get_config_path("settings.json")

    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "indexed_folders": [],
        "performance_tier": "auto",
        "big_vision_model": "gemma-4-E2B-it",
        "voice_enabled": True,
        "voice_key": "left alt",
        "voice_hold_ms": 400,
        "voice_max_seconds": 15,
        "voice_auto_search": True,
        "voice_preroll": True,
        "voice_tail_ms": 300,
        "voice_initial_prompt": (
            "Open, show, find. open corrective rag. show the roadmap."
        ),
        "voice_hotwords": "open show find",
        "voice_command_aliases": {
            "show": ["show", "sho", "so", "shoe", "shaw"],
            "open": ["open", "opan"],
            "find": ["find", "fine", "fined"],
        },
        "voice_model_override": "",
        "voice_slow_hint_shown": False,
    }
    assert any("settings.json" in message for message in messages)


def test_missing_harness_limits_file_is_created_with_defaults(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "source"
    _set_config_root(monkeypatch, project_root)
    messages = []
    import observability.logger

    monkeypatch.setattr(observability.logger, "log", messages.append)

    assert load_limits() == {
        "max_reformulations": 2,
        "timeout_seconds": 5,
        "confidence_threshold": 0.55,
        "ambiguity_margin": 0.07,
        "options_floor": 0.48,
        "min_option_score": 0.5,
        "option_score_window": 0.03,
        "max_options": 3,
    }
    assert app_paths.get_config_path("harness_limits.json").exists()
    assert any("harness_limits.json" in message for message in messages)


def test_config_files_are_found_from_a_different_working_directory(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "source"
    _set_config_root(monkeypatch, project_root)
    config_dir = project_root / "config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps({"indexed_folders": [str(tmp_path / "extra")]}),
        encoding="utf-8",
    )
    (config_dir / "harness_limits.json").write_text(
        json.dumps({
            "max_reformulations": 4,
            "timeout_seconds": 9,
            "confidence_threshold": 0.6,
        }),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert load_limits() == {
        "max_reformulations": 4,
        "timeout_seconds": 9,
        "confidence_threshold": 0.6,
        "ambiguity_margin": 0.07,
        "options_floor": 0.48,
        "min_option_score": 0.5,
        "option_score_window": 0.03,
        "max_options": 3,
    }
    assert app_paths.normalize_path(tmp_path / "extra") in load_indexed_folders()


def test_base_data_directory_can_be_isolated_for_eval_and_tests(
    tmp_path, monkeypatch
):
    isolated = tmp_path / "isolated-data"
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("SMART_FILE_FINDER_EVAL_DATA_DIR", str(isolated))

    assert app_paths.get_base_dir() == isolated
    assert isolated.is_dir()
    assert app_paths.get_models_dir() == (
        local_app_data / app_paths.APP_FOLDER_NAME / "models"
    )
