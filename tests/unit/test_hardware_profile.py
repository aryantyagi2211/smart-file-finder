import json
import subprocess
import sys
from types import SimpleNamespace

import app_paths
from app import hardware_profile


def _profile(ram_gb, vram_gb=None, free_disk_gb=200, gpu_name=None,
             architecture="AMD64"):
    return {
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
        "free_disk_gb": free_disk_gb,
        "gpu_name": gpu_name,
        "architecture": architecture,
    }


def test_selects_mid_for_gtx_1650_machine():
    tier, reason = hardware_profile.choose_tier(
        _profile(16, 4, gpu_name="NVIDIA GTX 1650")
    )

    assert tier == "mid"
    assert "NVIDIA GTX 1650" in reason
    assert "16 GB RAM" in reason


def test_selects_big_for_high_end_gpu_and_ram():
    tier, _ = hardware_profile.choose_tier(
        _profile(32, 16, gpu_name="NVIDIA RTX")
    )

    assert tier == "big"


def test_no_gpu_with_eight_or_thirty_two_gb_ram_selects_small():
    for ram in (8, 32):
        tier, _ = hardware_profile.choose_tier(_profile(ram))
        assert tier == "small"


def test_arm64_without_cuda_selects_small_and_notes_unimplemented_npu(
    monkeypatch,
):
    monkeypatch.setattr(hardware_profile.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(hardware_profile, "_detect_torch_cuda", lambda: None)
    monkeypatch.setattr(
        hardware_profile.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    profile = hardware_profile.detect_hardware()
    tier, reason = hardware_profile.choose_tier(profile)

    assert profile["architecture"] == "ARM64"
    assert "Snapdragon NPU path not implemented yet, using CPU/GPU tiers" in (
        profile["note"]
    )
    assert tier == "small"
    assert profile["note"] in reason


def test_low_disk_steps_down_one_tier_and_reports_reason():
    tier, reason = hardware_profile.choose_tier(
        _profile(32, 16, free_disk_gb=0.5, gpu_name="NVIDIA RTX")
    )

    assert tier == "mid"
    assert "stepped down to mid" in reason
    assert "1 GB minimum" in reason


def test_valid_override_wins_even_below_hardware_minimums(monkeypatch):
    warnings = []
    monkeypatch.setattr(hardware_profile, "log", warnings.append)

    tier, reason = hardware_profile.choose_tier(
        _profile(8), override="big"
    )

    assert tier == "big"
    assert "manual override" in reason
    assert any("WARNING" in message and "override" in message for message in warnings)


def test_invalid_override_warns_and_falls_back_to_auto(monkeypatch):
    warnings = []
    monkeypatch.setattr(hardware_profile, "log", warnings.append)

    tier, reason = hardware_profile.choose_tier(
        _profile(16, 4, gpu_name="NVIDIA GTX 1650"), override="turbo"
    )

    assert tier == "mid"
    assert "mid:" in reason
    assert any("Invalid performance_tier" in message for message in warnings)


def test_nvidia_smi_missing_timeout_or_garbage_leaves_gpu_none(
    monkeypatch,
):
    cases = (
        FileNotFoundError("missing"),
        subprocess.TimeoutExpired("nvidia-smi", 3),
        subprocess.CompletedProcess("nvidia-smi", 0, "garbage\n", ""),
    )
    for result in cases:
        if isinstance(result, subprocess.CompletedProcess):
            monkeypatch.setattr(
                hardware_profile.subprocess, "run", lambda *a, **k: result
            )
        else:
            def fail(*args, _error=result, **kwargs):
                raise _error

            monkeypatch.setattr(hardware_profile.subprocess, "run", fail)
        monkeypatch.setattr(
            hardware_profile, "_detect_torch_cuda", lambda: None
        )

        profile = hardware_profile.detect_hardware()

        assert profile["gpu_name"] is None
        assert profile["vram_gb"] is None


def test_nvidia_smi_parses_gpu_name_and_vram(monkeypatch):
    monkeypatch.setattr(
        hardware_profile.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "NVIDIA GTX 1650, 4096\n", ""
        ),
    )

    gpu = hardware_profile._detect_nvidia_smi()

    assert gpu == {"gpu_name": "NVIDIA GTX 1650", "vram_gb": 4.0}


def test_torch_cuda_fallback_is_bounded_and_parses_profile(monkeypatch):
    observed = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            args, 0,
            '{"gpu_name": "NVIDIA CUDA Device", "vram_gb": 8.0}',
            "",
        )

    monkeypatch.setattr(hardware_profile.subprocess, "run", fake_run)

    gpu = hardware_profile._detect_torch_cuda()

    assert gpu == {"gpu_name": "NVIDIA CUDA Device", "vram_gb": 8.0}
    assert observed["timeout"] == 5


def test_frozen_torch_cuda_fallback_uses_bundled_torch(monkeypatch):
    properties = SimpleNamespace(
        name="Bundled CUDA GPU",
        total_memory=4 * (1024 ** 3),
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: properties,
        )
    )
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        hardware_profile.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("frozen detection should not spawn the app executable")
        ),
    )

    assert hardware_profile._detect_torch_cuda() == {
        "gpu_name": "Bundled CUDA GPU",
        "vram_gb": 4.0,
    }


def test_missing_model_tier_config_is_created_with_defaults(
    tmp_path, monkeypatch
):
    project = tmp_path / "source"
    project.mkdir()
    monkeypatch.setattr(app_paths, "__file__", str(project / "app_paths.py"))
    monkeypatch.setattr(hardware_profile, "get_config_path", app_paths.get_config_path)

    config = hardware_profile.load_tier_config()

    assert set(config["tiers"]) == {"small", "mid", "big"}
    assert config["tiers"]["big"]["min_vram_gb"] == 12
    assert config["tiers"]["mid"]["vision_model"] == (
        "google/gemma-4-E2B-it (4-bit)"
    )
    assert (project / "config" / "model_tiers.json").is_file()


def test_settings_missing_file_defaults_to_auto(tmp_path, monkeypatch):
    project = tmp_path / "source"
    project.mkdir()
    monkeypatch.setattr(app_paths, "__file__", str(project / "app_paths.py"))
    monkeypatch.setattr(
        hardware_profile, "get_config_path", app_paths.get_config_path
    )

    assert hardware_profile.get_performance_tier_override() == "auto"
    settings = json.loads(
        (project / "config" / "settings.json").read_text(encoding="utf-8")
    )
    assert settings["performance_tier"] == "auto"
