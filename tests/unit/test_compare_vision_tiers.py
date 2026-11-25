import pytest

from app import hardware_profile
from eval import compare_vision_tiers


def test_e4b_comparison_is_opt_in_and_runs_as_big_tier():
    registry = compare_vision_tiers._read_registry()

    standard, _ = compare_vision_tiers._build_comparisons(registry)
    with_e4b, _ = compare_vision_tiers._build_comparisons(
        registry, include_e4b=True
    )

    assert not any(
        model["repo_id"] == "google/gemma-4-E4B-it"
        for model, _tier in standard
    )
    assert any(
        model["repo_id"] == "google/gemma-4-E4B-it" and tier == "big"
        for model, tier in with_e4b
    )


def test_e4b_comparison_requires_compatible_hardware(monkeypatch):
    monkeypatch.setattr(
        hardware_profile,
        "detect_hardware",
        lambda: {
            "ram_gb": 24,
            "vram_gb": 4,
            "free_disk_gb": 30,
        },
    )

    with pytest.raises(RuntimeError, match="12 GB NVIDIA VRAM required"):
        compare_vision_tiers._require_e4b_hardware()


def test_e4b_comparison_accepts_minimum_hardware(monkeypatch):
    monkeypatch.setattr(
        hardware_profile,
        "detect_hardware",
        lambda: {
            "ram_gb": 24,
            "vram_gb": 12,
            "free_disk_gb": 28,
        },
    )

    compare_vision_tiers._require_e4b_hardware()
