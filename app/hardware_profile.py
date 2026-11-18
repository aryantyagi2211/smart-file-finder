"""Hardware detection and rule-based performance-tier selection."""

import csv
import io
import json
import os
import platform
import subprocess
import sys

from app_paths import get_base_dir, get_config_path
from observability.logger import log

_current_tier = "small"
_TIER_ORDER = ("small", "mid", "big")
_ARM64_NOTE = (
    "Snapdragon NPU path not implemented yet, using CPU/GPU tiers"
)


def _machine_architecture():
    machine = platform.machine().strip().casefold()
    if machine in ("amd64", "x86_64", "x64"):
        return "AMD64"
    if machine in ("arm64", "aarch64"):
        return "ARM64"
    return platform.machine() or "Unknown"


def _detect_nvidia_smi():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=3,
        check=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if not rows or len(rows[0]) < 2:
        raise ValueError("nvidia-smi returned no parseable GPU rows")
    name = rows[0][0].strip()
    memory_mb = float(rows[0][1].strip())
    if not name or memory_mb <= 0:
        raise ValueError("nvidia-smi returned invalid GPU information")
    return {"gpu_name": name, "vram_gb": memory_mb / 1024}


def _detect_torch_cuda():
    try:
        if getattr(sys, "frozen", False):
            import torch

            if not torch.cuda.is_available():
                return None
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "gpu_name": properties.name,
                "vram_gb": properties.total_memory / (1024 ** 3),
            }
        else:
            probe = (
                "import json, torch; "
                "available = torch.cuda.is_available(); "
                "props = torch.cuda.get_device_properties(0) "
                "if available else None; "
                "print(json.dumps(None if props is None else "
                "{'gpu_name': props.name, 'vram_gb': "
                "props.total_memory / (1024 ** 3)}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            gpu = json.loads(result.stdout)
        if not isinstance(gpu, dict):
            return None
        name = str(gpu.get("gpu_name", "")).strip()
        memory = float(gpu.get("vram_gb", 0))
        if not name or memory <= 0:
            return None
        return {"gpu_name": name, "vram_gb": memory}
    except Exception:
        return None


def detect_hardware():
    """Return a best-effort local hardware profile without unbounded waits."""
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        logical_cores = psutil.cpu_count(logical=True)
    except Exception as error:
        log(f"WARNING: Could not read RAM or CPU core count: {error}")
        ram_gb = None
        logical_cores = None

    try:
        import psutil

        free_disk_gb = psutil.disk_usage(str(get_base_dir())).free / (1024 ** 3)
    except Exception as error:
        log(f"WARNING: Could not read app-data disk space: {error}")
        free_disk_gb = None

    cpu_name = (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.machine()
        or "Unknown CPU"
    )
    architecture = _machine_architecture()
    profile = {
        "ram_gb": ram_gb,
        "free_disk_gb": free_disk_gb,
        "cpu_name": cpu_name,
        "logical_cores": logical_cores,
        "architecture": architecture,
        "os_version": platform.platform(),
        "gpu_name": None,
        "vram_gb": None,
    }

    try:
        gpu = _detect_nvidia_smi()
    except Exception as error:
        log(f"nvidia-smi unavailable or failed; trying torch CUDA: {error}")
        gpu = _detect_torch_cuda()
    if gpu is not None:
        profile.update(gpu)
    if architecture == "ARM64":
        profile["note"] = _ARM64_NOTE
    return profile


def load_tier_config(config_path=None):
    """Load editable model-tier thresholds through the app config resolver."""
    path = config_path or get_config_path("model_tiers.json")
    with open(path, encoding="utf-8-sig") as config_file:
        config = json.load(config_file)
    if not isinstance(config.get("tiers"), dict):
        raise ValueError(f"Invalid model tier configuration in {path}")
    return config


def _profile_value(profile, name):
    value = profile.get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _meets_tier(profile, requirements):
    return (
        _profile_value(profile, "ram_gb")
        >= float(requirements.get("min_ram_gb", 0))
        and _profile_value(profile, "vram_gb")
        >= float(requirements.get("min_vram_gb", 0))
    )


def _hardware_summary(profile):
    gpu_name = profile.get("gpu_name")
    vram = profile.get("vram_gb")
    ram = profile.get("ram_gb")
    parts = []
    if gpu_name:
        try:
            parts.append(f"{gpu_name} {float(vram):g} GB VRAM")
        except (TypeError, ValueError):
            parts.append(str(gpu_name))
    else:
        parts.append("no NVIDIA GPU")
    if ram is not None:
        parts.append(f"{float(ram):g} GB RAM")
    return ", ".join(parts)


def choose_tier(profile, override="auto", tiers=None):
    """Choose a tier from a supplied profile; no hardware probing is performed."""
    tier_config = tiers if tiers is not None else load_tier_config()
    tier_definitions = tier_config.get("tiers", tier_config)
    normalized_override = str(override or "auto").strip().casefold()
    if normalized_override not in ("auto", *_TIER_ORDER):
        log(
            f"WARNING: Invalid performance_tier {override!r}; "
            "falling back to auto"
        )
        normalized_override = "auto"

    summary = _hardware_summary(profile)
    if normalized_override != "auto":
        selected = normalized_override
        requirements = tier_definitions[selected]
        unmet = []
        if not _meets_tier(profile, requirements):
            if _profile_value(profile, "ram_gb") < float(
                requirements.get("min_ram_gb", 0)
            ):
                unmet.append(
                    f"{requirements.get('min_ram_gb', 0)} GB RAM required"
                )
            if _profile_value(profile, "vram_gb") < float(
                requirements.get("min_vram_gb", 0)
            ):
                unmet.append(
                    f"{requirements.get('min_vram_gb', 0)} GB VRAM required"
                )
        if _profile_value(profile, "free_disk_gb") < float(
            requirements.get("min_free_disk_gb", 0)
        ):
            unmet.append(
                f"{requirements.get('min_free_disk_gb', 0)} GB free disk required"
            )
        if unmet:
            log(
                f"WARNING: Performance tier override {selected!r} is above "
                f"hardware minimums: {', '.join(unmet)}"
            )
        reason = f"{selected}: manual override; {summary}"
        if unmet:
            reason += f" (below minimums: {', '.join(unmet)})"
        return selected, reason

    selected = "small"
    for candidate in reversed(_TIER_ORDER):
        if _meets_tier(profile, tier_definitions[candidate]):
            selected = candidate
            break

    reason = f"{selected}: {summary}"
    free_disk = _profile_value(profile, "free_disk_gb")
    disk_minimum = float(
        tier_definitions[selected].get("min_free_disk_gb", 0)
    )
    if selected != "small" and free_disk < disk_minimum:
        previous_tier = _TIER_ORDER[_TIER_ORDER.index(selected) - 1]
        reason = (
            f"{selected} stepped down to {previous_tier}: "
            f"{free_disk:g} GB free disk is below its "
            f"{disk_minimum:g} GB minimum; {summary}"
        )
        selected = previous_tier
    if profile.get("architecture") == "ARM64":
        reason += f"; {_ARM64_NOTE}"
    return selected, reason


def get_performance_tier_override(settings_path=None):
    """Read the user's tier override; missing or invalid settings use auto."""
    path = settings_path or get_config_path("settings.json")
    try:
        with open(path, encoding="utf-8-sig") as settings_file:
            settings = json.load(settings_file)
    except (OSError, json.JSONDecodeError) as error:
        log(f"WARNING: Could not read performance tier setting: {error}")
        return "auto"
    return settings.get("performance_tier", "auto")


def set_current_tier(tier):
    """Publish the startup-selected tier for other app modules."""
    global _current_tier
    if tier not in _TIER_ORDER:
        raise ValueError(f"Unsupported performance tier: {tier}")
    _current_tier = tier


def get_current_tier():
    """Return the tier selected at startup; defaults to small during detection."""
    return _current_tier


def _print_profile():
    profile = detect_hardware()
    override = get_performance_tier_override()
    tier, reason = choose_tier(profile, override)
    print(json.dumps(
        {"hardware": profile, "performance_tier": tier, "reason": reason},
        indent=2,
    ))


if __name__ == "__main__":
    _print_profile()
