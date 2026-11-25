"""Safely inventory and optionally remove unused local Hugging Face models."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import stat

from app.hardware_profile import choose_tier, detect_hardware
from app_paths import (
    get_config_path,
    get_hardware_profile_path,
    get_models_dir,
    get_voice_lab_dir,
    load_model_registry,
)


_TIER_NAMES = ("small", "mid", "big")
_CANDIDATE_SOURCES = {"voice_lab", "vision_comparison"}
_NEVER_DELETE_REPO_IDS = {"google/gemma-4-E2B-it"}
_COMPLETE_MARKER = ".smart-file-finder-complete.json"


@dataclass
class ModelCache:
    repo_id: str
    path: Path
    cache_root: Path
    size_bytes: int
    status: str
    underlying_status: str
    protected: bool
    complete: bool
    reason: str
    is_external: bool
    scan_error: str = ""


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default


def load_settings(settings_path=None):
    path = settings_path or get_config_path("settings.json")
    try:
        return _read_json(path, {})
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read settings from {path}: {error}") from error


def _cache_roots(app_models_dir, env=None, home=None):
    env = os.environ if env is None else env
    home = Path.home() if home is None else Path(home)
    hf_home = Path(
        env.get("HF_HOME", home / ".cache" / "huggingface")
    ).expanduser()
    default_hf_home = home / ".cache" / "huggingface"
    roots = [
        Path(app_models_dir),
        Path(env["HF_HUB_CACHE"]).expanduser()
        if env.get("HF_HUB_CACHE")
        else None,
        hf_home / "hub",
        hf_home,
        default_hf_home / "hub",
        default_hf_home,
    ]
    unique = []
    seen = set()
    for root in roots:
        if root is None:
            continue
        resolved = root.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            unique.append(resolved)
            seen.add(key)
    return unique


def _repo_id_from_folder(folder):
    prefix = "models--"
    if not folder.name.startswith(prefix):
        return None
    encoded = folder.name[len(prefix):]
    parts = encoded.split("--")
    if len(parts) < 2 or any(not part for part in parts):
        return None
    return "/".join(parts)


def _all_registry_models(registry):
    return list(registry.get("models", [])) + list(
        registry.get("optional_models", [])
    )


def _registry_models_by_repo(registry):
    result = {}
    for model in _all_registry_models(registry):
        repo_id = model.get("repo_id")
        if repo_id:
            result.setdefault(repo_id, []).append(model)
    return result


def _valid_relative_file(name):
    relative = Path(str(name))
    return (
        bool(name)
        and not relative.is_absolute()
        and ".." not in relative.parts
    )


def _is_reparse_point(path):
    try:
        info = Path(path).lstat()
    except OSError:
        return True
    attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return Path(path).is_symlink() or bool(
        getattr(info, "st_file_attributes", 0) & attribute
    )


def _contains_reparse_directory(directory):
    found = False

    def record_walk_error(_error):
        nonlocal found
        found = True

    for root, directory_names, _file_names in os.walk(
        directory,
        topdown=True,
        followlinks=False,
        onerror=record_walk_error,
    ):
        safe_names = []
        for name in directory_names:
            if _is_reparse_point(Path(root) / name):
                found = True
            else:
                safe_names.append(name)
        directory_names[:] = safe_names
    return found


def _marker_data(snapshot, repo_id):
    marker_path = snapshot / _COMPLETE_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        marker.get("repo_id") != repo_id
        or marker.get("revision") != snapshot.name
    ):
        return None
    return marker


def _expected_files(snapshot, repo_id, model):
    marker = _marker_data(snapshot, repo_id)
    if marker:
        files = marker.get("files")
        if isinstance(files, dict) and files:
            return files
    if model and model.get("revision") == snapshot.name:
        files = model.get("files")
        if isinstance(files, dict) and files:
            return files
    return None


def _snapshot_is_complete(snapshot, repo_id, model):
    if _is_reparse_point(snapshot) or not snapshot.is_dir():
        return False
    files = _expected_files(snapshot, repo_id, model)
    if not files:
        return False
    for name, expected_size in files.items():
        if not _valid_relative_file(name):
            return False
        try:
            path = snapshot / name
            if not path.is_file() or path.stat().st_size != int(expected_size):
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True


def _complete_snapshot_dirs(repo_folder, repo_id, model):
    snapshots = repo_folder / "snapshots"
    if not snapshots.is_dir():
        return []
    try:
        candidates = sorted(snapshots.iterdir())
    except OSError:
        return []
    return [
        snapshot for snapshot in candidates
        if _snapshot_is_complete(snapshot, repo_id, model)
    ]


def _cache_sources(repo_folder, repo_id):
    snapshots = repo_folder / "snapshots"
    if not snapshots.is_dir():
        return []
    sources = []
    try:
        snapshot_dirs = sorted(snapshots.iterdir())
    except OSError:
        return []
    for snapshot in snapshot_dirs:
        marker = _marker_data(snapshot, repo_id)
        sources.append(marker.get("download_source") if marker else None)
    return sources


def _is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _size_on_disk(directory):
    total = 0
    identities = set()
    errors = []

    def record_walk_error(error):
        errors.append(str(error))

    for root, directory_names, file_names in os.walk(
        directory,
        topdown=True,
        followlinks=False,
        onerror=record_walk_error,
    ):
        directory_names[:] = [
            name for name in directory_names
            if not _is_reparse_point(Path(root) / name)
        ]
        for file_name in file_names:
            path = Path(root) / file_name
            try:
                if _is_reparse_point(path) and not _is_relative_to(
                    path.resolve(), directory
                ):
                    continue
                stat = path.stat()
            except OSError as error:
                errors.append(f"{path}: {error}")
                continue
            identity = (
                (stat.st_dev, stat.st_ino)
                if stat.st_ino
                else os.path.normcase(str(path.resolve()))
            )
            if identity not in identities:
                identities.add(identity)
                total += stat.st_size
    return total, errors


def _superseded_entries(registry):
    result = {}
    for entry in registry.get("superseded") or []:
        if isinstance(entry, str):
            result[entry] = {}
        elif isinstance(entry, dict) and entry.get("repo_id"):
            result[entry["repo_id"]] = entry
    return result


def _required_repo_ids(registry, settings, current_tier):
    all_models = _all_registry_models(registry)
    if current_tier not in _TIER_NAMES:
        result = {
            model["repo_id"]
            for model in all_models
            if model.get("repo_id")
            and set(model.get("tiers") or []) & set(_TIER_NAMES)
        }
    else:
        from app_paths import (
            get_required_models_for_tier,
            get_voice_model_for_tier,
        )

        selected_vision = settings.get("big_vision_model", "gemma-4-E2B-it")
        required = get_required_models_for_tier(
            current_tier,
            registry,
            big_vision_model=selected_vision,
        )
        result = {
            model["repo_id"] for model in required if model.get("repo_id")
        }

    from app_paths import get_voice_model_for_tier

    voice_override = settings.get("voice_model_override")
    if voice_override:
        selected_voice = get_voice_model_for_tier(
            current_tier or "small",
            registry,
            model_name=voice_override,
        )
        if selected_voice is not None:
            result.add(selected_voice["repo_id"])

    selected_vision = settings.get("big_vision_model", "gemma-4-E2B-it")
    if selected_vision == "gemma-4-E4B-it":
        result.update(
            model["repo_id"]
            for model in registry.get("optional_models", [])
            if model.get("setting_value") == selected_vision
            and model.get("repo_id")
        )
    result.update(_NEVER_DELETE_REPO_IDS)
    return result


def _classify_model(
    repo_id,
    model,
    current_tier,
    required_repo_ids,
    superseded,
    sources,
    is_external,
):
    if repo_id in required_repo_ids or repo_id in _NEVER_DELETE_REPO_IDS:
        underlying = "IN USE"
        reason = (
            f"required by the current {current_tier} tier"
            if current_tier in _TIER_NAMES and repo_id in required_repo_ids
            else "protected to preserve a registered performance-tier model"
        )
    elif sources and all(source in _CANDIDATE_SOURCES for source in sources):
        underlying = "CANDIDATE"
        reason = "downloaded for Voice Lab or vision comparison"
    elif repo_id in superseded:
        underlying = "SUPERSEDED"
        reason = superseded[repo_id].get(
            "reason", "listed as superseded in the model registry"
        )
    elif model and (
        model.get("benchmark_only")
        or model.get("candidate_tier")
        or (
            model.get("purpose") == "voice"
            and not model.get("tiers")
        )
    ):
        underlying = "CANDIDATE"
        reason = "registered as a Voice Lab or vision comparison candidate"
    elif model and model.get("tiers"):
        registered_tiers = [
            tier for tier in (model.get("tiers") or [])
            if tier in _TIER_NAMES
        ]
        if current_tier not in _TIER_NAMES or current_tier in registered_tiers:
            underlying = "IN USE"
            reason = "required by a registered performance tier"
        else:
            underlying = "SUPERSEDED"
            tier_names = ", ".join(registered_tiers)
            reason = (
                f"not required by the current {current_tier} tier; "
                f"registry assigns it to {tier_names or 'no current tier'}"
            )
    else:
        underlying = "SUPERSEDED"
        reason = "not assigned to an active tier or registered as a candidate"

    protected = underlying == "IN USE"
    status = "EXTERNAL" if is_external else underlying
    if is_external:
        reason = (
            "shared Hugging Face cache; other projects may use it. "
            + reason
        )
    return status, underlying, protected, reason


def scan_model_folders(
    app_data_dir=None,
    app_models_dir=None,
    hub_cache_dirs=None,
    registry=None,
    settings=None,
    current_tier=None,
):
    app_models_dir = Path(
        app_models_dir or get_models_dir()
    ).resolve()
    app_data_dir = Path(
        app_data_dir or app_models_dir.parent
    ).resolve()
    registry = load_model_registry() if registry is None else registry
    settings = {} if settings is None else settings
    model_map = _registry_models_by_repo(registry)
    superseded = _superseded_entries(registry)
    required_repo_ids = _required_repo_ids(
        registry, settings, current_tier
    )
    roots = (
        [app_models_dir, *(Path(root).resolve() for root in hub_cache_dirs)]
        if hub_cache_dirs is not None
        else _cache_roots(app_models_dir)
    )
    cache_roots = []
    seen_roots = set()
    for root in roots:
        resolved = Path(root).resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen_roots:
            cache_roots.append(resolved)
            seen_roots.add(key)

    found = []
    seen_locations = set()
    for cache_root in cache_roots:
        if not cache_root.is_dir():
            continue
        try:
            folders = sorted(cache_root.iterdir())
        except OSError as error:
            print(f"WARNING: Cannot inspect Hugging Face cache {cache_root}: {error}")
            continue
        for folder in folders:
            repo_id = _repo_id_from_folder(folder)
            if repo_id is None or not folder.is_dir():
                continue
            location_key = os.path.normcase(str(folder.resolve()))
            if location_key in seen_locations:
                continue
            seen_locations.add(location_key)

            model = next(
                iter(model_map.get(repo_id, [])),
                None,
            )
            linked_folder = _is_reparse_point(folder)
            complete_snapshots = (
                []
                if linked_folder
                else _complete_snapshot_dirs(folder, repo_id, model)
            )
            sources = [] if linked_folder else _cache_sources(folder, repo_id)
            external = not _is_relative_to(folder, app_models_dir)
            status, underlying, protected, reason = _classify_model(
                repo_id,
                model,
                current_tier,
                required_repo_ids,
                superseded,
                sources,
                external,
            )
            if linked_folder:
                size_bytes, errors = 0, ["cache folder is a link"]
                has_reparse_directory = False
            else:
                size_bytes, errors = _size_on_disk(folder)
                has_reparse_directory = _contains_reparse_directory(folder)
            complete = (
                not linked_folder
                and bool(complete_snapshots)
                and not has_reparse_directory
            )
            if not complete:
                reason += "; no complete pinned snapshot could be verified"
            if has_reparse_directory:
                reason += "; linked subdirectories prevent safe folder removal"
            if errors:
                reason += "; some files could not be sized"
            found.append(ModelCache(
                repo_id=repo_id,
                path=folder,
                cache_root=cache_root,
                size_bytes=size_bytes,
                status=status,
                underlying_status=underlying,
                protected=protected,
                complete=complete,
                reason=reason,
                is_external=external,
                scan_error="; ".join(errors),
            ))
    return sorted(found, key=lambda model: (
        model.repo_id.casefold(),
        os.path.normcase(str(model.path)),
    ))


def resolve_current_tier(
    settings=None,
    profile_path=None,
    detect_profile=None,
    output_fn=print,
):
    settings = settings or {}
    profile_path = Path(
        profile_path
        or get_hardware_profile_path()
    )
    profile = {}
    try:
        report = _read_json(profile_path, {})
        profile = report.get("hardware", {}) if isinstance(report, dict) else {}
        if not isinstance(profile, dict) or not profile:
            profile = (detect_profile or detect_hardware)()
        tier, reason = choose_tier(
            profile,
            settings.get("performance_tier", "auto"),
        )
        return tier, reason
    except Exception as error:
        output_fn(
            "WARNING: Could not determine the current performance tier "
            f"({error}); all registered tier models will be protected."
        )
        return None, "tier detection failed"


def _format_size(byte_count):
    if byte_count >= 1_000_000_000:
        return f"{byte_count / 1_000_000_000:.2f} GB"
    return f"{byte_count / 1_000_000:.1f} MB"


def _print_model(model, output_fn=print):
    protected = " [protected]" if model.protected else ""
    complete = "complete" if model.complete else "incomplete"
    output_fn(
        f"{model.status}{protected} | {model.repo_id} | "
        f"{_format_size(model.size_bytes)} | {model.path} | "
        f"{complete} | {model.reason}"
    )
    if model.is_external:
        output_fn(
            "  WARNING: This is a shared Hugging Face cache; other projects "
            "on this PC may use these files."
        )
    if model.scan_error:
        output_fn(f"  WARNING: Could not size all cache files: {model.scan_error}")


def _confirm(prompt, input_fn):
    answer = input_fn(prompt)
    return str(answer or "").strip().casefold() in {"y", "yes"}


def _safe_remove_model_folder(model):
    if (
        model.status == "IN USE"
        or model.underlying_status == "IN USE"
        or model.protected
    ):
        raise ValueError(f"Refusing to delete protected model: {model.repo_id}")
    if not model.complete:
        raise ValueError(
            f"Refusing to delete incomplete model cache: {model.path}"
        )
    folder = Path(model.path)
    cache_root = Path(model.cache_root).resolve()
    if (
        _is_reparse_point(folder)
        or _contains_reparse_directory(folder)
        or folder.name != "models--" + model.repo_id.replace("/", "--")
        or folder.parent.resolve() != cache_root
    ):
        raise ValueError(
            f"Refusing to delete a path that is not this model's cache folder: {folder}"
        )
    shutil.rmtree(folder)


def cleanup_models(
    models,
    apply=False,
    input_fn=input,
    output_fn=print,
    remove_model_folder=None,
):
    remove_model_folder = remove_model_folder or _safe_remove_model_folder
    for model in models:
        _print_model(model, output_fn)
    if not apply:
        output_fn("DRY RUN: no model folders were deleted.")
        return []

    deleted = []
    for model in models:
        if (
            model.status == "IN USE"
            or model.underlying_status == "IN USE"
            or model.protected
        ):
            output_fn(f"Keeping {model.repo_id}: IN USE models cannot be deleted.")
            continue
        if not model.complete:
            output_fn(
                f"Keeping {model.repo_id}: the cache is incomplete or could "
                "not be verified."
            )
            continue
        if model.is_external:
            output_fn(
                "WARNING: Removing this shared cache entry may affect other "
                "projects on this PC. The default response is no."
            )
        prompt = (
            f"Delete {model.repo_id} ({_format_size(model.size_bytes)}) "
            f"at {model.path}? [y/N]: "
        )
        if not _confirm(prompt, input_fn):
            output_fn(f"Kept {model.repo_id}.")
            continue
        remove_model_folder(model)
        deleted.append(model)
        output_fn(f"Deleted {model.repo_id}.")
    if not deleted:
        output_fn("No model folders were deleted.")
    return deleted


def voice_lab_recordings_dir(app_data_dir=None):
    app_data_dir = Path(
        app_data_dir or Path(get_voice_lab_dir()).parent
    )
    return app_data_dir / "voice_lab" / "recordings"


def _safe_remove_recordings_folder(path, app_data_dir):
    folder = Path(path)
    expected_parent = Path(app_data_dir) / "voice_lab"
    if (
        _is_reparse_point(folder)
        or _contains_reparse_directory(folder)
        or folder.name != "recordings"
        or folder.parent.resolve() != expected_parent.resolve()
    ):
        raise ValueError(
            f"Refusing to delete a path outside Voice Lab recordings: {folder}"
        )
    shutil.rmtree(folder)


def cleanup_voice_lab_recordings(
    app_data_dir=None,
    apply=False,
    input_fn=input,
    output_fn=print,
    remove_recordings=None,
):
    app_data_dir = Path(
        app_data_dir or Path(get_voice_lab_dir()).parent
    )
    recordings_dir = voice_lab_recordings_dir(app_data_dir)
    if not recordings_dir.is_dir():
        output_fn(f"No Voice Lab recordings folder found at {recordings_dir}.")
        return False
    if _is_reparse_point(recordings_dir):
        output_fn(
            f"Keeping Voice Lab recordings: refusing linked folder "
            f"{recordings_dir}."
        )
        return False
    size_bytes, errors = _size_on_disk(recordings_dir)
    output_fn(
        f"Voice Lab recordings: {_format_size(size_bytes)} at {recordings_dir}"
    )
    if _contains_reparse_directory(recordings_dir):
        output_fn(
            "Keeping recordings: linked subdirectories prevent safe removal."
        )
        return False
    if errors:
        output_fn("WARNING: Some recording files could not be sized.")
    if not apply:
        output_fn(
            "DRY RUN: no recordings were deleted. Re-run with --apply "
            "to confirm removal."
        )
        return False
    if not _confirm(
        f"Delete the Voice Lab recordings folder "
        f"({_format_size(size_bytes)}) at {recordings_dir}? [y/N]: ",
        input_fn,
    ):
        output_fn("Kept Voice Lab recordings.")
        return False
    remove_recordings = remove_recordings or _safe_remove_recordings_folder
    remove_recordings(recordings_dir, app_data_dir)
    output_fn("Deleted Voice Lab recordings.")
    return True


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inventory and safely clean unused model cache folders."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="ask for explicit confirmation before deleting eligible folders",
    )
    parser.add_argument(
        "--voice-lab-recordings",
        action="store_true",
        help="inspect and optionally remove local Voice Lab recordings",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.voice_lab_recordings:
        cleanup_voice_lab_recordings(
            apply=args.apply,
        )
        return 0

    app_models_dir = Path(get_models_dir())
    app_data_dir = app_models_dir.parent
    registry = load_model_registry()
    settings = load_settings()
    tier, reason = resolve_current_tier(settings)
    tier_text = tier or "unknown"
    print(f"Current performance tier: {tier_text} ({reason})")
    models = scan_model_folders(
        app_data_dir=app_data_dir,
        app_models_dir=app_models_dir,
        registry=registry,
        settings=settings,
        current_tier=tier,
    )
    if not models:
        print("No downloaded model folders found.")
    cleanup_models(models, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
