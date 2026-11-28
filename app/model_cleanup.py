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
_TIER_LABELS = {
    "small": "Low",
    "mid": "Mid",
    "big": "High",
}
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
    registry_size_bytes: int = 0
    size_mismatch: bool = False
    revision_hashes: tuple = ()
    incomplete_revisions: tuple = ()
    cache_layout: bool = True
    model_name: str = ""
    purpose: str = ""
    incomplete_size_bytes: int = 0


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

    visited_directories = set()
    for root, directory_names, file_names in os.walk(
        directory,
        topdown=True,
        followlinks=True,
        onerror=record_walk_error,
    ):
        root_path = Path(root)
        try:
            root_stat = root_path.stat()
            root_id = (root_stat.st_dev, root_stat.st_ino)
        except OSError as error:
            errors.append(f"{root_path}: {error}")
            directory_names[:] = []
            continue
        if root_id in visited_directories:
            directory_names[:] = []
            continue
        visited_directories.add(root_id)

        safe_directories = []
        for name in directory_names:
            candidate = root_path / name
            try:
                resolved = candidate.resolve()
                resolved.relative_to(Path(directory).resolve())
                candidate_stat = candidate.stat()
            except (OSError, ValueError):
                if _is_reparse_point(candidate):
                    continue
                safe_directories.append(name)
                continue
            if not stat.S_ISDIR(candidate_stat.st_mode):
                continue
            safe_directories.append(name)
        directory_names[:] = safe_directories

        for file_name in file_names:
            path = root_path / file_name
            try:
                resolved = path.resolve()
                resolved.relative_to(Path(directory).resolve())
                file_stat = path.stat()
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
            except OSError as error:
                errors.append(f"{path}: {error}")
                continue
            except ValueError:
                if _is_reparse_point(path):
                    continue
                errors.append(f"{path}: resolved outside model folder")
                continue
            identity = (
                (file_stat.st_dev, file_stat.st_ino)
                if file_stat.st_ino
                else os.path.normcase(str(resolved))
            )
            if identity not in identities:
                identities.add(identity)
                total += file_stat.st_size
    return total, errors


def _scan_cache(cache_root):
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        return scan_cache_dir(cache_dir=cache_root)
    except CacheNotFound:
        return None


def _cache_info_total_size(cache_info):
    if cache_info is None:
        return 0
    return sum(int(repo.size_on_disk) for repo in cache_info.repos)


def _registry_download_size(model):
    if not model:
        return 0
    return int(
        model.get("download_size_bytes")
        or sum(model.get("files", {}).values())
    )


def _delete_strategy_for_revisions(model, revision_hashes):
    cache_info = _scan_cache(model.cache_root)
    repo_info = _find_repo_info(cache_info, model.repo_id)
    if repo_info is None:
        return None, None, None
    available = {
        revision.commit_hash for revision in repo_info.revisions
    }
    selected = tuple(
        revision_hash for revision_hash in revision_hashes
        if revision_hash in available
    )
    if not selected:
        return cache_info, repo_info, None
    strategy = cache_info.delete_revisions(*selected)
    target_repo_path = os.path.normcase(
        str(Path(repo_info.repo_path).resolve())
    )
    strategy_repo_paths = {
        os.path.normcase(str(Path(path).resolve()))
        for path in strategy.repos
    }
    if strategy_repo_paths and strategy_repo_paths != {target_repo_path}:
        raise RuntimeError(
            "The Hugging Face delete strategy includes folders outside "
            f"{repo_info.repo_path}; refusing to continue."
        )
    return cache_info, repo_info, strategy


def _has_size_mismatch(size_bytes, registry_size_bytes):
    return bool(
        registry_size_bytes
        and abs(size_bytes - registry_size_bytes) / registry_size_bytes > 0.10
    )


def _complete_revision_hashes(repo_id, model, revisions):
    if model is None:
        return set()
    return {
        revision.commit_hash
        for revision in revisions
        if _snapshot_is_complete(
            Path(revision.snapshot_path),
            repo_id,
            model,
        )
    }


def _cache_model_record(
    repo_info,
    cache_root,
    app_models_dir,
    model,
    current_tier,
    required_repo_ids,
    fallback_repo_ids,
    superseded,
    model_sources,
):
    repo_id = repo_info.repo_id
    repo_folder = Path(repo_info.repo_path)
    revisions = tuple(repo_info.revisions)
    if model is not None:
        completed_revisions = _complete_revision_hashes(
            repo_id,
            model,
            revisions,
        )
        incomplete_revisions = tuple(
            revision.commit_hash for revision in revisions
            if revision.commit_hash not in completed_revisions
        )
        complete = bool(completed_revisions)
        incomplete_size_bytes = sum(
            int(revision.size_on_disk)
            for revision in revisions
            if revision.commit_hash in incomplete_revisions
        )
        if not complete and not incomplete_revisions:
            incomplete_size_bytes = int(repo_info.size_on_disk)
    else:
        completed_revisions = {
            revision.commit_hash for revision in revisions
            if revision.refs
        }
        incomplete_revisions = tuple(
            revision.commit_hash for revision in revisions
            if revision.commit_hash not in completed_revisions
        )
        complete = bool(completed_revisions)
        incomplete_size_bytes = sum(
            int(revision.size_on_disk)
            for revision in revisions
            if revision.commit_hash in incomplete_revisions
        )
    revision_hashes = tuple(
        revision.commit_hash for revision in revisions
    )
    sources = _cache_sources(repo_folder, repo_id)
    if model_sources:
        sources.extend(model_sources)
    is_external = not _is_relative_to(repo_folder, app_models_dir)
    status, underlying, protected, reason = _classify_model(
        repo_id,
        model,
        current_tier,
        required_repo_ids,
        superseded,
        sources,
        is_external,
        fallback_repo_ids,
        downloaded=complete,
    )
    if not complete:
        reason += "; no complete pinned snapshot could be verified"
    if incomplete_revisions:
        reason += (
            f"; {len(incomplete_revisions)} incomplete or unverified "
            "revision(s) are present"
        )

    size_bytes = int(repo_info.size_on_disk)
    registry_size_bytes = _registry_download_size(model)
    return ModelCache(
        repo_id=repo_id,
        path=repo_folder,
        cache_root=Path(cache_root),
        size_bytes=size_bytes,
        status=status,
        underlying_status=underlying,
        protected=protected,
        complete=complete,
        reason=reason,
        is_external=is_external,
        registry_size_bytes=registry_size_bytes,
        size_mismatch=_has_size_mismatch(
            size_bytes,
            registry_size_bytes,
        ),
        revision_hashes=revision_hashes,
        incomplete_revisions=incomplete_revisions,
        cache_layout=True,
        model_name=(model or {}).get("name", repo_id),
        purpose=(model or {}).get("purpose", "unknown"),
        incomplete_size_bytes=incomplete_size_bytes,
    )


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
    return result


def _fallback_repo_ids(registry, settings, current_tier):
    if current_tier not in _TIER_NAMES:
        return set()
    from app_paths import get_primary_vision_model_for_tier

    tier_index = _TIER_NAMES.index(current_tier)
    selected_vision = settings.get("big_vision_model", "gemma-4-E2B-it")
    chain = []
    for tier in reversed(_TIER_NAMES[:tier_index + 1]):
        model = get_primary_vision_model_for_tier(
            tier,
            registry,
            big_vision_model=selected_vision,
        )
        repo_id = model.get("repo_id") if model else None
        if repo_id and repo_id not in chain:
            chain.append(repo_id)
    return set(chain[1:2])


def _tier_label(tier):
    return _TIER_LABELS.get(tier, str(tier))


def _classify_model(
    repo_id,
    model,
    current_tier,
    required_repo_ids,
    superseded,
    sources,
    is_external,
    fallback_repo_ids,
    downloaded,
):
    registered_tiers = (
        set(model.get("tiers") or []) if model is not None else set()
    )
    optional_tier = model.get("for_tier") if model is not None else None
    if optional_tier in _TIER_NAMES:
        registered_tiers.add(optional_tier)

    if model is None:
        underlying = "EXTERNAL"
        reason = "not present in the model registry; may belong to another project"
    elif repo_id in required_repo_ids:
        underlying = "IN USE"
        tier = _tier_label(current_tier)
        reason = f"required by the current {tier} tier"
    elif repo_id in fallback_repo_ids and downloaded:
        underlying = "FALLBACK"
        reason = "next vision fallback for the current tier"
    else:
        assigned_tiers = [
            tier for tier in _TIER_NAMES if tier in registered_tiers
        ]
        other_tiers = [
            tier for tier in assigned_tiers
            if tier != current_tier
        ]
        if other_tiers:
            underlying = "OTHER TIER"
            assigned = ", ".join(_tier_label(tier) for tier in other_tiers)
            reason = (
                f"assigned to {assigned}; needed if this PC's tier changes "
                "or for users of that tier"
            )
        elif assigned_tiers:
            tier_names = ", ".join(
                _tier_label(tier) for tier in assigned_tiers
            )
            underlying = "OTHER TIER"
            reason = (
                f"registered for {tier_names} but not selected for the "
                f"current {_tier_label(current_tier)} tier"
            )
        elif (
            sources
            and all(source in _CANDIDATE_SOURCES for source in sources)
        ) or (
            model.get("benchmark_only")
            or model.get("candidate_tier")
        ):
            underlying = "CANDIDATE"
            reason = "downloaded for Voice Lab or vision comparison"
        elif repo_id in superseded:
            underlying = "SUPERSEDED"
            reason = superseded[repo_id].get(
                "reason", "listed as superseded in the model registry"
            )
        else:
            underlying = "CANDIDATE"
            reason = "registered without a tier assignment"

    protected = (
        underlying in {"IN USE", "FALLBACK"}
        or (
            repo_id in _NEVER_DELETE_REPO_IDS
            and bool(registered_tiers)
        )
    )
    if is_external and model is None:
        reason = (
            "shared Hugging Face cache; other projects may use it. "
            + reason
        )
    return underlying, underlying, protected, reason


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
    fallback_repo_ids = _fallback_repo_ids(
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
        cache_info = _scan_cache(cache_root)
        scanned_folders = set()
        if cache_info is not None:
            for repo_info in cache_info.repos:
                folder = Path(repo_info.repo_path)
                location_key = os.path.normcase(str(folder.resolve()))
                if location_key in seen_locations:
                    continue
                seen_locations.add(location_key)
                scanned_folders.add(folder.name)
                model = next(
                    iter(model_map.get(repo_info.repo_id, [])),
                    None,
                )
                if _is_reparse_point(folder):
                    revisions = tuple(repo_info.revisions)
                    registry_size = _registry_download_size(model)
                    is_external = not _is_relative_to(
                        folder, app_models_dir
                    )
                    status, underlying, protected, reason = _classify_model(
                        repo_info.repo_id,
                        model,
                        current_tier,
                        required_repo_ids,
                        superseded,
                        [],
                        is_external,
                        fallback_repo_ids,
                        downloaded=False,
                    )
                    found.append(ModelCache(
                        repo_id=repo_info.repo_id,
                        path=folder,
                        cache_root=cache_root,
                        size_bytes=0,
                        status=status,
                        underlying_status=underlying,
                        protected=protected,
                        complete=False,
                        reason=reason + "; repo folder is a link",
                        is_external=is_external,
                        scan_error="repo cache folder is a link",
                        registry_size_bytes=registry_size,
                        size_mismatch=_has_size_mismatch(
                            0, registry_size
                        ),
                        revision_hashes=tuple(
                            revision.commit_hash for revision in revisions
                        ),
                        incomplete_revisions=tuple(
                            revision.commit_hash for revision in revisions
                        ),
                        cache_layout=True,
                        model_name=(model or {}).get(
                            "name", repo_info.repo_id
                        ),
                        purpose=(model or {}).get("purpose", "unknown"),
                        incomplete_size_bytes=sum(
                            int(revision.size_on_disk)
                            for revision in revisions
                        ),
                    ))
                    continue
                found.append(_cache_model_record(
                    repo_info=repo_info,
                    cache_root=cache_root,
                    app_models_dir=app_models_dir,
                    model=model,
                    current_tier=current_tier,
                    required_repo_ids=required_repo_ids,
                    fallback_repo_ids=fallback_repo_ids,
                    superseded=superseded,
                    model_sources=[],
                ))

        try:
            folders = sorted(cache_root.iterdir())
        except OSError as error:
            print(f"WARNING: Cannot inspect Hugging Face cache {cache_root}: {error}")
            continue
        for folder in folders:
            repo_id = _repo_id_from_folder(folder)
            if (
                repo_id is None
                or not folder.is_dir()
                or folder.name in scanned_folders
            ):
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
            status, underlying, protected, reason = _classify_model(
                repo_id,
                model,
                current_tier,
                required_repo_ids,
                superseded,
                sources,
                external,
                fallback_repo_ids,
                downloaded=complete,
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
                registry_size_bytes=_registry_download_size(model),
                size_mismatch=_has_size_mismatch(
                    size_bytes,
                    _registry_download_size(model),
                ),
                revision_hashes=tuple(
                    snapshot.name
                    for snapshot in (folder / "snapshots").iterdir()
                ) if (folder / "snapshots").is_dir() else (),
                incomplete_revisions=(
                    tuple(
                        snapshot.name
                        for snapshot in (folder / "snapshots").iterdir()
                        if snapshot not in complete_snapshots
                    )
                    if (folder / "snapshots").is_dir()
                    else ()
                ),
                cache_layout=False,
                model_name=(model or {}).get("name", repo_id),
                purpose=(model or {}).get("purpose", "unknown"),
                incomplete_size_bytes=sum(
                    _size_on_disk(snapshot)[0]
                    for snapshot in (folder / "snapshots").iterdir()
                    if snapshot not in complete_snapshots
                ) if (folder / "snapshots").is_dir() else 0,
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
    if byte_count >= 1_000_000:
        return f"{byte_count / 1_000_000:.1f} MB"
    if byte_count >= 1_000:
        return f"{byte_count / 1_000:.1f} KB"
    return f"{byte_count} B"


def _format_gb(byte_count):
    return f"{byte_count / 1_000_000_000:.3f} GB"


def _print_model(model, output_fn=print):
    protected = " [protected]" if model.protected else ""
    complete = "complete" if model.complete else "incomplete"
    name = model.model_name or model.repo_id
    location = (
        "shared Hugging Face cache"
        if model.is_external
        else "app models folder"
    )
    if model.status == "IN USE":
        deletable = f"No - {model.reason}"
    elif (
        model.repo_id in _NEVER_DELETE_REPO_IDS
        and model.protected
    ):
        deletable = "No - protected while assigned to a performance tier"
    elif model.status == "FALLBACK":
        deletable = "Only after fallback warning and explicit confirmation"
    elif model.status == "OTHER TIER":
        deletable = "Yes - explicit other-tier warning and confirmation"
    elif model.status == "EXTERNAL":
        deletable = "Yes - shared-cache warning; confirmation defaults to No"
    else:
        deletable = "Yes - confirmation required (default No)"
    registry_size = (
        _format_gb(model.registry_size_bytes)
        if model.registry_size_bytes
        else "not in registry"
    )
    mismatch = " | SIZE MISMATCH >10%" if model.size_mismatch else ""
    output_fn(
        f"{model.status}{protected} | {name} ({model.repo_id}) | "
        f"real {_format_gb(model.size_bytes)} | registry {registry_size} | "
        f"{complete}{mismatch} | location: {location} | {model.path} | "
        f"deletable: {deletable} | {model.reason}"
    )
    if model.is_external and model.status == "EXTERNAL":
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
        or (
            model.protected
            and not (
                model.status == "FALLBACK"
                and model.repo_id not in _NEVER_DELETE_REPO_IDS
            )
        )
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


def _find_repo_info(cache_info, repo_id):
    if cache_info is None:
        return None
    return next(
        (repo for repo in cache_info.repos if repo.repo_id == repo_id),
        None,
    )


def _prepare_cache_delete(model):
    folder = Path(model.path)
    cache_root = Path(model.cache_root).resolve()
    if (
        _is_reparse_point(folder)
        or _contains_reparse_directory(folder)
        or folder.name != "models--" + model.repo_id.replace("/", "--")
        or folder.parent.resolve() != cache_root
    ):
        return None
    cache_info, repo_info, strategy = _delete_strategy_for_revisions(
        model,
        model.revision_hashes,
    )
    if cache_info is None or repo_info is None or strategy is None:
        return None
    return cache_info, strategy, repo_info


def _delete_with_cache_strategy(model, prepared, output_fn):
    cache_info, strategy, repo_info = prepared
    before_total = _cache_info_total_size(cache_info)
    expected = int(strategy.expected_freed_size)
    strategy.execute()

    after_info = _scan_cache(model.cache_root)
    after_total = _cache_info_total_size(after_info)
    actual = before_total - after_total
    after_repo = _find_repo_info(after_info, model.repo_id)
    before_repo_size = int(repo_info.size_on_disk)
    after_repo_size = (
        int(after_repo.size_on_disk) if after_repo is not None else 0
    )
    output_fn(
        f"Verified after deletion: cache size decreased by "
        f"{_format_size(actual)} (expected "
        f"{_format_size(expected)}); this repo decreased from "
        f"{_format_size(before_repo_size)} to "
        f"{_format_size(after_repo_size)}."
    )
    if actual != expected:
        output_fn(
            "WARNING: measured freed space differs from the Hugging Face "
            "strategy estimate."
        )
    return actual


def _delete_non_cache_folder(model, output_fn):
    before, errors = _size_on_disk(model.path)
    if errors:
        output_fn(
            "WARNING: Some files could not be measured before deletion: "
            + "; ".join(errors)
        )
    _safe_remove_model_folder(model)
    if Path(model.path).exists():
        after, _ = _size_on_disk(model.path)
    else:
        after = 0
    output_fn(
        f"Verified after deletion: folder size decreased by "
        f"{_format_size(before - after)}."
    )


def cleanup_models(
    models,
    apply=False,
    input_fn=input,
    output_fn=print,
    remove_model_folder=None,
):
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
            or (
                model.protected
                and not (
                    model.status == "FALLBACK"
                    and model.repo_id not in _NEVER_DELETE_REPO_IDS
                )
            )
        ):
            if (
                model.repo_id in _NEVER_DELETE_REPO_IDS
                and model.protected
            ):
                output_fn(
                    f"Keeping {model.repo_id}: protected while assigned "
                    "to a performance tier."
                )
            else:
                output_fn(
                    f"Keeping {model.repo_id}: IN USE models cannot be deleted."
                )
            continue
        if model.status == "FALLBACK":
            output_fn(
                f"WARNING: {model.repo_id} is the next vision fallback. "
                "Deleting it may reduce fallback coverage."
            )
        if model.status == "OTHER TIER":
            output_fn(
                f"WARNING: {model.repo_id} is needed if this PC's tier "
                "changes or for users of that tier."
            )
        if model.is_external:
            output_fn(
                "WARNING: Removing this shared cache entry may affect other "
                "projects on this PC. The default response is no."
            )
        prepared = None
        if model.cache_layout and remove_model_folder is None:
            prepared = _prepare_cache_delete(model)
            if prepared is None:
                output_fn(
                    f"Keeping {model.repo_id}: no Hugging Face cached "
                    "revision is available for a safe deletion strategy."
                )
                continue
            _cache_info, strategy, _repo_info = prepared
            expected_freed = int(strategy.expected_freed_size)
            output_fn(
                f"Hugging Face cache strategy expects to free "
                f"{_format_size(expected_freed)}."
            )
        elif not model.complete:
            output_fn(
                f"INCOMPLETE: {model.repo_id} has unverified cache files; "
                "delete only if you explicitly approve this folder."
            )

        incomplete_note = " [INCOMPLETE]" if not model.complete else ""
        if model.status == "FALLBACK":
            prompt = (
                f"Delete protected fallback {model.repo_id}{incomplete_note} "
                f"({_format_size(model.size_bytes)}) at {model.path}? "
                "[y/N]: "
            )
        else:
            prompt = (
                f"Delete {model.repo_id}{incomplete_note} "
                f"({_format_size(model.size_bytes)}) at {model.path}? [y/N]: "
            )
        if not _confirm(prompt, input_fn):
            output_fn(f"Kept {model.repo_id}.")
            continue
        if remove_model_folder is not None:
            remove_model_folder(model)
        elif model.cache_layout:
            _delete_with_cache_strategy(model, prepared, output_fn)
        else:
            _delete_non_cache_folder(model, output_fn)
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


def _print_inventory_summary(models, output_fn=print):
    total_bytes = sum(model.size_bytes for model in models)
    output_fn(f"Total model-cache size: {_format_gb(total_bytes)}")
    incomplete_models = [
        model for model in models
        if model.incomplete_revisions or not model.complete
    ]
    output_fn("Partial or incomplete downloads:")
    if not incomplete_models:
        output_fn("  None found.")
        return
    for model in incomplete_models:
        if model.cache_layout:
            _info, _repo, strategy = _delete_strategy_for_revisions(
                model,
                model.incomplete_revisions,
            )
            unique_bytes = (
                int(strategy.expected_freed_size)
                if strategy is not None
                else 0
            )
        else:
            unique_bytes = model.incomplete_size_bytes
        output_fn(
            f"  {model.model_name or model.repo_id} ({model.repo_id}) | "
            f"revision size {_format_size(model.incomplete_size_bytes)} | "
            f"unique reclaimable {_format_size(unique_bytes)} | "
            f"{model.path}"
        )


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
    _print_inventory_summary(models)
    if not models:
        print("No downloaded model folders found.")
    cleanup_models(models, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
