"""Local, tier-aware image descriptions used only while indexing."""

import gc
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from app.hardware_profile import get_current_tier
from app_paths import (
    find_model,
    get_big_vision_model_setting,
    get_config_path,
    get_primary_vision_model_for_tier,
)
from observability.logger import log

MAX_IMAGE_DIMENSION = 768
_TIER_ORDER = ("small", "mid", "big")
_VISION_STATUSES = {"ready", "untested"}

_model = None
_processor = None
_loaded_model = None
_active_model_repo_id = None
_unavailable_models = set()


def _load_vision_registry():
    with open(get_config_path("model_tiers.json"), encoding="utf-8-sig") as file:
        return json.load(file)


VISION_REGISTRY = _load_vision_registry()
VISION_MODELS = [
    model
    for model in (
        list(VISION_REGISTRY.get("models", []))
        + list(VISION_REGISTRY.get("optional_models", []))
    )
    if model.get("purpose") == "vision"
]
_DEFAULT_VISION_MODELS = VISION_MODELS


def _primary_vision_model(tier, models):
    if models is _DEFAULT_VISION_MODELS:
        return get_primary_vision_model_for_tier(
            tier,
            VISION_REGISTRY,
            big_vision_model=get_big_vision_model_setting(),
        )
    return next(
        (
            model for model in models
            if tier in model.get("tiers", [])
            and model.get("status") in _VISION_STATUSES
            and model.get("repo_id")
        ),
        None,
    )


def _iter_vision_models(tier, hardware, models=None):
    models = VISION_MODELS if models is None else models
    use_registry = models is _DEFAULT_VISION_MODELS
    try:
        tier_index = _TIER_ORDER.index(tier)
    except ValueError:
        return
    available = bool(hardware.get("cuda_available"))
    seen = set()
    for candidate_tier in reversed(_TIER_ORDER[:tier_index + 1]):
        if use_registry:
            model = get_primary_vision_model_for_tier(
                candidate_tier,
                VISION_REGISTRY,
                big_vision_model=get_big_vision_model_setting(),
            )
        else:
            model = next(
                (
                    item for item in models
                    if candidate_tier in item.get("tiers", [])
                    and item.get("status") in _VISION_STATUSES
                    and item.get("repo_id")
                ),
                None,
            )
        if model is not None and (
            available or not model.get("requires_cuda", False)
        ) and model.get("repo_id") not in seen:
            seen.add(model["repo_id"])
            yield model


def get_vision_model(tier, hardware, models=None):
    """Return the highest usable configured model at or below ``tier``."""
    return next(_iter_vision_models(tier, hardware, models=models), None)


def _runtime_hardware():
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as error:
        log(f"WARNING: CUDA detection failed for image indexing: {error}")
        cuda_available = False
    return {"cuda_available": cuda_available}


def image_description_status():
    """Return whether an installed model can describe images on this tier."""
    hardware = _runtime_hardware()
    tier = get_current_tier()
    models = VISION_MODELS
    candidates = list(_iter_vision_models(tier, hardware, models))
    primary = _primary_vision_model(tier, models)
    primary_repo = primary.get("repo_id") if primary else None
    missing_fallback = False
    for model in candidates:
        if model["repo_id"] in _unavailable_models:
            continue
        if find_model(model["repo_id"]) is not None:
            return True, ""
        if model["repo_id"] != primary_repo:
            missing_fallback = True
    if not candidates:
        return False, "no compatible vision model for the available hardware"
    if missing_fallback:
        return False, "fallback model not downloaded"
    return False, "no compatible pinned vision model is installed"


def _load_gemma(model_entry, model_path):
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        padding_side="left",
        local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="cuda",
        local_files_only=True,
    )
    return model, processor


def _load_smolvlm(model_entry, model_path, hardware):
    cuda_available = bool(hardware.get("cuda_available"))
    load_options = {
        "torch_dtype": torch.float16 if cuda_available else torch.float32,
        "local_files_only": True,
    }
    if cuda_available:
        load_options["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), **load_options
    )
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True
    )
    return model, processor


def _load_model(model_entry, model_path, hardware):
    if model_entry.get("family") == "gemma4":
        return _load_gemma(model_entry, model_path)
    if model_entry.get("family") == "smolvlm":
        return _load_smolvlm(model_entry, model_path, hardware)
    raise ValueError(f"Unknown vision model family: {model_entry.get('family')!r}")


def _model_device(model, hardware):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return "cuda" if hardware.get("cuda_available") else "cpu"


def _generate_description(model_entry, model, processor, image, hardware):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"} if model_entry["family"] == "smolvlm"
            else {"type": "image", "image": image},
            {"type": "text", "text": model_entry["prompt"]},
        ],
    }]
    if model_entry["family"] == "smolvlm":
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = processor(
            text=prompt, images=[image], return_tensors="pt"
        )
    else:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    inputs = inputs.to(_model_device(model, hardware))
    outputs = None
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(model_entry.get("max_new_tokens", 60)),
            )
        input_length = inputs["input_ids"].shape[-1]
        result = processor.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )[0].strip()
        if not result:
            raise RuntimeError("The vision model returned an empty description")
        return result
    finally:
        del inputs
        if outputs is not None:
            del outputs
        gc.collect()
        if hardware.get("cuda_available"):
            torch.cuda.empty_cache()


def unload_vision_model():
    """Release vision-model RAM/VRAM after an image batch."""
    global _model, _processor, _loaded_model, _active_model_repo_id
    _model = None
    _processor = None
    _loaded_model = None
    _active_model_repo_id = None
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:
        log(f"WARNING: Could not clear CUDA cache after image indexing: {error}")


def reset_vision_state():
    """Reset model state for isolated evaluation and tests."""
    global _unavailable_models
    unload_vision_model()
    _unavailable_models = set()


def get_active_vision_model():
    """Return the repository id that generated the latest description."""
    return _active_model_repo_id


def describe_image(file_path, status_callback=None):
    """Describe an image locally, falling back through lower tiers if needed."""
    global _model, _processor, _loaded_model, _active_model_repo_id
    _active_model_repo_id = None
    hardware = _runtime_hardware()
    tier = get_current_tier()
    models = VISION_MODELS
    candidates = list(_iter_vision_models(tier, hardware, models))
    primary = _primary_vision_model(tier, models)
    primary_repo = primary.get("repo_id") if primary else None
    warned_missing_fallback = False

    try:
        with Image.open(file_path) as source:
            image = source.convert("RGB")
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
    except Exception as error:
        log(f"WARNING: Cannot open image {file_path}: {error}")
        return Path(file_path).stem

    try:
        for model_entry in candidates:
            repo_id = model_entry["repo_id"]
            if repo_id in _unavailable_models:
                continue
            model_path = find_model(repo_id)
            if model_path is None:
                if repo_id != primary_repo and not warned_missing_fallback:
                    message = (
                        "Fallback model not downloaded - using file names only for now"
                    )
                    if status_callback is not None:
                        status_callback(message)
                    else:
                        log(f"WARNING: {message} ({repo_id})")
                    warned_missing_fallback = True
                continue
            try:
                if _loaded_model != repo_id:
                    unload_vision_model()
                    _model, _processor = _load_model(
                        model_entry, model_path, hardware
                    )
                    _loaded_model = repo_id
                description = _generate_description(
                    model_entry, _model, _processor, image, hardware
                )
                _active_model_repo_id = repo_id
                return description
            except Exception as error:
                _unavailable_models.add(repo_id)
                log(
                    f"WARNING: Vision model {repo_id} failed for "
                    f"{file_path}; trying the next tier: {error}"
                )
                unload_vision_model()
        return Path(file_path).stem
    finally:
        image.close()
