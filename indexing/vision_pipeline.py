"""
indexing/vision_pipeline.py

Real image understanding via Gemma 4 E2B-it (google/gemma-4-E2B-it),
4-bit quantized. Switched from the originally-planned Qwen3-VL-4B-Instruct
during Task 3.8 testing: Qwen3-VL-4B took ~60 seconds per image on this
4GB GPU, while Gemma E2B (2B active parameters vs Qwen's dense 4B) took
~18 seconds for a comparable description -- roughly 3.4x faster, with
comparable description quality. This follows CLAUDE.md's own documented
fallback plan ("Fallback to Gemma-4-E2B-it only if latency proves
unworkable on-device").

This only runs at INDEXING time (once per image, in the background via
file_watcher.py / backfill_index.py) -- never at search/query time.
"""

import os
from pathlib import Path

HF_HOME = os.environ.setdefault(
    "HF_HOME", str(Path.home() / ".cache" / "huggingface")
)
HF_HUB_CACHE = os.environ.setdefault(
    "HF_HUB_CACHE", str(Path(HF_HOME) / "hub")
)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image

_model = None
_processor = None

MODEL_NAME = "google/gemma-4-E2B-it"
MAX_IMAGE_DIMENSION = 768


def get_model_and_processor():
    """Load the vision model once and reuse it (loading is slow)."""
    global _model, _processor
    if _model is None:
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME, padding_side="left", cache_dir=HF_HUB_CACHE
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="cuda",
            cache_dir=HF_HUB_CACHE,
        )
    return _model, _processor


def describe_image(file_path):
    """
    Generate a real description of the image at file_path using
    Gemma 4 E2B-it. Takes roughly 15-20 seconds per call on a 4GB GPU --
    intended for one-time use at indexing time only (see module docstring).
    """
    model, processor = get_model_and_processor()

    image = Image.open(file_path)
    image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one or two sentences, focusing on what it depicts."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60)

    response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    del inputs, outputs
    torch.cuda.empty_cache()

    return response.strip()