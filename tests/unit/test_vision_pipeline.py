import torch
from PIL import Image

from indexing import vision_pipeline


def _model(repo_id, tier, family, requires_cuda=False, status="ready"):
    return {
        "repo_id": repo_id,
        "tiers": [tier],
        "family": family,
        "requires_cuda": requires_cuda,
        "status": status,
        "prompt": "Describe only what is visible.",
        "max_new_tokens": 12,
    }


def test_tier_resolution_steps_down_without_cuda_and_marks_big_untested():
    models = [
        _model("small/repo", "small", "smolvlm"),
        _model("mid/repo", "mid", "gemma4", requires_cuda=True),
        _model(
            "big/repo", "big", "gemma4",
            requires_cuda=True, status="untested",
        ),
    ]

    assert vision_pipeline.get_vision_model(
        "mid", {"cuda_available": False}, models
    )["repo_id"] == "small/repo"
    assert vision_pipeline.get_vision_model(
        "big", {"cuda_available": True}, models
    )["status"] == "untested"


class _Batch(dict):
    def to(self, _device):
        return self


class _FakeSmolProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, add_generation_prompt):
        self.messages = messages
        assert add_generation_prompt
        return "formatted prompt"

    def __call__(self, text, images, return_tensors):
        assert text == "formatted prompt"
        assert len(images) == 1
        assert return_tensors == "pt"
        return _Batch(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, _tokens, skip_special_tokens):
        assert skip_special_tokens
        return ["A small blue icon with a white cloud."]


class _FakeModel:
    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def generate(self, input_ids, max_new_tokens):
        assert input_ids.shape == (1, 2)
        assert max_new_tokens == 12
        return torch.tensor([[1, 2, 3]])


def test_mocked_smolvlm_loader_describes_image_with_registry_prompt(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "blue").save(image_path)
    model = _model(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "small",
        "smolvlm",
    )
    processor = _FakeSmolProcessor()
    monkeypatch.setattr(vision_pipeline, "VISION_MODELS", [model])
    monkeypatch.setattr(vision_pipeline, "get_current_tier", lambda: "small")
    monkeypatch.setattr(
        vision_pipeline, "_runtime_hardware",
        lambda: {"cuda_available": False},
    )
    monkeypatch.setattr(vision_pipeline, "find_model", lambda _repo: tmp_path)
    monkeypatch.setattr(
        vision_pipeline, "_load_smolvlm",
        lambda *_args: (_FakeModel(), processor),
    )
    vision_pipeline.reset_vision_state()
    model_statuses = []

    description = vision_pipeline.describe_image(
        image_path,
        model_status_callback=model_statuses.append,
    )

    assert description == "A small blue icon with a white cloud."
    assert processor.messages[0]["content"][1]["text"] == model["prompt"]
    assert vision_pipeline.get_active_vision_model() == model["repo_id"]
    assert model_statuses == [
        f"Loading vision model: {model['repo_id']}..."
    ]
    vision_pipeline.reset_vision_state()


def test_failed_untested_big_model_falls_back_to_mid_once(tmp_path, monkeypatch):
    big = _model(
        "google/gemma-4-E4B-it", "big", "gemma4",
        requires_cuda=True, status="untested",
    )
    mid = _model(
        "google/gemma-4-E2B-it", "mid", "gemma4",
        requires_cuda=True,
    )
    small = _model(
        "HuggingFaceTB/SmolVLM-500M-Instruct", "small", "smolvlm"
    )
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "green").save(image_path)
    attempts = []

    def load(model_entry, *_args):
        attempts.append(model_entry["repo_id"])
        if model_entry is big:
            raise MemoryError("simulated out-of-memory")
        return object(), object()

    monkeypatch.setattr(
        vision_pipeline, "VISION_MODELS", [small, mid, big]
    )
    monkeypatch.setattr(vision_pipeline, "get_current_tier", lambda: "big")
    monkeypatch.setattr(
        vision_pipeline, "_runtime_hardware",
        lambda: {"cuda_available": True},
    )
    monkeypatch.setattr(vision_pipeline, "find_model", lambda _repo: tmp_path)
    monkeypatch.setattr(vision_pipeline, "_load_model", load)
    monkeypatch.setattr(
        vision_pipeline, "_generate_description",
        lambda entry, *_args: f"caption from {entry['repo_id']}",
    )
    vision_pipeline.reset_vision_state()

    assert vision_pipeline.describe_image(image_path) == (
        "caption from google/gemma-4-E2B-it"
    )
    assert attempts == ["google/gemma-4-E4B-it", "google/gemma-4-E2B-it"]
    assert vision_pipeline.get_active_vision_model() == "google/gemma-4-E2B-it"
    assert "google/gemma-4-E4B-it" in vision_pipeline._unavailable_models
    vision_pipeline.reset_vision_state()


def test_vision_failures_fall_through_to_filename_only_without_retrying(
    tmp_path, monkeypatch
):
    models = [
        _model("small/repo", "small", "smolvlm"),
        _model("mid/repo", "mid", "gemma4", requires_cuda=True),
        _model(
            "big/repo", "big", "gemma4",
            requires_cuda=True, status="untested",
        ),
    ]
    image_path = tmp_path / "fallback-name.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    attempts = []

    def fail(model_entry, *_args):
        attempts.append(model_entry["repo_id"])
        raise RuntimeError("simulated model load failure")

    monkeypatch.setattr(vision_pipeline, "VISION_MODELS", models)
    monkeypatch.setattr(vision_pipeline, "get_current_tier", lambda: "big")
    monkeypatch.setattr(
        vision_pipeline, "_runtime_hardware",
        lambda: {"cuda_available": True},
    )
    monkeypatch.setattr(vision_pipeline, "find_model", lambda _repo: tmp_path)
    monkeypatch.setattr(vision_pipeline, "_load_model", fail)
    vision_pipeline.reset_vision_state()

    assert vision_pipeline.describe_image(image_path) == "fallback-name"
    assert attempts == ["big/repo", "mid/repo", "small/repo"]
    assert vision_pipeline.get_active_vision_model() is None

    assert vision_pipeline.describe_image(image_path) == "fallback-name"
    assert attempts == ["big/repo", "mid/repo", "small/repo"]
    vision_pipeline.reset_vision_state()


def test_missing_local_fallback_uses_filename_only_and_never_downloads(
    tmp_path, monkeypatch
):
    big = _model(
        "big/repo", "big", "gemma4",
        requires_cuda=True, status="untested",
    )
    small = _model("small/repo", "small", "smolvlm")
    image_path = tmp_path / "fallback-absent.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    callbacks = []
    download_calls = []

    def fail_big(model_entry, *_args):
        if model_entry is big:
            raise RuntimeError("simulated primary load failure")
        raise AssertionError("an absent model must not be loaded")

    monkeypatch.setattr(vision_pipeline, "VISION_MODELS", [small, big])
    monkeypatch.setattr(vision_pipeline, "get_current_tier", lambda: "big")
    monkeypatch.setattr(
        vision_pipeline, "_runtime_hardware",
        lambda: {"cuda_available": True},
    )
    monkeypatch.setattr(
        vision_pipeline, "find_model",
        lambda repo: tmp_path if repo == big["repo_id"] else None,
    )
    monkeypatch.setattr(vision_pipeline, "_load_model", fail_big)
    monkeypatch.setattr(
        "indexing.model_downloader.download_model",
        lambda *args, **kwargs: download_calls.append(args),
    )
    vision_pipeline.reset_vision_state()

    assert vision_pipeline.describe_image(
        image_path, status_callback=callbacks.append
    ) == "fallback-absent"
    assert callbacks == [
        "Fallback model not downloaded - using file names only for now"
    ]
    assert download_calls == []
    vision_pipeline.reset_vision_state()


def test_runtime_out_of_memory_steps_down_to_small_model(tmp_path, monkeypatch):
    mid = _model(
        "google/gemma-4-E2B-it", "mid", "gemma4",
        requires_cuda=True,
    )
    small = _model(
        "HuggingFaceTB/SmolVLM-500M-Instruct", "small", "smolvlm"
    )
    image_path = tmp_path / "runtime-oom.png"
    Image.new("RGB", (4, 4), "yellow").save(image_path)
    generated = []
    monkeypatch.setattr(vision_pipeline, "VISION_MODELS", [small, mid])
    monkeypatch.setattr(vision_pipeline, "get_current_tier", lambda: "mid")
    monkeypatch.setattr(
        vision_pipeline, "_runtime_hardware",
        lambda: {"cuda_available": True},
    )
    monkeypatch.setattr(vision_pipeline, "find_model", lambda _repo: tmp_path)
    monkeypatch.setattr(
        vision_pipeline, "_load_model",
        lambda *_args: (object(), object()),
    )

    def generate(entry, *_args):
        generated.append(entry["repo_id"])
        if entry is mid:
            raise torch.cuda.OutOfMemoryError("simulated runtime OOM")
        return "small model description"

    monkeypatch.setattr(vision_pipeline, "_generate_description", generate)
    vision_pipeline.reset_vision_state()

    assert vision_pipeline.describe_image(image_path) == (
        "small model description"
    )
    assert generated == [mid["repo_id"], small["repo_id"]]
    assert vision_pipeline.get_active_vision_model() == small["repo_id"]
    vision_pipeline.reset_vision_state()


def test_image_indexing_records_description_model(tmp_path, monkeypatch):
    from indexing import backfill_index

    image_path = tmp_path / "with-model.png"
    image_path.write_bytes(b"image bytes")
    saved = {}
    vectors = []
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    monkeypatch.setattr(backfill_index, "init_db", lambda: None)
    monkeypatch.setattr(backfill_index, "init_chunks_table", lambda: None)
    monkeypatch.setattr(backfill_index, "get_file_metadata", lambda _path: None)
    monkeypatch.setattr(
        backfill_index, "image_description_status", lambda: (True, "")
    )
    monkeypatch.setattr(
        backfill_index, "describe_image",
        lambda _path, **_kwargs: "a short caption"
    )
    monkeypatch.setattr(
        backfill_index, "get_active_vision_model", lambda: model_id
    )
    monkeypatch.setattr(backfill_index, "embed_text", lambda _text: [0.5])
    monkeypatch.setattr(
        backfill_index, "add_item",
        lambda item_id, _vector, metadata: vectors.append((item_id, metadata)),
    )
    monkeypatch.setattr(
        backfill_index, "save_file_metadata",
        lambda *args, **kwargs: saved.update(kwargs),
    )
    monkeypatch.setattr(backfill_index, "route_file", lambda _path: "image")
    monkeypatch.setattr(
        backfill_index, "unload_vision_model", lambda: None
    )

    backfill_index.index_one_file(image_path)

    assert saved["vision_model"] == model_id
    assert any(metadata["match_type"] == "content" for _, metadata in vectors)


def test_backfill_unloads_vision_model_after_image_batch(tmp_path, monkeypatch):
    from indexing import backfill_index

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "one.png").write_bytes(b"not decoded by this test")
    (image_dir / "two.jpg").write_bytes(b"not decoded by this test")
    indexed = []
    unloads = []
    monkeypatch.setattr(backfill_index, "init_db", lambda: None)
    monkeypatch.setattr(backfill_index, "init_chunks_table", lambda: None)
    monkeypatch.setattr(backfill_index, "path_migration_complete", lambda: True)
    monkeypatch.setattr(backfill_index, "count_items", lambda: 1)
    monkeypatch.setattr(
        backfill_index,
        "index_one_file",
        lambda path, **kwargs: indexed.append((path, kwargs)) or "[INDEXED]",
    )
    monkeypatch.setattr(
        backfill_index, "unload_vision_model", lambda: unloads.append(True)
    )
    monkeypatch.setattr(
        backfill_index, "cleanup_missing_files", lambda **_kwargs: []
    )

    backfill_index.backfill(indexed_folders=[image_dir])

    assert len(indexed) == 2
    assert all(call[1]["retain_vision_model"] for call in indexed)
    assert unloads == [True]
