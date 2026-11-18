"""
indexing/embedder.py

Wraps sentence-transformers (all-MiniLM-L6-v2) to turn text into embedding
vectors, running on CPU. This is the Phase 2 starter model per CLAUDE.md —
gets swapped for the QNN-converted embedding model in Phase 3.
"""
import os
import threading
from pathlib import Path

from app_paths import find_model
from indexing.model_downloader import set_huggingface_offline

HF_HOME = os.environ.setdefault(
    "HF_HOME", str(Path.home() / ".cache" / "huggingface")
)
os.environ.setdefault("HF_HUB_CACHE", str(Path(HF_HOME) / "hub"))

from sentence_transformers import SentenceTransformer

_model = None  # loaded lazily, once, on first use
_model_lock = threading.Lock()
MODEL_REPO_ID = "sentence-transformers/all-mpnet-base-v2"


def get_model():
    """
    Load the embedding model once and reuse it (loading is slow).
    Using all-mpnet-base-v2 (768-dim) instead of the original
    all-MiniLM-L6-v2 (384-dim) starter model -- stronger semantic
    accuracy, still free/local/CPU-only, chosen after Task 3.6's golden-
    set eval showed MiniLM struggling to distinguish closely-related
    technical documents. Real QNN-optimized embedding model still comes
    later in Task 3.8's NPU porting.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model_path = find_model(MODEL_REPO_ID)
                if model_path is None:
                    raise RuntimeError(
                        "The all-mpnet-base-v2 model is not fully installed. "
                        "Complete model setup before embedding."
                    )
                set_huggingface_offline(True)
                _model = SentenceTransformer(
                    str(model_path),
                    local_files_only=True,
                )
    return _model


def preload_model():
    """Load the embedding model before the first interactive query."""
    return get_model()


def embed_text(text):
    """
    Turn a single string into an embedding vector (list of floats).
    """
    model = get_model()
    vector = model.encode(text)
    return vector.tolist()


def embed_texts(texts):
    """
    Turn a list of strings into a list of embedding vectors.
    More efficient than calling embed_text() in a loop for many chunks.
    """
    model = get_model()
    vectors = model.encode(texts)
    return [v.tolist() for v in vectors]