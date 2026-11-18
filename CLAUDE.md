# Smart File Finder

## Locked decisions

- Confident search results show one file.
- Confident result cards use the compact reference style and can be moved by dragging the card surface.
- Not-sure results show a vertical stack of 1-3 separate, movable popup cards on the left side of the cursor/input-bar screen, selected by relevance.
- The best file is always shown; additional cards must score at least `0.50` and be within `0.03` of the best score.
- There is no low-confidence confirmation popup.
- An uncertain result is never opened automatically, including for `/open`.

## Models and search thresholds

- Embeddings use `all-mpnet-base-v2` (768 dimensions, local CPU).
- Image understanding runs only while indexing: SmolVLM-500M on small, Gemma E2B 4-bit on mid, and Gemma E4B 4-bit on big (big is untested).
- `Qwen/Qwen3-1.7B` is used by the indexing-time summarizer only.
- There is no live LLM reasoning in the search path; query refinement and result decisions are rule-based.
- The only network use is downloading public model files from Hugging Face during explicit first-run setup. No user file, query, or telemetry is ever sent. After setup the app works fully offline.
- `config/model_tiers.json` is the model registry. Download only the loader-required files pinned to each model revision; `download_size_gb` is that selected payload, while `repository_size_gb` records the full repository size.
- Models are stored under the app data folder's `models` directory. Resolve models with `app_paths.find_model`; enable Hugging Face offline mode only after required models are local.
- Vision fallback order is big -> mid -> small -> filename-only; CUDA-required models are skipped without CUDA. Small uses SmolVLM on CPU or GPU without bitsandbytes.
- Load vision models lazily for the first image in a scan, release them and clear CUDA memory after the batch, and store the generating model repo id with image metadata. Tier changes do not re-index existing files automatically.
- Confidence threshold: `0.55`; ambiguity margin: `0.07`; options floor: `0.48`; maximum options: `3`.

## Update round (Nov 2026)

Completed fixes:

1. Remove stale vectors and metadata when indexed files change, are deleted, or are moved, and clean missing files during startup.
2. Save indexed metadata only after successful vector writes; remove partial data on failure and cap consecutive retries.
3. Resolve packaged configuration paths reliably and create missing configuration files with defaults.
4. Enforce a hard per-query deadline.
5. Add fast, rule-based query refinement and stop duplicate retries.
6. Normalize stored paths, make cleanup safe, fix integration teardown, and make evaluation index its own corpus.
7. Make the startup shortcut optional in the installer.

The golden set has been expanded to 20 cases and includes text, document, image, ambiguous, and no-match queries.

Feature 1 (top-three pick list) is implemented: confident queries return one file, uncertain queries offer up to three distinct files, and low-confidence confirmation popups are not used.

Device-tier plan:

- Device tiers keep embeddings fixed at `all-mpnet-base-v2` for every tier so existing vector indexes remain compatible. Tiers affect vision, voice, and optional summarizer choices only.
- Auto-tier minimums: big = 24 GB RAM, 12 GB NVIDIA VRAM, 42 GB free disk; mid = 12 GB RAM, 4 GB NVIDIA VRAM, 18 GB free disk; small is the fallback with 3 GB free disk.
- Disk minimums are the rounded-up 1.5x size of each tier's required model payloads. Embeddings remain `all-mpnet-base-v2` for every tier.
- Evaluation can isolate its data with `SMART_FILE_FINDER_EVAL_DATA_DIR`; the comparison runner sets it only in worker processes, leaving normal app runs unchanged.
- Small-tier comparison favors SmolVLM-500M over 256M on this machine: higher image-case top-1/shown hit rates (60% vs 40%) with similar peak RAM, despite slower image processing (15.15 vs 10.93 seconds/image).
- Gemma E4B is registered as untested; it has not been downloaded or run.
- Feature 3: voice search, English-only initially, using local Whisper.
