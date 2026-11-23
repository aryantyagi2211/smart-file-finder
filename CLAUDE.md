# Smart File Finder

## Locked decisions

- Confident search results show one file.
- Confident result cards use the compact reference style and can be moved by dragging the card surface.
- Not-sure results show a vertical stack of 1-3 separate, movable popup cards on the left side of the cursor/input-bar screen, selected by relevance.
- The best file is always shown; additional cards must score at least `0.50` and be within `0.03` of the best score.
- There is no low-confidence confirmation popup.
- An uncertain result is never opened automatically, including for `/open`.
- The app targets Windows PCs generally, including CPU-only machines; Snapdragon NPU support is a later enhancement.
- Typed input keeps slash-command parsing (`/open`, `/show`, `/find`); plain typed text remains a find. Voice transcripts alone use spoken `open`, `show`, `find`, or `search` commands without symbols.
- `Ctrl+Shift+Space` remains a tap-only typing-bar shortcut. Voice starts by holding the configured `voice_key` alone (default `left alt`) for `voice_hold_ms` (default 500 ms); `right alt` may act as AltGr on some layouts. Modifier combinations and short Alt taps must not start voice.
- The Alt voice flow uses a click-through, non-focusing floating microphone overlay. It displays the transcript before acting; `voice_auto_search` defaults to `true`, and `false` opens the transcript in the typing bar for editing.
- Typed slash commands and spoken commands share `_dispatch_command`. A confident `open` opens the absolute file path; a confident `show` selects the file in Explorer; `find` keeps the standard result card. OPTIONS shows cards without opening or revealing a file, and NO_MATCH keeps the normal no-match message.
- Spoken command parsing ignores surrounding whitespace and case, strips punctuation around the command word and at the end of a transcript, and removes leading filler words. Voice feedback uses “Opening <file name>”, “Showing <file name> in Explorer”, “Not sure which file - pick one”, and “No match found”. Every overlay state is protected by a logged 10-second safety hide.
- The voice overlay uses the input bar's shared position calculation: centered over the bar, with its bottom aligned to the bar's bottom, on the cursor's screen and within that screen's taskbar-excluded available area. A remembered dragged bar position is reused and clamped for the overlay; the typing bar's own behavior is unchanged.

## Models and search thresholds

- Embeddings use `all-mpnet-base-v2` (768 dimensions, local CPU).
- Image understanding runs only while indexing: SmolVLM-500M on small and Gemma E2B 4-bit on mid and big by default. Big may opt into Gemma E4B 4-bit with `big_vision_model: "gemma-4-E4B-it"`; E4B is untested.
- `Qwen/Qwen3-1.7B` is used by the indexing-time summarizer only.
- There is no live LLM reasoning in the search path; query refinement and result decisions are rule-based.
- Feature 3 Alt-hold voice search uses faster-whisper English-only CPU int8 models: tiny.en on Low/small, base.en on Mid, small.en on High/big. The selected voice model is a required part of the same first-run model setup and consent as embeddings and vision. Audio stays in memory only, is never written to disk or sent anywhere, and is discarded immediately after transcription. The only network use remains one-time public model downloads from Hugging Face.
- Voice transcription is lazy-loaded, uses fixed language `en`, small beam search and VAD filtering, and unloads after five idle minutes. Transcription time is outside the five-second search budget.
- The only network use is downloading public model files from Hugging Face during explicit first-run setup. No user file, query, or telemetry is ever sent. After setup the app works fully offline.
- `config/model_tiers.json` is the model registry. Each tier downloads its embedding model, exactly one primary vision model, and one Whisper model in the same setup: small uses SmolVLM-500M; mid and big currently use Gemma E2B by default. Gemma E4B is in `optional_models`, is untested, and is not included in standard setup pending a compatible-hardware benchmark. Download only the loader-required files pinned to each model revision; `download_size_gb` is that selected payload, while `repository_size_gb` records the full repository size.
- Models are stored under the app data folder's `models` directory. Resolve models with `app_paths.find_model`; enable Hugging Face offline mode only after required models are local.
- Vision fallback order is big -> mid -> small -> filename-only; CUDA-required models are skipped without CUDA. A lower-tier model is used only if its pinned snapshot is already local. If it is absent, use the filename, show `Fallback model not downloaded - using file names only for now`, and never trigger a download from indexing. Small uses SmolVLM on CPU or GPU without bitsandbytes.
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
- Auto-tier minimums: High/big = 24 GB RAM and 12 GB NVIDIA VRAM; Mid = 12 GB RAM and 4 GB NVIDIA VRAM; Low/small is the fallback. Selection chooses the highest tier meeting RAM, VRAM, and free-disk requirements; manual settings cannot exceed detected requirements. Free-disk minimums are rounded-up 1.5x the embedding, vision, and voice payloads: Low = 3 GB, Mid = 17 GB, High = 17 GB by default (26 GB with E4B).
- Startup status reads PC specifications on the first run only, then shows the selected Low/Mid/High tier. Required model downloads are named one at a time; model-loading status distinguishes embedding, lazy vision, and lazy voice loads from indexing. Live file changes show indexing activity and return to "Watching for changes".
- Disk minimums are the rounded-up 1.5x size of each tier's required embedding, vision, and voice model payloads. Embeddings remain `all-mpnet-base-v2` for every tier.
- Evaluation can isolate its data with `SMART_FILE_FINDER_EVAL_DATA_DIR`; the comparison runner sets it only in worker processes, leaving normal app runs unchanged.
- Small-tier comparison favors SmolVLM-500M over 256M on this machine: higher image-case top-1/shown hit rates (60% vs 40%) with similar peak RAM, despite slower image processing (15.15 vs 10.93 seconds/image).
- Gemma E4B is registered as optional and untested; it is not downloaded by default.
- Feature 3: voice search, English-only initially, using local Whisper.
