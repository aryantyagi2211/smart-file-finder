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
- `Ctrl+Shift+Space` remains a tap-only typing-bar shortcut. Voice starts by holding the configured `voice_key` alone (default `left alt`) for `voice_hold_ms` (default 400 ms); `right alt` may act as AltGr on some layouts. Modifier combinations and short Alt taps must not start voice.
- With `voice_preroll: true`, open the microphone stream at key-down and keep captured samples only in memory. Retain them only after a confirmed hold; early release or another key cancels and discards them. Windows may briefly show its microphone indicator on short Alt presses.
- Continue recording for `voice_tail_ms` (default 300 ms) after release. Keep audio in memory only during normal voice search; never write it to disk.
- The Alt voice flow uses a click-through, non-focusing floating microphone overlay. It displays the transcript before acting; `voice_auto_search` defaults to `true`, and `false` opens the transcript in the typing bar for editing.
- Typed slash commands and spoken commands share `_dispatch_command`. A confident `open` opens the absolute file path; a confident `show` selects the file in Explorer; `find` keeps the standard result card. OPTIONS shows cards without opening or revealing a file, and NO_MATCH keeps the normal no-match message.
- Spoken command parsing ignores surrounding whitespace and case, strips punctuation around the command word and at the end of a transcript, and removes leading filler words. Voice feedback uses “Opening <file name>”, “Showing <file name> in Explorer”, “Not sure which file - pick one”, and “No match found”. Every overlay state is protected by a logged 10-second safety hide.
- The voice overlay uses the input bar's shared position calculation: centered over the bar, with its bottom aligned to the bar's bottom, on the cursor's screen and within that screen's taskbar-excluded available area. A remembered dragged bar position is reused and clamped for the overlay; the typing bar's own behavior is unchanged.
- Final faster-whisper decoding uses English, beam size 5, temperature 0, `condition_on_previous_text=False`, and `vad_filter=False` for the app's clips (up to 15 seconds). Trim near-digital silence and normalize peak amplitude to -3 dBFS without clipping. `voice_initial_prompt` and `voice_hotwords` are configurable; pass hotwords only when the installed faster-whisper supports them.
- Voice-only parsing uses configurable aliases for the first command word and can accept edit-distance-one variants only when a query follows. The default aliases include show: `show`, `sho`, `so`, `shoe`, `shaw`; open: `open`, `opan`; find: `find`, `fine`, `fined`. Never apply voice aliases to typed input and preserve the original transcript for overlay display.
- Voice Lab is `python -m app.voice_lab` (record), `--eval` (consent-gated model comparison), and `--live` (the real Alt pipeline). Model candidates are not tier defaults; do not change tier-to-Whisper assignments for Voice Lab work. Recordings remain in app data, are never uploaded, and must not be committed. Reports contain transcripts/metrics only; `.gitignore` excludes `voice_lab/` and `eval/voice_report.md`.
- `python -m app.model_cleanup` inventories app-local and Hugging Face cache models as a dry run. `scan_cache_dir()` provides real cache sizes (not snapshot-link sizes), revision metadata, and deletion estimates; inventory flags registry-size mismatches over 10% and lists incomplete revisions. Model status is independent of cache location: IN USE, FALLBACK, OTHER TIER (with assigned tier), CANDIDATE, SUPERSEDED, or EXTERNAL (not in the registry). Confirmed cache deletion uses `delete_revisions()` and verifies freed space with a second scan. IN USE models are never deletable; FALLBACK requires a specific warning and explicit default-no confirmation; OTHER TIER requires the warning that it is needed if this PC's tier changes or for users of that tier. Keep Gemma 4 E2B protected while assigned to a tier. The superseded registry list contains only obsolete former defaults tiny.en and base.en, not Low-tier small.en or SmolVLM-500M. `--voice-lab-recordings` only targets the Voice Lab recordings folder and also requires `--apply` plus confirmation.

## Models and search thresholds

- Embeddings use `all-mpnet-base-v2` (768 dimensions, local CPU).
- Image understanding runs only while indexing: SmolVLM-500M on small and Gemma E2B 4-bit on mid and big by default. Big may opt into Gemma E4B 4-bit with `big_vision_model: "gemma-4-E4B-it"`; E4B is untested.
- `Qwen/Qwen3-1.7B` is used by the indexing-time summarizer only.
- There is no live LLM reasoning in the search path; query refinement and result decisions are rule-based.
- Feature 3 Alt-hold voice search uses faster-whisper small.en (CPU int8, 486 MB) on Low/small, and large-v3-turbo (1,622 MB) on Mid and High/big. The pinned CTranslate2 repo is `dropbox-dash/faster-whisper-large-v3-turbo` (MIT, public, ungated). The selected voice model is required in the same consent-gated setup as embeddings and vision. Other Whisper sizes remain registry options, not automatic tier downloads. Normal voice-search audio stays in memory only, is never written to disk or sent anywhere, and is discarded immediately after transcription.
- Mid/High lazily try CTranslate2 CUDA with `int8_float16`, after checking `ctranslate2.get_cuda_device_count()` and performing a tiny warm-up decode. CUDA initialization/decode failures use the same local files on CPU `int8`; a CUDA decode failure retries that clip once on CPU. No fallback downloads another model. Low always uses CPU. Voice models unload after five idle minutes. If CPU turbo exceeds 8 seconds for a 5-second clip, show a one-time tray offer to review the small.en download size and explicitly consent before downloading and switching.
- On the GTX 1650 test PC, CTranslate2 detects one CUDA device but turbo CUDA fails because `cublas64_12.dll` is missing. This installation needs the CUDA 12.x runtime and cuDNN 8 for CTranslate2 speech; PyTorch cu130 alone does not supply that DLL. CPU int8 loaded in 3.72 seconds; five synthetic Windows TTS clips (3.99–4.46 seconds, not real speech) took 12.16, 12.04, 12.13, 12.95, and 12.64 seconds, with 1,731 MiB peak process RAM. No CUDA model VRAM peak was measured because the model did not load on the GPU.
- The app downloads public model files from Hugging Face only during explicit first-run setup or, for Voice Lab comparisons, after separate explicit consent. No user file, query, audio, or telemetry is ever sent. After the selected models are downloaded, the app works fully offline.
- `config/model_tiers.json` is the model registry. Each tier downloads its embedding model, exactly one primary vision model, and one voice model in the same setup: small uses SmolVLM-500M; mid and big use Gemma E2B and large-v3-turbo by default. Required payload totals are 1.945 GB Low, 12.339 GB Mid, and 12.339 GB High; standard free-disk minimums are 3 GB, 19 GB, and 19 GB respectively. High's optional E4B setup totals 18.085 GB and needs 28 GB free. Gemma E4B is untested and is not included in standard setup pending a compatible-hardware benchmark. Download only the loader-required files pinned to each model revision; `download_size_gb` is that selected payload, while `repository_size_gb` records the full repository size.
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
- Auto-tier minimums: High/big = 24 GB RAM and 12 GB NVIDIA VRAM; Mid = 12 GB RAM and 4 GB NVIDIA VRAM; Low/small is the fallback. Selection chooses the highest tier meeting RAM, VRAM, and free-disk requirements; manual settings cannot exceed detected requirements. Free-disk minimums are rounded-up 1.5x the embedding, vision, and voice payloads: Low = 3 GB, Mid = 19 GB, High = 19 GB by default (28 GB with E4B).
- Startup status reads PC specifications on the first run only, then shows the selected Low/Mid/High tier. Required model downloads are named one at a time; model-loading status distinguishes embedding, lazy vision, and lazy voice loads from indexing. Live file changes show indexing activity and return to "Watching for changes".
- Disk minimums are the rounded-up 1.5x size of each tier's required embedding, vision, and voice model payloads. Embeddings remain `all-mpnet-base-v2` for every tier.
- Evaluation can isolate its data with `SMART_FILE_FINDER_EVAL_DATA_DIR`; the comparison runner sets it only in worker processes, leaving normal app runs unchanged.
- Small-tier comparison favors SmolVLM-500M over 256M on this machine: higher image-case top-1/shown hit rates (60% vs 40%) with similar peak RAM, despite slower image processing (15.15 vs 10.93 seconds/image).
- Gemma E4B is registered as optional and untested; it is not downloaded by default.
- Feature 3: voice search, English-only initially, using local Whisper.
