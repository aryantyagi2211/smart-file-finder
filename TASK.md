# Smart File Finder Task Status

## Completed fixes

- Fixed stale indexing by removing old vectors and metadata when files change, are deleted, or are moved, and by cleaning up missing files on startup.
- Fixed embedding-failure handling so incomplete indexes are removed, failed files can be retried, and repeated failures are temporarily skipped.
- Fixed packaged configuration lookup to resolve settings beside the executable and create missing configuration files with defaults.
- Enforced a hard per-query deadline for retrieval, corrective checks, reasoning, and explanations.
- Implemented fast, rule-based query refinement and stopped repeated identical searches.
- Normalized indexed paths, made cleanup safe across working directories, fixed integration-test teardown, and made evaluation index its own corpus.
- Made the startup shortcut an unchecked, optional installer task.

## Golden set

- Expanded the evaluation corpus to 20 cases with text, document, image, ambiguous, and no-match queries.
- Evaluator supports acceptable files, top-three reporting, and confidence-threshold sweeps.

## Feature 1 — Top-three pick list

Status: **Done** — unit and integration tests, the golden-set evaluation, three CLI checks, and an offscreen pick-list interaction check have passed.

- Confident searches return one file without a confirmation popup.
- The confident result uses the compact reference card style and is movable by dragging its non-button surface.
- Uncertain searches show a vertical left-side stack of 1-3 independently movable popup cards, filtered by relevance, with ranks, scores, explanations, folder paths, and Open / Show actions.
- Queries below the options floor return NO_MATCH.
- Decision thresholds: confidence `0.55`, ambiguity margin `0.07`, options floor `0.48`, maximum options `3`.
- Additional popup cards require a score of at least `0.50` and must be within `0.03` of the best score.
- `/open` does not automatically open an uncertain result.

## Open work

- Real Left Alt press/release, focus behavior in other apps, microphone, and accent verification of voice search is still needed.
- Final repackaging of the executable.
- Rebuild of the installer.
- README update for the completed update round.

## Current status

Feature 1 is implemented and validated with confidence threshold `0.55`, ambiguity margin `0.07`, options floor `0.48`, maximum options `3`, minimum additional-card score `0.50`, and option score window `0.03`. Both confident and uncertain results use the compact reference card style; confident cards and the 1-3-card uncertain stack are movable. The full desktop UI has not yet been visually verified interactively. Device-tier steps 1-3b are implemented; model setup requires explicit consent and the app runs offline after required model files are local. On the first launch, the status panel reports PC-spec detection once, then shows Low/Mid/High and named model download/loading stages before and during indexing. Live changes return to “Watching for changes” when complete. Gemma E4B remains untested and is not included in the standard setup.

## Device tiers

1. **Hardware detection and tier choice — Done.** Detect RAM, disk, CPU, architecture, OS and NVIDIA GPU; select the highest eligible Low/Mid/High (internal small/mid/big) tier using RAM, NVIDIA VRAM, and disk requirements. Manual settings cannot exceed detected requirements. Keep embeddings fixed at `all-mpnet-base-v2`.
2. **Model registry and first-run downloader — Done.** Pin public model revisions, check local model files, prompt before downloads, show byte progress with cancel/retry, and keep the app offline after setup. If CUDA is unavailable or no vision model is configured for the tier, index images by filename only.
3. **Vision model per tier — Done.** Small uses SmolVLM-500M (chosen over 256M from the isolated image comparison); mid and big use Gemma E2B 4-bit by default. Big can opt into Gemma E4B, which is untested. Image description falls back only to already-downloaded lower-tier models, unloads after each scan, and records the model in metadata. Embeddings remain fixed at `all-mpnet-base-v2`.
3b. **Required model download policy — Done.** Each tier requires its embedding, primary vision, and voice model in the same consent-gated first-run setup. Gemma E4B is untested and is not part of the standard High setup. The comparison runner's `--include-e4b` option is opt-in and requires 24 GB RAM, 12 GB NVIDIA VRAM, and 28 GB free disk; it verifies E4B was actually used rather than silently benchmarking a fallback. Default free-disk thresholds are 3 GB (Low), 19 GB (Mid), and 19 GB (High); selecting E4B raises High to 28 GB with the bundled voice model. Missing vision fallbacks use filenames without starting a download.
4. **Whisper size per tier — Done.** Low uses faster-whisper small.en (CPU int8, 486 MB). Mid and High use faster-whisper large-v3-turbo (1,622 MB) from the pinned CTranslate2 snapshot; other Whisper sizes remain registry options, not automatic tier downloads. Mid/High try CTranslate2 CUDA with int8_float16 and a warm-up decode, then fall back to the same local files on CPU int8 if CUDA fails. CUDA decode failures retry once on CPU; no fallback downloads another model. The model stays lazy-loaded and unloads after five idle minutes. A one-time tray offer to review and consent to small.en is shown if CPU turbo exceeds 8 seconds for a 5-second clip.

Required model download payload totals (decimal GB): Low 1.945 GB, Mid 12.339 GB, High 12.339 GB. The Mid/High totals include the shared 10.279 GB vision model and 1.622 GB turbo voice model. High's optional E4B setup totals 18.085 GB.

GTX 1650 local test: CTranslate2 detected one CUDA device, but CUDA load failed because `cublas64_12.dll` was missing; no turbo CUDA timing or model VRAM peak is available. The separate CPU int8 run loaded in 3.72 s and transcribed five synthetic Windows TTS clips (3.99–4.46 s) in 12.16, 12.04, 12.13, 12.95, and 12.64 s; peak process RAM was 1,731 MiB. These generated clips are not real speech. CTranslate2's Windows GPU runtime requires CUDA 12.x and cuDNN 8 for speech models; the installed PyTorch cu130 runtime does not provide the missing CUDA 12 cuBLAS DLL.

## Feature 3 - Alt-hold voice search

Status: **Implemented; real-device checks remain** — holding the configured Alt key alone starts an in-memory recording and release transcribes locally. The floating microphone overlay is click-through and non-focusing, and shows the transcript before the voice-only parser acts. Voice transcripts can start with `open`, `show`, `find`, or `search`; typed text continues to require `/open`, `/show`, or `/find`, with plain typed text remaining a find. `Ctrl+Shift+Space` remains a tap-only typing-bar shortcut.

- Settings: `voice_key` defaults to `"left alt"` (`"right alt"` can act as AltGr on some layouts), `voice_hold_ms` defaults to `400`, and `voice_auto_search` defaults to `true`. When auto-search is false, the transcript opens in the typing bar for editing and Enter.
- A lone spoken `open` does not search; the overlay asks what to open. OPTIONS results for voice `open` remain choices and are not opened automatically.
- faster-whisper remains English-only: Low uses small.en on CPU/int8, while Mid and High use large-v3-turbo and try CUDA/int8_float16 before falling back to the same local files on CPU/int8. Normal voice audio stays in memory, is not written or sent, and is cleared after transcription.
- Real Alt key behavior across apps, menu activation, microphone availability, and accents still require manual verification.
- Fixed the Alt-hold suppression event being mistaken for another key: the Windows keyboard hook reports injected VK `0xE8` as an unnamed scan code `-232`; INFO voice-stage logs now trace hook, recorder, overlay, transcription, and command dispatch.
- Fixed real Windows Alt auto-repeat cancelling its own recording: repeated key-down events can arrive with `is_repeat=False`, so the hook now distinguishes an already-down key from a newly pressed cancel key. The integration test reproduces this event sequence; ordinary printable key names are redacted in voice logs.
- Typed slash commands and spoken commands now share `_dispatch_command`: confident open launches the absolute file path, confident show selects the file in Explorer, and find keeps the result card. OPTIONS only shows cards and voice feedback; NO_MATCH keeps its normal message.
- Spoken command parsing tolerates surrounding whitespace, capitalization, command punctuation, and sentence-ending punctuation. Voice feedback reports “Opening <file name>”, “Showing <file name> in Explorer”, “Not sure which file - pick one”, or “No match found”; every overlay state has a logged 10-second safety hide.
- The voice overlay now aligns to the input bar's shared position calculation, including a remembered bar position, clamped within the cursor screen's taskbar-excluded available area. The typing bar's own placement and drag behavior are unchanged.

## Voice accuracy upgrade

- Pre-roll opens the microphone stream at the configured voice-key down and keeps the audio in memory. Confirmed holds retain audio from key-down; short taps and shortcuts discard it and close the stream. Windows may briefly show its microphone indicator on short Alt presses while pre-roll is enabled.
- Recording continues for `voice_tail_ms` (default `300`) after release. Defaults are `voice_preroll: true` and `voice_hold_ms: 400`.
- Final decoding uses English, beam size 5, temperature 0, no previous-clip conditioning, and no VAD for these clips (maximum recording length 15 seconds). Audio is trimmed at near-digital-silence boundaries and peak-normalized to about -3 dBFS without clipping. `voice_initial_prompt` and `voice_hotwords` are configurable; faster-whisper 1.2.1 supports hotwords.
- Voice-only parsing accepts the configured first-word aliases and edit-distance-one command variants when followed by a query (for example, “So corrective rag” means `show`). Recognized transcript text remains unchanged in the overlay. Typed text and slash-command parsing are unaffected.
- `python -m app.voice_lab` records 14 user-spoken phrases to the local app-data `voice_lab` folder. `python -m app.voice_lab --eval` compares public faster-whisper candidates and an experimental Transformers/PyTorch GPU candidate; it verifies Hugging Face metadata and asks before each missing-model download. `python -m app.voice_lab --live` exercises the real Alt-hold pipeline.
- Voice Lab audio is never uploaded and stays in app data; generated reports contain transcripts and metrics only, not audio. `voice_lab/` and `eval/voice_report.md` are ignored so private recordings and reports are not committed.

## Model cleanup

- `python -m app.model_cleanup` inventories app-local and Hugging Face cached model folders as a dry run. It reports repo id, disk size, path, and IN USE / CANDIDATE / SUPERSEDED / EXTERNAL status.
- `--apply` asks default-no confirmation per complete cache folder. IN USE models are never deletable; EXTERNAL shared-cache models warn that other projects may use them. Gemma 4 E2B stays protected.
- `python -m app.model_cleanup --voice-lab-recordings` reports the local Voice Lab recordings folder; removal requires `--apply` and explicit confirmation. Only dry-run inventories were performed during implementation; no model or recording folders were deleted.
