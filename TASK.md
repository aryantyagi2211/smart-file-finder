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

Feature 1 is implemented and validated with confidence threshold `0.55`, ambiguity margin `0.07`, options floor `0.48`, maximum options `3`, minimum additional-card score `0.50`, and option score window `0.03`. Both confident and uncertain results use the compact reference card style; confident cards and the 1-3-card uncertain stack are movable. The full desktop UI has not yet been visually verified interactively. Device-tier steps 1-3b are implemented; model setup requires explicit consent and the app runs offline after required model files are local. Gemma E4B is optional, untested, and not downloaded by default.

## Device tiers

1. **Hardware detection and tier choice — Done.** Detect RAM, disk, CPU, architecture, OS and NVIDIA GPU; select or override small/mid/big. Keep embeddings fixed at `all-mpnet-base-v2`.
2. **Model registry and first-run downloader — Done.** Pin public model revisions, check local model files, prompt before downloads, show byte progress with cancel/retry, and keep the app offline after setup. If CUDA is unavailable or no vision model is configured for the tier, index images by filename only.
3. **Vision model per tier — Done.** Small uses SmolVLM-500M (chosen over 256M from the isolated image comparison); mid and big use Gemma E2B 4-bit by default. Big can opt into Gemma E4B, which is untested. Image description falls back only to already-downloaded lower-tier models, unloads after each scan, and records the model in metadata. Embeddings remain fixed at `all-mpnet-base-v2`.
3b. **Primary-only model download policy — Done.** Each tier requires the embedding model and only its primary vision model. Gemma E4B is optional and enabled with `big_vision_model: "gemma-4-E4B-it"`; it is never downloaded by default. Default free-disk thresholds are 3 GB (small), 17 GB (mid), and 17 GB (big); selecting E4B raises big to 25 GB. Missing fallbacks use filenames without starting a download.
4. **Whisper size per tier — Done.** Small uses faster-whisper tiny.en, mid uses base.en, and big uses small.en; all run on CPU with int8. The models are optional consent-gated downloads. Synthetic TTS clips passed; a real microphone/real accent check remains for Aryan.

## Feature 3 - Alt-hold voice search

Status: **Implemented; real-device checks remain** — holding the configured Alt key alone starts an in-memory recording and release transcribes locally. The floating microphone overlay is click-through and non-focusing, and shows the transcript before the voice-only parser acts. Voice transcripts can start with `open`, `show`, `find`, or `search`; typed text continues to require `/open`, `/show`, or `/find`, with plain typed text remaining a find. `Ctrl+Shift+Space` remains a tap-only typing-bar shortcut.

- Settings: `voice_key` defaults to `"left alt"` (`"right alt"` can act as AltGr on some layouts), `voice_hold_ms` defaults to `500`, and `voice_auto_search` defaults to `true`. When auto-search is false, the transcript opens in the typing bar for editing and Enter.
- A lone spoken `open` does not search; the overlay asks what to open. OPTIONS results for voice `open` remain choices and are not opened automatically.
- faster-whisper remains English-only and CPU/int8 by tier. The optional model download and privacy behavior is unchanged: audio stays in memory, is not written or sent, and is cleared after transcription.
- Real Alt key behavior across apps, menu activation, microphone availability, and accents still require manual verification.
- Fixed the Alt-hold suppression event being mistaken for another key: the Windows keyboard hook reports injected VK `0xE8` as an unnamed scan code `-232`; INFO voice-stage logs now trace hook, recorder, overlay, transcription, and command dispatch.
- Fixed real Windows Alt auto-repeat cancelling its own recording: repeated key-down events can arrive with `is_repeat=False`, so the hook now distinguishes an already-down key from a newly pressed cancel key. The integration test reproduces this event sequence; ordinary printable key names are redacted in voice logs.
