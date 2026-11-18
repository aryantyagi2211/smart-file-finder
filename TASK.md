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

- Feature 3: voice search, English-only initially, using local Whisper.
- Final repackaging of the executable.
- Rebuild of the installer.
- README update for the completed update round.

## Current status

Feature 1 is implemented and validated with confidence threshold `0.55`, ambiguity margin `0.07`, options floor `0.48`, maximum options `3`, minimum additional-card score `0.50`, and option score window `0.03`. Both confident and uncertain results use the compact reference card style; confident cards and the 1-3-card uncertain stack are movable. The full desktop UI has not yet been visually verified interactively. Device-tier steps 1 and 2 are implemented; model setup requires explicit consent and the app runs offline after required model files are local.

## Device tiers

1. **Hardware detection and tier choice — Done.** Detect RAM, disk, CPU, architecture, OS and NVIDIA GPU; select or override small/mid/big. Keep embeddings fixed at `all-mpnet-base-v2`.
2. **Model registry and first-run downloader — Done.** Pin public model revisions, check local model files, prompt before downloads, show byte progress with cancel/retry, and keep the app offline after setup. If CUDA is unavailable or no vision model is configured for the tier, index images by filename only.
3. **Vision model per tier — Open.** Choose and wire a vision model for each tier.
4. **Whisper size per tier — Open.** Apply the noted Whisper model size when voice search is implemented.
