## Project Walkthrough

<details open>
<summary>01 — Smart File Finder</summary>

<p align="center">
<img src="https://github.com/user-attachments/assets/927ea47b-1802-4086-b125-ca124cf5e5a5" width="100%">
</p>

</details>

<details>
<summary>02 — Search Only Works If You Remember Exactly</summary>

<p align="center">
<img src="https://github.com/user-attachments/assets/ebc15758-9a89-472a-a84f-ad2a92e32436" width="100%">
</p>

</details>

<details>
<summary>03 — Describe It Naturally. Find It Instantly. Stay Private.</summary>

<p align="center">
<img src="https://github.com/user-attachments/assets/0eb7dae0-8d91-42b4-9a34-4d2a5acc6df5" width="100%">
</p>

</details>

<details>
<summary>04 — A Retrieval Pipeline That Checks Its Own Confidence</summary>

<p align="center">
<img src="https://github.com/user-attachments/assets/46e8de5c-0fd8-4b31-81d2-74a4cac046a7" width="100%">
</p>

</details>

<details>
<summary>05 — Local Models. Local Storage. No External Calls.</summary>

<p align="center">
<img src="https://github.com/user-attachments/assets/7fb76e59-33c1-4fe0-a15a-b17b9dbe9db0" width="100%">
</p>

</details>

<details>
<summary>06 — Built for Daily Use. Not Just a Demo.</summary>

<p align="center">
<img src="https://github.com/user-attachments/assets/28139ad4-baf0-403e-997d-f3e5b6f816e9" width="100%">
</p>

</details>

<details>
<summary>07 — Measured. Not Assumed.</summary>

<p align="center">
<img src="YOUR_SLIDE_7_IMAGE_URL" width="100%">
</p>

</details>

<details>
<summary>08 — What's Next</summary>

<p align="center">
<img src="https://github.com/user-attachments/assets/a2e324d4-7791-493e-9999-ad9b2d6e8d3d" width="100%">
</p>

</details>


# Smart File Finder

A desktop app that finds files by what's actually *inside* them — including
screenshots and images that were never named or tagged — using plain
natural language, entirely on your own device. No cloud calls, no
uploads: every model runs locally, so your files and your searches never
leave your machine.

Press a hotkey, type what you're looking for the way you'd describe it to
a person, and get back the actual file — with a short explanation of why
it matched.

Smart File Finder is for Windows PCs generally, including CPU-only
machines; Snapdragon NPU support is a future enhancement, not a requirement.

---

## Why this exists

Traditional file search only matches filenames or exact text. If you
remember what a document was *about* but not what it was *called*, or
you're trying to find a screenshot by describing what's in the picture,
normal search comes up empty. Smart File Finder indexes the real content
of your files — text, images, and metadata — and lets you search that
content the way you'd naturally describe it.

## Key features

- **Natural language search** — describe what you're looking for; you
  don't need the exact filename or exact wording from the document.
- **Real image understanding** — photos and screenshots are described by
  a local vision model at indexing time, so you can find an image by
  describing what it shows.
- **Fully local and private** — every model (embeddings, vision) runs on
  your own CPU/GPU. Nothing is uploaded anywhere, ever.
- **Hardware-matched model downloads** — the app detects a performance
  tier before model setup, tells you which tier and model files it selected,
  and downloads only the required files for that tier.
- **Global hotkey launcher** — `Ctrl+Shift+Space` opens a small, always-on-top
  search bar from anywhere in Windows, similar to Spotlight or PowerToys Run.
- **Hold-to-talk voice search** — hold Left Alt by itself to speak a query.
  Spoken `open`, `show`, `find`, and `search` commands work without a slash;
  a floating, click-through overlay shows listening state and a live,
  continually updated partial transcript while you speak. Pre-roll captures
  from key-down and keeps audio in memory; short Alt presses may briefly show
  the Windows microphone indicator.
- **Live, incremental indexing** — a background watcher picks up new and
  changed files automatically while the app runs; files that haven't
  changed are never re-processed.
- **Broad file support** — PDF, Word (`.docx`), Markdown, plain text, CSV,
  and common image formats (PNG, JPG, WEBP, etc.).
- **Confidence-aware results** — low-confidence matches ask for
  confirmation before opening, instead of guessing silently.
- **Runs quietly in the background** — a system tray icon, no visible
  console window, and optional auto-start on login.

## How it works

At a high level, the app is built around a retrieval-augmented pipeline:

1. **Indexing** — files in your chosen folders (Desktop, Documents,
   Downloads, or any folder you add) are scanned. Text is extracted and
   split into chunks; images are described by a local vision-language
   model. Everything is turned into vector embeddings and stored in a
   local vector database, alongside a lightweight metadata index.
2. **Retrieval** — when you search, your query is embedded the same way,
   and the vector database returns the most semantically similar content
   — whether that's a paragraph from a PDF, a filename, or an image
   description.
3. **Confidence checking** — matches below a confidence threshold are
   flagged, so you're never silently handed an irrelevant result; a
   reformulation step can broaden an overly narrow query automatically.
4. **Explanation** — the final result comes with a short, human-readable
   explanation of why it matched, built from the actual matched content.

The retrieval strategy combines ideas from adaptive, corrective, and
agentic retrieval-augmented generation: queries are checked for
confidence before being trusted, and reformulated automatically if the
first attempt comes back too weak or too narrow.

## Tech stack

- **UI**: PyQt6 (custom-styled, frameless windows)
- **Embeddings**: `sentence-transformers` (`all-mpnet-base-v2`)
- **Vision**: a locally-run, 4-bit quantized vision-language model
  (Gemma), invoked only during indexing — never at search time, to keep
  search itself fast
- **Vector storage**: ChromaDB (local, persistent)
- **Metadata storage**: SQLite
- **File watching**: `watchdog`
- **Packaging**: PyInstaller + Inno Setup (Windows installer)

## Installation

### Option 1 — Windows installer (recommended)

Download and run `SmartFileFinderSetup.exe` from the latest release. This
installs the app and creates a Start Menu entry. The installer offers an
optional **Start Smart File Finder when Windows starts** checkbox; it is
off by default. To disable auto-start later, delete the Smart File Finder
shortcut from `shell:startup` or disable it in **Task Manager → Startup apps**.

> **Note:** since this build isn't code-signed, Windows may flag the
> Startup entry as "disabled" if you opted in — if the app doesn't auto-start
> after a restart, open **Task Manager → Startup apps** and enable
> "SmartFileFinder" manually. This is a one-time step.

### Performance tiers and model downloads

Smart File Finder supports Windows PCs with or without an NVIDIA GPU. Before
model setup, it checks available RAM, NVIDIA GPU memory, and free disk space,
selects a performance tier, and shows the tier and the exact missing files
and download size. The selected tier is also shown in the indexing status
and system-tray menu.

Automatic selection chooses the highest tier whose RAM, NVIDIA VRAM, and free
disk minimums are all met: High (internal Big) requires 24 GB RAM and 12 GB
NVIDIA VRAM; Mid requires 12 GB RAM and 4 GB NVIDIA VRAM; otherwise the app
uses Low (internal Small). A manual tier setting cannot exceed the detected
hardware and disk requirements. CPU-only PCs are supported by Low.

All tiers use the same `all-mpnet-base-v2` embedding model, so tier selection
does not invalidate an existing vector index. Each tier downloads one primary
vision model:

| Tier | Models downloaded during first setup | Approx. total download | Minimum free disk |
|---|---|---:|---:|
| Low (Small) | `all-mpnet-base-v2` (0.44 GB) + SmolVLM-500M (1.02 GB) + Whisper small.en (486 MB) | 1.945 GB | 3 GB |
| Mid | `all-mpnet-base-v2` (0.44 GB) + Gemma 4 E2B-it 4-bit (10.28 GB) + Whisper large-v3-turbo (1.622 GB) | 12.339 GB | 19 GB |
| High (Big) | `all-mpnet-base-v2` (0.44 GB) + Gemma 4 E2B-it 4-bit (10.28 GB) + Whisper large-v3-turbo (1.622 GB) | 12.339 GB | 19 GB |

These are model payload sizes; additional free space is required for download
and setup. On the first run, the status panel briefly reports that it is
reading PC specifications, then shows the selected Low, Mid, or High tier.
Later runs show the saved tier immediately while checking the current setup.
During setup, each model download is named in sequence; after confirmation,
the embedding, vision, and voice models are downloaded in one setup flow.
During startup, the status distinguishes embedding, vision, and voice model
loading from indexing progress. Vision loads when an image is indexed; voice
loads on the first voice search. Low runs Whisper small.en on CPU int8. Mid
and High try large-v3-turbo on CUDA int8_float16 after a device check and warm
up; failures fall back to the same local files on CPU int8 without downloading
another model. A CUDA out-of-memory decode retries once on CPU. Models unload
after five idle minutes. If CPU turbo takes longer than 8 seconds for a
5-second clip, the tray offers a consent-gated small.en alternative (486 MB).
Small-tier image understanding runs on CPU; Mid and Big use the CUDA-capable
Gemma vision model. Big currently uses the same E2B model as Mid; the stronger
E4B model remains untested and is not included in the standard setup. If
selected for Big, E4B adds about 16.02 GB and raises the free-disk minimum to
28 GB.

### Option 2 — Run from source

Requires Python 3.10. A CUDA-capable NVIDIA GPU can improve image indexing;
CPU-only Windows PCs are supported and select the small tier.

```bash
git clone <repository-url>
cd smart-file-finder
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -m app.main
```

The first run will download the embedding and vision models (a few GB
total) and perform an initial scan of your Desktop, Documents, and
Downloads folders. This first scan can take a while depending on how
many files you have — subsequent runs only process new or changed files.

## Usage

- **`Ctrl+Shift+Space`** — open the search bar (drag it anywhere; it
  remembers its position). A tap opens the bar; holding it does not start voice.
- **Hold `Left Alt`** — start voice search. The default `voice_key` is
  `"left alt"`; `"right alt"` is configurable but can act as AltGr on some
  keyboard layouts. A short Alt tap does not start a search, though pre-roll
  may briefly activate the Windows microphone indicator.
- Spoken `open`, `show`, `find`, or `search` at the start of a transcript
  selects that action; voice-only aliases such as “So” for “show” are accepted
  when followed by a query. Spoken `me`, `up`, `the`, `my`, `a`, or `an`
  immediately after a command are ignored. Typed text is unchanged.
- `voice_auto_search` defaults to `true`. Set it to `false` to review and edit
  the transcript in the search bar before pressing Enter.
- `python -m app.voice_lab` records your 14 test phrases locally;
  `python -m app.voice_lab --eval` compares models after consent before any
  missing-model download; `python -m app.voice_lab --live` checks the real
  Alt-hold pipeline. Voice Lab recordings stay in app data and are never
  uploaded. Its report contains transcripts and metrics, not audio.
- `python -m app.model_cleanup` lists downloaded app and Hugging Face models
  without deleting anything. Use `--apply` to confirm each eligible model
  individually; models in use are protected, and shared-cache entries warn
  that other projects may depend on them. Add `--voice-lab-recordings` to
  inspect the local recordings folder; `--apply` and confirmation are
  required to remove it.
- Type a plain description, or use an explicit command:
  - `/find <query>` — show the best match with an explanation (also the
    default for plain text)
  - `/open <query>` — find and open the file directly
  - `/show <query>` — find the file and reveal it in File Explorer
- **`Escape`** or clicking elsewhere closes the search bar
- The tray icon's right-click menu offers **Search** and **Quit**

## Project structure

```
app/            UI: hotkey listener, search bar, result card, command parser
indexing/       file watcher, router, vision pipeline, text pipeline, embedder
agent/          retrieval harness, planner, corrective/reformulation logic, explainer
storage/        vector database + metadata store
observability/  activity log and structured per-query traces
eval/           accuracy evaluation (golden-set queries + expected results)
installer/      Windows installer script (Inno Setup)
```

## Known limitations

- Video files are detected but not yet processed for search (no
  transcription or frame analysis).
- `.rtf` files are not yet supported for text extraction.
- No content-based deduplication yet — identical files under different
  names or locations are indexed independently.
- Image indexing is meaningfully slower than text indexing, since it
  runs a real vision-language model per image; this only happens once
  per image, not on every search.
- Windows may disable the optional auto-start entry for unsigned
  executables — see the installation note above.

## License

This project is provided as-is for evaluation and personal use.
