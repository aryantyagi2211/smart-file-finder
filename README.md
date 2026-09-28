<img width="2400" height="1350" alt="9_Smart-File-Finder" src="https://github.com/user-attachments/assets/2fd657df-5d8c-4a56-82cb-2a14152fd489" />
<img width="2400" height="1350" alt="5_Local-Models-Local-Storage-No-External-Calls" src="https://github.com/user-attachments/assets/7fb76e59-33c1-4fe0-a15a-b17b9dbe9db0" />
<img width="2400" height="1520" alt="4_A-Retrieval-Pipeline-That-Checks-Its-Own-Confidence" src="https://github.com/user-attachments/assets/46e8de5c-0fd8-4b31-81d2-74a4cac046a7" />
<img width="2400" height="1350" alt="3_Describe-It-Naturally-Find-It-Instantly-Stay-Private" src="https://github.com/user-attachments/assets/0eb7dae0-8d91-42b4-9a34-4d2a5acc6df5" />
<img width="2400" height="1426" alt="2_Search-Only-Works-If-You-Remember-Exactly" src="https://github.com/user-attachments/assets/ebc15758-9a89-472a-a84f-ad2a92e32436" />
<img width="2400" height="1350" alt="1_Smart-File-Finder" src="https://github.com/user-attachments/assets/927ea47b-1802-4086-b125-ca124cf5e5a5" />
<img width="2400" height="1682" alt="6_Measured-Not-Assumed" src="https://github.com/user-attachments/a<img width="2400" height="1350" alt="8_Whats-Next" src="https://github.com/user-attachments/assets/a2e324d4-7791-493e-9999-ad9b2d6e8d3d" />
<img width="2400" height="1362" alt="7_Built-for-Daily-Use-Not-Just-a-Demo" src="https://github.com/user-attachments/assets/28139ad4-baf0-403e-997d-f3e5b6f816e9" />


# Smart File Finder

A desktop app that finds files by what's actually *inside* them — including
screenshots and images that were never named or tagged — using plain
natural language, entirely on your own device. No cloud calls, no
uploads: every model runs locally, so your files and your searches never
leave your machine.

Press a hotkey, type what you're looking for the way you'd describe it to
a person, and get back the actual file — with a short explanation of why
it matched.

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
- **Global hotkey launcher** — `Ctrl+Shift+Space` opens a small, always-on-top
  search bar from anywhere in Windows, similar to Spotlight or PowerToys Run.
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
installs the app, creates a Start Menu entry, and (optionally) sets it to
start automatically on login.

> **Note:** since this build isn't code-signed, Windows may flag the
> installer's Startup entry as "disabled" the first time — if the app
> doesn't auto-start after a restart, open **Task Manager → Startup apps**
> and enable "SmartFileFinder" manually. This is a one-time step.

### Option 2 — Run from source

Requires Python 3.10 and a CUDA-capable GPU for reasonable indexing
speed (the app also runs on CPU, just more slowly for image indexing).

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
  remembers its position)
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
- Windows may disable the app's auto-start entry for unsigned
  executables — see the installation note above.

## License

This project is provided as-is for evaluation and personal use.
