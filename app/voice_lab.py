"""Private, local voice recordings and model-comparison utilities."""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import re
import threading
import time
import wave

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

from app.command_parser import (
    DEFAULT_VOICE_COMMAND_ALIASES,
    parse_voice_command,
)
from app.voice_recorder import VoiceRecorder
from app.voice_transcriber import (
    DEFAULT_VOICE_HOTWORDS,
    DEFAULT_VOICE_INITIAL_PROMPT,
    VoiceTranscriber,
)
from app_paths import (
    get_config_path,
    get_models_dir,
    get_voice_lab_dir,
)


FIXED_PHRASES = [
    "show corrective rag",
    "open the agentic system design roadmap",
    "find my resume",
    "open gnani api",
    "show me the roadmap",
    "open adaptive rag",
    "show generative AI notes",
    "find the file about pricing strategy",
]

NATURAL_FILE_NAMES = [
    "meeting_notes.txt",
    "roasted_vegetable_soup_recipe.txt",
    "lake_alder_water_report.docx",
    "Google_Cloud_Storage_logo.png",
    "urban_gardening_guide.md",
    "astronomy_study_notes.md",
]

VOICE_LAB_CANDIDATES = [
    ("tiny.en", "Systran/faster-whisper-tiny.en", "faster-whisper"),
    ("base.en", "Systran/faster-whisper-base.en", "faster-whisper"),
    ("small.en", "Systran/faster-whisper-small.en", "faster-whisper"),
    (
        "distil-medium.en",
        "Systran/faster-distil-whisper-medium.en",
        "faster-whisper",
    ),
    ("medium.en", "Systran/faster-whisper-medium.en", "faster-whisper"),
    (
        "large-v3-turbo (CTranslate2 CPU)",
        "dropbox-dash/faster-whisper-large-v3-turbo",
        "faster-whisper",
    ),
    (
        "large-v3-turbo (Transformers GPU, experimental)",
        "openai/whisper-large-v3-turbo",
        "transformers-gpu",
    ),
]

GOLDEN_SET_PATH = Path(__file__).resolve().parents[1] / "eval" / "golden_set.json"
REPORT_PATH = GOLDEN_SET_PATH.parent / "voice_report.md"
_CT2_REQUIRED = ("config.json", "model.bin", "tokenizer.json")
_TRANSFORMERS_FILES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


@dataclass
class Recording:
    phrase: str
    wav_path: Path
    session: str


@dataclass
class CandidateResult:
    name: str
    repo_id: str
    license: str = "Unknown"
    gated: object = "unknown"
    model_size_bytes: int = 0
    repository_size_bytes: int = 0
    revision: str = ""
    load_seconds: float = 0.0
    average_seconds_per_clip: float = 0.0
    peak_ram_mib: float = 0.0
    peak_vram_mib: object = "n/a"
    transcripts: list = field(default_factory=list)
    wer: object = "n/a"
    command_accuracy: object = "n/a"
    status: str = "pending"
    error: str = ""


def voice_lab_recordings_dir(root=None):
    return Path(root or get_voice_lab_dir()) / "recordings"


def recording_phrases(
    golden_set_path=GOLDEN_SET_PATH,
    test_data_dir=None,
):
    root = Path(test_data_dir or GOLDEN_SET_PATH.parents[1] / "test_data")
    with open(golden_set_path, encoding="utf-8-sig") as file:
        cases = json.load(file)
    existing_names = {
        Path(name).name.casefold()
        for case in cases
        for name in (
            [case.get("expected_file")]
            if case.get("expected_file")
            else case.get("acceptable_files", [])
        )
        if name
    }

    phrases = list(FIXED_PHRASES)
    selected = []
    for name in NATURAL_FILE_NAMES:
        if name.casefold() in existing_names and (root / name).is_file():
            selected.append(name)
    if len(selected) < 6:
        for name in sorted(existing_names):
            candidate = root / name
            if candidate.is_file() and name not in {
                selected_name.casefold() for selected_name in selected
            }:
                selected.append(candidate.name)
            if len(selected) == 6:
                break

    commands = ("find", "show", "open", "find", "show", "open")
    for command, name in zip(commands, selected[:6]):
        words = re.sub(r"[_().-]+", " ", Path(name).stem)
        words = " ".join(words.split())
        phrases.append(f"{command} {words}")
    if len(phrases) != 14:
        raise ValueError(
            "Voice Lab needs six existing file names from test_data and "
            "eval/golden_set.json to make its 14 recording phrases."
        )
    return phrases


def write_recording(voice_lab_root, session, index, audio, phrase):
    directory = voice_lab_recordings_dir(voice_lab_root) / session
    directory.mkdir(parents=True, exist_ok=True)
    file_name = f"phrase-{index:02d}.wav"
    path = directory / file_name
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(VoiceRecorder.SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    manifest_path = directory / "manifest.json"
    manifest = {"session": session, "sample_rate": 16000, "recordings": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recordings"] = [
        item for item in manifest["recordings"]
        if item.get("file") != file_name
    ]
    manifest["recordings"].append({"file": file_name, "phrase": phrase})
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def record_phrases(
    recorder_factory=VoiceRecorder,
    input_fn=input,
    voice_lab_root=None,
    phrases=None,
):
    phrases = list(phrases or recording_phrases())
    root = Path(voice_lab_root or get_voice_lab_dir())
    session = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    recorder = recorder_factory(max_seconds=15, min_seconds=0.6)
    print(f"Voice Lab recordings stay on this PC: {root}")
    for index, phrase in enumerate(phrases, start=1):
        while True:
            print(f"\n{index}/{len(phrases)}: Say: {phrase}")
            input_fn("Press Enter to start recording...")
            try:
                recorder.start()
                input_fn("Speak now; press Enter to stop...")
                audio = recorder.stop()
            except KeyboardInterrupt:
                recorder.cancel()
                raise
            except Exception:
                recorder.cancel()
                raise
            if audio is None:
                print("Recording was too short; repeat this phrase.")
                continue
            path = write_recording(
                root, session, index, audio, phrase
            )
            audio.fill(0)
            print(f"recorded: {path.name}")
            break
    print(f"Saved local recordings under: {voice_lab_recordings_dir(root)}")
    return session


def load_recordings(voice_lab_root=None):
    base = voice_lab_recordings_dir(voice_lab_root)
    recordings = []
    if not base.is_dir():
        return recordings
    for manifest_path in sorted(base.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("recordings", []):
            file_name = Path(str(entry.get("file", ""))).name
            wav_path = manifest_path.parent / file_name
            if not wav_path.is_file():
                continue
            recordings.append(
                Recording(
                    phrase=str(entry.get("phrase", "")),
                    wav_path=wav_path,
                    session=str(manifest.get("session", "")),
                )
            )
    return recordings


def read_recording(recording):
    with wave.open(str(recording.wav_path), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getframerate() != VoiceRecorder.SAMPLE_RATE
        ):
            raise ValueError(
                f"Expected a 16 kHz mono 16-bit WAV: {recording.wav_path}"
            )
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _candidate_files(engine, siblings):
    sizes = {
        str(sibling.rfilename): int(sibling.size)
        for sibling in siblings
        if getattr(sibling, "size", None) is not None
    }
    if engine == "faster-whisper":
        required = list(_CT2_REQUIRED)
        vocabulary = next(
            (name for name in ("vocabulary.json", "vocabulary.txt") if name in sizes),
            None,
        )
        if vocabulary is None:
            raise ValueError("Repository has no faster-whisper vocabulary file.")
        required.append(vocabulary)
        if "preprocessor_config.json" in sizes:
            required.append("preprocessor_config.json")
    else:
        required = sorted(_TRANSFORMERS_FILES.intersection(sizes))
        if not {"config.json", "model.safetensors"}.issubset(required):
            raise ValueError(
                "Transformers repository is missing its config or safetensors weights."
            )
    missing = [name for name in required if name not in sizes]
    if missing:
        raise ValueError(
            "Repository is missing required loader files: "
            + ", ".join(missing)
        )
    return {name: sizes[name] for name in required}


def fetch_candidate_metadata(candidate, api=None):
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    _, repo_id, engine = candidate
    info = api.model_info(repo_id, files_metadata=True, timeout=30)
    siblings = getattr(info, "siblings", None) or []
    files = _candidate_files(engine, siblings)
    license_name = None
    card_data = getattr(info, "card_data", None)
    if card_data is not None:
        if isinstance(card_data, dict):
            license_name = card_data.get("license")
        else:
            license_name = getattr(card_data, "license", None)
    tags = getattr(info, "tags", []) or []
    if not license_name:
        license_name = next(
            (
                tag.split(":", 1)[1]
                for tag in tags
                if str(tag).startswith("license:")
            ),
            None,
        )
    size_bytes = sum(files.values())
    repository_bytes = sum(
        int(sibling.size)
        for sibling in siblings
        if getattr(sibling, "size", None) is not None
    )
    return {
        "name": candidate[0],
        "repo_id": repo_id,
        "engine": engine,
        "revision": str(getattr(info, "sha", "") or ""),
        "license": str(license_name or "Unknown"),
        "gated": getattr(info, "gated", None),
        "private": bool(getattr(info, "private", False)),
        "files": files,
        "download_size_bytes": size_bytes,
        "download_size_gb": size_bytes / 1_000_000_000,
        "repository_size_bytes": repository_bytes,
        "status": "ready",
        "purpose": "voice",
        "tiers": [],
    }


def _snapshot_path(model):
    return (
        get_models_dir()
        / ("models--" + model["repo_id"].replace("/", "--"))
        / "snapshots"
        / model["revision"]
    )


def find_voice_lab_model(model):
    try:
        from app_paths import find_model

        found = find_model(model["repo_id"])
    except (OSError, ValueError, KeyError):
        found = None
    if (
        found is not None
        and Path(found).name == model["revision"]
        and _files_are_complete(Path(found), model["files"])
    ):
        return Path(found)
    app_models = get_models_dir()
    hf_home = Path(
        os.environ.get(
            "HF_HOME",
            Path.home() / ".cache" / "huggingface",
        )
    )
    caches = [
        app_models,
        Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")),
    ]
    for cache in caches:
        path = (
            cache
            / ("models--" + model["repo_id"].replace("/", "--"))
            / "snapshots"
            / model["revision"]
        )
        if _files_are_complete(path, model["files"]):
            return path
    return None


def _files_are_complete(directory, files):
    try:
        return all(
            (directory / name).is_file()
            and (directory / name).stat().st_size == size
            for name, size in files.items()
        )
    except OSError:
        return False


def load_voice_settings(settings_path=None):
    path = settings_path or get_config_path("settings.json")
    settings = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return {
        "initial_prompt": settings.get(
            "voice_initial_prompt", DEFAULT_VOICE_INITIAL_PROMPT
        ),
        "hotwords": settings.get("voice_hotwords", DEFAULT_VOICE_HOTWORDS),
        "command_aliases": settings.get(
            "voice_command_aliases", DEFAULT_VOICE_COMMAND_ALIASES
        ),
    }


def consent_to_download(model, input_fn=input):
    size_mb = model["download_size_bytes"] / 1_000_000
    while True:
        answer = input_fn(
            f"Download {model['name']} ({size_mb:.0f} MB) to app data? "
            "[y]es / [n]o / [c]ancel evaluation: "
        ).strip().casefold()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no", ""}:
            return False
        if answer in {"c", "cancel"}:
            raise InterruptedError("Voice Lab evaluation cancelled.")
        print("Enter y, n, or c.")


def _model_display_size(size_bytes):
    return f"{size_bytes / 1_000_000:.1f} MB"


def word_error_counts(reference, hypothesis):
    ref = re.findall(r"\w+", reference.casefold())
    hyp = re.findall(r"\w+", hypothesis.casefold())
    rows = list(range(len(hyp) + 1))
    for ref_index, ref_word in enumerate(ref, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    current[hyp_index - 1] + 1,
                    rows[hyp_index] + 1,
                    rows[hyp_index - 1] + (ref_word != hyp_word),
                )
            )
        rows = current
    return rows[-1], len(ref)


class _PeakRam:
    def __init__(self):
        import psutil

        self._process = psutil.Process()
        self._stop = threading.Event()
        self.peak_bytes = self._process.memory_info().rss
        self._thread = threading.Thread(
            target=self._sample,
            daemon=True,
            name="voice-lab-memory",
        )

    def _sample(self):
        while not self._stop.wait(0.05):
            self.peak_bytes = max(
                self.peak_bytes, self._process.memory_info().rss
            )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self.peak_bytes = max(
            self.peak_bytes, self._process.memory_info().rss
        )
        self._stop.set()
        self._thread.join(timeout=1)


def _calculate_result_metrics(result, recordings, aliases):
    errors = words = correct_commands = 0
    transcripts = list(result.transcripts)
    for recording, transcript in zip(recordings, transcripts):
        errors_for_clip, words_for_clip = word_error_counts(
            recording.phrase, transcript
        )
        errors += errors_for_clip
        words += words_for_clip
        expected = parse_voice_command(
            recording.phrase, aliases=aliases
        ).command
        actual = parse_voice_command(transcript, aliases=aliases).command
        correct_commands += expected == actual
    result.transcripts = list(zip(
        (recording.phrase for recording in recordings),
        transcripts,
    ))
    result.wer = errors / words if words else 0.0
    result.command_accuracy = (
        correct_commands / len(recordings) if recordings else 0.0
    )


def evaluate_faster_candidate(
    model,
    model_path,
    recordings,
    settings,
    transcriber_factory=None,
):
    factory = transcriber_factory or VoiceTranscriber
    transcriber = factory(
        model_path=model_path,
        model_name=model["name"],
        prefer_cuda=False,
        initial_prompt=settings["initial_prompt"],
        hotwords=settings["hotwords"],
    )
    result = CandidateResult(
        name=model["name"],
        repo_id=model["repo_id"],
        license=model["license"],
        gated=model["gated"],
        model_size_bytes=model["download_size_bytes"],
        repository_size_bytes=model["repository_size_bytes"],
        revision=model["revision"],
    )
    try:
        with _PeakRam() as memory:
            load_started = time.perf_counter()
            transcriber.get_model()
            result.load_seconds = time.perf_counter() - load_started
            timings = []
            transcripts = []
            for recording in recordings:
                audio = read_recording(recording)
                started = time.perf_counter()
                try:
                    transcript = transcriber.transcribe(audio)
                finally:
                    audio.fill(0)
                timings.append(time.perf_counter() - started)
                transcripts.append(transcript)
            result.peak_ram_mib = memory.peak_bytes / (1024 ** 2)
    finally:
        transcriber.unload()
    result.average_seconds_per_clip = (
        sum(timings) / len(timings) if timings else 0.0
    )
    result.transcripts = transcripts
    _calculate_result_metrics(result, recordings, settings["command_aliases"])
    result.status = "complete"
    return result


def evaluate_transformers_candidate(model, model_path, recordings, settings):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    result = CandidateResult(
        name=model["name"],
        repo_id=model["repo_id"],
        license=model["license"],
        gated=model["gated"],
        model_size_bytes=model["download_size_bytes"],
        repository_size_bytes=model["repository_size_bytes"],
        revision=model["revision"],
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch reports no available CUDA GPU for the experimental candidate."
        )

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    prompt = " ".join(
        item for item in (settings["initial_prompt"], settings["hotwords"])
        if item
    )
    with _PeakRam() as memory:
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device)
        whisper_model.eval()
        result.load_seconds = time.perf_counter() - load_started
        prompt_ids = (
            torch.as_tensor(
                processor.get_prompt_ids(prompt),
                dtype=torch.long,
                device=device,
            )
            if prompt
            else None
        )
        timings = []
        transcripts = []
        for recording in recordings:
            audio = read_recording(recording)
            audio = VoiceTranscriber.normalize_audio(audio)
            inputs = processor(
                audio=audio,
                sampling_rate=VoiceRecorder.SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(
                device=device, dtype=torch.float16
            )
            generate_options = {
                "language": "en",
                "task": "transcribe",
                "num_beams": 5,
                "do_sample": False,
                "temperature": 0.0,
            }
            if hasattr(
                whisper_model.generation_config,
                "condition_on_prev_tokens",
            ):
                generate_options["condition_on_prev_tokens"] = False
            if prompt_ids is not None:
                generate_options["prompt_ids"] = prompt_ids
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = whisper_model.generate(
                        input_features, **generate_options
                    )
                transcript = processor.batch_decode(
                    output, skip_special_tokens=True
                )[0].strip()
            finally:
                audio.fill(0)
                del input_features, inputs
            timings.append(time.perf_counter() - started)
            transcripts.append(transcript)
            result.peak_vram_mib = max(
                float(result.peak_vram_mib)
                if isinstance(result.peak_vram_mib, (int, float))
                else 0.0,
                torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            )
        del whisper_model, processor
        torch.cuda.empty_cache()
        result.peak_ram_mib = memory.peak_bytes / (1024 ** 2)
    result.average_seconds_per_clip = (
        sum(timings) / len(timings) if timings else 0.0
    )
    result.transcripts = transcripts
    _calculate_result_metrics(result, recordings, settings["command_aliases"])
    result.status = "complete"
    return result


def _safe_markdown(value):
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_report(results, report_date=None):
    date_text = report_date or datetime.now().astimezone().strftime("%Y-%m-%d")
    lines = [
        f"# Voice Lab report — {date_text}",
        "",
        "Recordings and evaluation were run locally. No audio is included in this report.",
        "",
        "| Candidate | Repository | License | Gated | Model files | Repository snapshot | WER | Command accuracy | Avg s/clip | Load s | Peak RAM MiB | Peak VRAM MiB | Status |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                _safe_markdown(value)
                for value in (
                    result.name,
                    result.repo_id,
                    result.license,
                    result.gated,
                    _model_display_size(result.model_size_bytes),
                    _model_display_size(result.repository_size_bytes),
                    f"{result.wer:.2%}" if isinstance(result.wer, float) else result.wer,
                    (
                        f"{result.command_accuracy:.1%}"
                        if isinstance(result.command_accuracy, float)
                        else result.command_accuracy
                    ),
                    f"{result.average_seconds_per_clip:.2f}",
                    f"{result.load_seconds:.2f}",
                    f"{result.peak_ram_mib:.0f}",
                    (
                        f"{result.peak_vram_mib:.0f}"
                        if isinstance(result.peak_vram_mib, (int, float))
                        else result.peak_vram_mib
                    ),
                    result.status,
                )
            )
            + " |"
        )
        if result.revision:
            lines.extend(["", f"Revision: `{result.revision}`."])
        if result.error:
            lines.extend(["", f"Result: {_safe_markdown(result.error)}"])
        lines.extend(["", f"## {result.name}", ""])
        if result.transcripts:
            lines.extend(
                [
                    "| Expected phrase | Exact transcript |",
                    "|---|---|",
                ]
            )
            for phrase, transcript in result.transcripts:
                lines.append(
                    f"| {_safe_markdown(phrase)} | {_safe_markdown(transcript)} |"
                )
        else:
            lines.append("No transcripts were produced.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_report(results, report_path=REPORT_PATH, report_date=None):
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_report(results, report_date=report_date),
        encoding="utf-8",
    )
    return path


def print_results(results):
    for result in results:
        print(f"\n{result.name} ({result.repo_id}) — {result.status}")
        print(
            f"  License: {result.license}; gated: {result.gated}; "
            f"loader files: {_model_display_size(result.model_size_bytes)}; "
            f"repo snapshot: "
            f"{_model_display_size(result.repository_size_bytes)}"
        )
        if result.error:
            print(f"  {result.error}")
        print(
            "  WER: "
            f"{result.wer:.2%}" if isinstance(result.wer, float)
            else f"  WER: {result.wer}"
        )
        print(
            "  Command-word accuracy: "
            f"{result.command_accuracy:.1%}"
            if isinstance(result.command_accuracy, float)
            else f"  Command-word accuracy: {result.command_accuracy}"
        )
        print(
            f"  Average seconds/clip: {result.average_seconds_per_clip:.2f}; "
            f"load: {result.load_seconds:.2f}s; "
            f"peak RAM: {result.peak_ram_mib:.0f} MiB; "
            f"peak VRAM: {result.peak_vram_mib}"
        )
        for phrase, transcript in result.transcripts:
            print(f"  {phrase!r} -> {transcript!r}")


def _metadata_result(candidate, error):
    return CandidateResult(
        name=candidate[0],
        repo_id=candidate[1],
        status="unavailable",
        error=str(error),
    )


def _model_result(model, status, error=""):
    return CandidateResult(
        name=model["name"],
        repo_id=model["repo_id"],
        license=model["license"],
        gated=model["gated"],
        model_size_bytes=model["download_size_bytes"],
        repository_size_bytes=model["repository_size_bytes"],
        revision=model["revision"],
        status=status,
        error=str(error),
    )


def _download_voice_lab_model(model):
    from indexing.model_downloader import download_model

    return download_model(model, download_source="voice_lab")


def evaluate_recordings(
    recordings=None,
    voice_lab_root=None,
    report_path=REPORT_PATH,
    input_fn=input,
    api=None,
    downloader=None,
    faster_transcriber_factory=None,
):
    recordings = recordings if recordings is not None else load_recordings(
        voice_lab_root
    )
    if not recordings:
        raise ValueError(
            "No Voice Lab recordings found. Run `python -m app.voice_lab` first."
        )
    settings = load_voice_settings()
    results = []
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    if downloader is None:
        downloader = download_voice_lab_model

    for candidate in VOICE_LAB_CANDIDATES:
        try:
            model = fetch_candidate_metadata(candidate, api=api)
        except Exception as error:
            results.append(_metadata_result(candidate, error))
            continue
        print(
            f"{model['name']}: {model['repo_id']} | "
            f"license={model['license']} | gated={model['gated']} | "
            f"loader files={_model_display_size(model['download_size_bytes'])} | "
            f"repo snapshot={_model_display_size(model['repository_size_bytes'])} | "
            f"revision={model['revision']}"
        )
        if model["private"] or model["gated"]:
            results.append(_model_result(
                model,
                "unavailable",
                "Repository is private or gated; not downloading it.",
            ))
            continue
        if not model["revision"] or model["license"] == "Unknown":
            results.append(_model_result(
                model,
                "unavailable",
                "Could not verify the repository revision and license; skipped.",
            ))
            continue
        model_path = find_voice_lab_model(model)
        if model_path is None:
            try:
                if not consent_to_download(model, input_fn=input_fn):
                    results.append(_model_result(
                        model,
                        "not downloaded",
                        "Consent declined.",
                    ))
                    continue
            except InterruptedError as error:
                results.append(
                    _model_result(model, "cancelled", str(error))
                )
                break
            try:
                model_path = Path(downloader(model))
            except Exception as error:
                results.append(
                    _model_result(model, "download failed", error)
                )
                continue
        try:
            if model["engine"] == "faster-whisper":
                result = evaluate_faster_candidate(
                    model,
                    model_path,
                    recordings,
                    settings,
                    transcriber_factory=faster_transcriber_factory,
                )
            else:
                result = evaluate_transformers_candidate(
                    model, model_path, recordings, settings
                )
        except Exception as error:
            result = _model_result(model, "failed", error)
        results.append(result)
        gc.collect()

    print_results(results)
    path = save_report(results, report_path=report_path)
    print(f"\nVoice Lab report saved to: {path}")
    return results


def _current_voice_model():
    from app.hardware_profile import choose_tier, detect_hardware
    from app_paths import find_model, get_voice_model_for_tier

    settings = load_voice_settings()
    try:
        raw_settings = json.loads(
            Path(get_config_path("settings.json")).read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError):
        raw_settings = {}
    tier, _reason = choose_tier(
        detect_hardware(), raw_settings.get("performance_tier", "auto")
    )
    model = get_voice_model_for_tier(
        tier, model_name=raw_settings.get("voice_model_override") or None
    )
    if model is None:
        raise RuntimeError(f"No configured voice model exists for tier {tier}.")
    model_path = find_model(model["repo_id"])
    if model_path is None:
        raise RuntimeError(
            f"{model['name']} is not installed. Start the app and consent to "
            "its required model download before using Voice Lab live mode."
        )
    return tier, model, model_path, settings, raw_settings


def run_live():
    from app.hotkey_listener import register_tap_hold_hotkey
    from app.voice_hotkey import VoiceGesture

    tier, model, model_path, settings, raw_settings = _current_voice_model()
    voice_key = str(raw_settings.get("voice_key", "left alt")).casefold()
    hold_ms = int(raw_settings.get("voice_hold_ms", 400))
    tail_ms = max(0, int(raw_settings.get("voice_tail_ms", 300)))
    pre_roll = bool(raw_settings.get("voice_preroll", True))
    transcriber = VoiceTranscriber(
        model_path=model_path,
        model_name=model["name"],
        prefer_cuda=(
            tier in {"mid", "big"}
            and model["name"] == "Whisper large-v3-turbo"
        ),
        initial_prompt=settings["initial_prompt"],
        hotwords=settings["hotwords"],
    )
    recorder = VoiceRecorder(
        max_seconds=int(raw_settings.get("voice_max_seconds", 15))
    )
    state_lock = threading.Lock()
    state = {"preroll": False, "recording": False, "timer": None}

    def transcribe_current_clip():
        with state_lock:
            state["timer"] = None
            state["recording"] = False
            audio = recorder.stop()
        if audio is None:
            print("No speech captured.")
            return
        try:
            transcript = transcriber.transcribe(audio)
            parsed = parse_voice_command(
                transcript, aliases=settings["command_aliases"]
            )
            print(f"Raw transcript: {transcript!r}")
            print(
                f"Parsed command: {parsed.command.value}; "
                f"query: {parsed.query!r}"
            )
        except Exception as error:
            print(f"Transcription failed: {error}")
        finally:
            audio.fill(0)

    def on_gesture(gesture):
        if gesture == VoiceGesture.PRE_ROLL_START:
            with state_lock:
                if state["recording"]:
                    if state["timer"] is not None:
                        state["timer"].cancel()
                        state["timer"] = None
                    return
                if state["preroll"]:
                    return
                recorder.start()
                state["preroll"] = True
        elif gesture == VoiceGesture.PRE_ROLL_CANCEL:
            with state_lock:
                if state["preroll"]:
                    state["preroll"] = False
                    recorder.cancel()
        elif gesture == VoiceGesture.HOLD_START:
            with state_lock:
                if not state["preroll"]:
                    recorder.start()
                state["preroll"] = False
                state["recording"] = True
            print("Listening...")
        elif gesture == VoiceGesture.HOLD_END:
            with state_lock:
                if not state["recording"]:
                    if state["preroll"]:
                        state["preroll"] = False
                        recorder.cancel()
                    return
                timer = threading.Timer(
                    tail_ms / 1000.0, transcribe_current_clip
                )
                timer.daemon = True
                state["timer"] = timer
                timer.start()
        elif gesture == VoiceGesture.CANCEL:
            with state_lock:
                if state["timer"] is not None:
                    state["timer"].cancel()
                    state["timer"] = None
                state["recording"] = False
                state["preroll"] = False
                recorder.cancel()

    unregister = register_tap_hold_hotkey(
        on_gesture,
        hold_threshold_ms=hold_ms,
        max_hold_ms=int(raw_settings.get("voice_max_seconds", 15)) * 1000,
        voice_key=voice_key,
        pre_roll_enabled=pre_roll,
    )
    print(
        f"Voice Lab live mode — hold {voice_key} for {hold_ms} ms. "
        "Press Ctrl+C to exit."
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        unregister()
        with state_lock:
            timer = state["timer"]
            state["timer"] = None
            recording = state["recording"] or state["preroll"]
            state["recording"] = False
            state["preroll"] = False
        if timer is not None:
            timer.cancel()
        if recording:
            recorder.cancel()
        transcriber.unload()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Record local voice samples or compare local Whisper models."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval", action="store_true", help="compare model candidates")
    mode.add_argument("--live", action="store_true", help="test the real Alt-hold pipeline")
    args = parser.parse_args(argv)
    try:
        if args.eval:
            evaluate_recordings()
        elif args.live:
            run_live()
        else:
            record_phrases()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Voice Lab: {error}\n")


if __name__ == "__main__":
    main()
