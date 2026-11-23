"""
app/main.py

Wires everything together: global hotkey -> input bar -> command parser
-> query handler (real harness) -> result card. Also runs the initial
file indexing scan on a background thread (not blocking the GUI) and
starts the live file watcher once that scan completes, per the "async
indexing" upgrade (real Desktop/Documents/Downloads folders can be large,
so blocking app startup on a full scan is a bad experience).
"""

import sys
import threading
import os
import json

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

if __name__ == "__main__":
    from app.instance_lock import acquire_single_instance_lock

    if not acquire_single_instance_lock():
        print("Smart File Finder is already running.")
        sys.exit(0)

from PyQt6.QtWidgets import QApplication, QMessageBox, QLabel
from PyQt6.QtCore import QObject, pyqtSignal, Qt

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.hotkey_listener import (
    DEFAULT_HOTKEY,
    register_tap_hold_hotkey,
    register_hotkey,
    unregister_hotkey,
)
from app.voice_hotkey import VoiceGesture
from app.voice_recorder import VoiceRecorder, VoiceRecorderError
from app.voice_transcriber import VoiceTranscriber
from app.voice_overlay import VoiceOverlay
from app.input_bar import InputBar
from app.command_parser import parse_command, parse_voice_command, Command
from app.query_handler import handle_query
from app.result_card import (
    OptionsPopupGroup,
    ResultCard,
    open_file,
    reveal_in_explorer,
)
from indexing.backfill_index import (
    EXCLUDED_FOLDER_NAMES,
    SUPPORTED_EXTENSIONS,
    backfill,
    index_one_file,
    load_indexed_folders,
    remove_file_index,
)
from pathlib import Path
from app_paths import (
    ensure_config_exists,
    get_big_vision_model_setting,
    get_hardware_profile_path,
    get_min_free_disk_gb_for_tier,
    get_resource_path,
    get_config_path,
    get_voice_model_for_tier,
)
from app.hardware_profile import (
    choose_tier,
    detect_hardware,
    get_performance_tier_override,
    get_tier_display_name,
    set_current_tier,
)
from app.setup_window import SetupWindow
from indexing.model_downloader import (
    missing_models_for_tier,
    missing_voice_model_for_tier,
    set_huggingface_offline,
)


from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon

from app.indexing_status import IndexingStatusWidget
from indexing.embedder import preload_model
from observability.logger import log, log_voice, log_voice_exception

import time
print(f"[TIMING] Script started at {time.time()}")

class HotkeyBridge(QObject):
    triggered = pyqtSignal()


class IndexingDoneBridge(QObject):
    finished = pyqtSignal()

class ProgressBridge(QObject):
    progress_updated = pyqtSignal(str, str, int, int)
    error_occurred = pyqtSignal(str)
    model_loading = pyqtSignal(str)
    notice = pyqtSignal(str)


class HardwareTierBridge(QObject):
    ready = pyqtSignal(str, str, object, str)


LIVE_TRANSCRIPTION_INTERVAL_SECONDS = 0.65
LIVE_TRANSCRIPTION_MIN_NEW_SECONDS = 0.45


class VoiceBridge(QObject):
    state = pyqtSignal(str)
    text = pyqtSignal(str)
    partial_text = pyqtSignal(str)
    error = pyqtSignal(str)


class LiveIndexHandler(FileSystemEventHandler):
    def __init__(self, status_callback=None, model_status_callback=None):
        super().__init__()
        self.status_callback = status_callback
        self.model_status_callback = model_status_callback

    def on_created(self, event):
        if not event.is_directory:
            self._index(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._index(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_supported_path(event.src_path):
            if self.status_callback is not None:
                self.status_callback("Removing deleted file from index...")
            remove_file_index(event.src_path, reason="file deleted")
            if self.status_callback is not None:
                self.status_callback("Watching for changes")

    def on_moved(self, event):
        if event.is_directory:
            return
        if self._is_supported_path(event.src_path):
            if self.status_callback is not None:
                self.status_callback("Updating moved file in index...")
            remove_file_index(event.src_path, reason="file moved")
        self._index(event.dest_path, force=True)
        if (
            not self._is_supported_path(event.dest_path)
            and self.status_callback is not None
        ):
            self.status_callback("Watching for changes")

    @staticmethod
    def _is_supported_path(file_path):
        path_obj = Path(file_path)
        return (
            path_obj.suffix.lower() in SUPPORTED_EXTENSIONS
            and not any(excluded in path_obj.parts for excluded in EXCLUDED_FOLDER_NAMES)
        )

    def _index(self, file_path, force=False):
        if not self._is_supported_path(file_path):
            return
        if self.status_callback is not None:
            self.status_callback(
                f"Indexing changed file: {Path(file_path).stem}"
            )
        try:
            index_options = {"force": force}
            if self.model_status_callback is not None:
                index_options["model_status_callback"] = (
                    self.model_status_callback
                )
            status = index_one_file(file_path, **index_options)
            print(status)
        except Exception as e:
            print(f"[ERROR - live watch] {file_path} -> {e}")
            if self.status_callback is not None:
                self.status_callback(
                    f"Indexing failed: {Path(file_path).name}"
                )
            return
        if self.status_callback is not None:
            self.status_callback("Watching for changes")


def _load_saved_hardware_report():
    path = get_hardware_profile_path()
    try:
        with open(path, encoding="utf-8-sig") as file:
            report = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        log(f"WARNING: Could not read saved hardware profile at {path}: {error}")
        return None
    if not isinstance(report, dict) or report.get("performance_tier") not in {
        "small", "mid", "big"
    }:
        log(f"WARNING: Invalid saved hardware profile at {path}")
        return None
    return report

class SmartFileFinderApp:
    def __init__(self):
        ensure_config_exists()
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.tray_icon = QSystemTrayIcon(QIcon(str(get_resource_path("app_icon.ico"))), self.app)
        self.tray_icon.setToolTip("Smart File Finder")

        tray_menu = QMenu()
        show_action = tray_menu.addAction("Search (Ctrl+Shift+Space)")
        show_action.triggered.connect(self._show_input_bar)
        tray_menu.addSeparator()
        self.performance_tier_action = tray_menu.addAction(
            "Performance tier: Detecting..."
        )
        self.performance_tier_action.setEnabled(False)
        tray_menu.addSeparator()
        self.voice_action = tray_menu.addAction("Voice search: On")
        self.voice_action.triggered.connect(self._toggle_voice)
        tray_menu.addSeparator()
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self.app.quit)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self.input_bar = InputBar(
            on_submit=self._on_query_submitted,
            on_escape=self._cancel_voice,
        )
        self.result_card = None
        self.options_group = None
        self.voice_overlay = VoiceOverlay(input_bar=self.input_bar)
        self.voice_overlay.dismissed.connect(
            self._on_voice_overlay_dismissed
        )
        self._voice_overlay_active = threading.Event()
        self._live_transcription_stop = threading.Event()
        self._live_transcription_thread = None
        self._pending_voice_transcript = None
        self._pending_voice_auto_search = True
        self.voice_bridge = VoiceBridge()
        self.voice_bridge.state.connect(self._on_voice_state)
        self.voice_bridge.text.connect(self._on_voice_text)
        self.voice_bridge.partial_text.connect(self._on_voice_partial_text)
        self.voice_bridge.error.connect(self._on_voice_error)
        self.voice_recorder = VoiceRecorder(
            max_seconds=self._load_voice_setting("voice_max_seconds", 15)
        )
        self.voice_transcriber = None
        self._voice_tier = None
        self.voice_available = self._check_voice_imports()
        self.voice_enabled = (
            self.voice_available
            and self._load_voice_setting("voice_enabled", True)
        )
        if self.voice_enabled:
            voice_reason = "enabled by settings and required imports are present"
        elif not self.voice_available:
            voice_reason = getattr(
                self, "_voice_unavailable_reason", "required imports are missing"
            )
        else:
            voice_reason = "disabled by settings"
        log_voice(f"enabled={self.voice_enabled} reason={voice_reason}")
        log_voice(
            "configured voice key="
            f"{self._load_voice_setting('voice_key', 'left alt')!r}"
        )
        self._hotkey_unregister = None
        self._search_hotkeys_registered = False
        self._recording = False
        self._voice_setup_window = None
        self._tier_reason = ""
        self._tier_mode = "auto"
        self._update_voice_tray_action()

        self.hotkey_bridge = HotkeyBridge()
        self.hotkey_bridge.triggered.connect(self._actually_show_input_bar)

        self.indexing_done_bridge = IndexingDoneBridge()
        self.indexing_done_bridge.finished.connect(self._on_indexing_finished)

        self._saved_hardware_report = _load_saved_hardware_report()
        self._show_hardware_scan = (
            self._saved_hardware_report is None
            or not self._saved_hardware_report.get("status_intro_shown", False)
        )
        self.status_window = IndexingStatusWidget()
        if self._show_hardware_scan:
            self.status_window.show_hardware_scan()
        else:
            saved_tier = self._saved_hardware_report["performance_tier"]
            saved_override = str(
                self._saved_hardware_report.get("override", "auto")
            ).strip().casefold()
            saved_mode = (
                "auto"
                if saved_override not in {"small", "mid", "big"}
                else "override"
            )
            self.status_window.set_performance_tier(saved_tier, saved_mode)
            self.status_window.show_status("Checking required models...")
        self.status_window.show()

        self.progress_bridge = ProgressBridge()
        self.progress_bridge.progress_updated.connect(self.status_window.show_progress)
        self.progress_bridge.error_occurred.connect(self.status_window.show_error)
        self.progress_bridge.model_loading.connect(
            self.status_window.show_status
        )
        self.progress_bridge.notice.connect(self.status_window.show_notice)

        self.setup_window = None
        self._normal_startup_started = False
        self._indexing_complete = False

        self.hardware_tier_bridge = HardwareTierBridge()
        self.hardware_tier_bridge.ready.connect(self._on_hardware_tier_ready)
        self._hardware_detection_thread = threading.Thread(
            target=self._detect_hardware_in_background,
            daemon=True,
            name="hardware-profile",
        )
        self._hardware_detection_thread.start()

    def _preload_embedding_model(self):
        self.progress_bridge.model_loading.emit(
            "Loading embedding model: all-mpnet-base-v2..."
        )
        try:
            preload_model()
        except Exception as error:
            log(f"Embedding model preload failed; it will retry on use: {error}")
            self.progress_bridge.model_loading.emit(
                "Embedding model failed to load; it will retry during indexing"
            )

    def _load_voice_setting(self, key, default):
        try:
            with open(get_config_path("settings.json"), encoding="utf-8-sig") as file:
                return json.load(file).get(key, default)
        except Exception:
            log_voice_exception(f"could not read voice setting {key!r}")
            return default

    def _save_voice_setting(self, key, value):
        path = get_config_path("settings.json")
        try:
            with open(path, encoding="utf-8-sig") as file:
                settings = json.load(file)
        except Exception:
            settings = {}
        settings[key] = value
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _check_voice_imports(self):
        missing = []
        for module_name in ("faster_whisper", "ctranslate2", "sounddevice", "numpy"):
            try:
                __import__(module_name)
            except Exception:
                missing.append(module_name)
        if missing:
            self._voice_unavailable_reason = (
                f"missing imports: {', '.join(missing)}"
            )
            log_voice(
                "availability=unavailable "
                f"reason={self._voice_unavailable_reason}"
            )
            return False
        self._voice_unavailable_reason = None
        return True

    def _update_voice_tray_action(self):
        if not self.voice_available:
            self.voice_action.setText(
                "Voice search is not available on this device yet"
            )
            self.voice_action.setEnabled(False)
        else:
            self.voice_action.setText(
                f"Voice search: {'On' if self.voice_enabled else 'Off'}"
            )

    def _toggle_voice(self):
        if not self.voice_available:
            return
        hotkeys_registered = self._search_hotkeys_registered
        if self._hotkey_unregister is not None:
            self._hotkey_unregister()
            self._hotkey_unregister = None
        self.voice_enabled = not self.voice_enabled
        self._save_voice_setting("voice_enabled", self.voice_enabled)
        self._update_voice_tray_action()
        if self.voice_enabled and self._voice_tier is not None:
            model = missing_voice_model_for_tier(self._voice_tier)
            if model is not None:
                self._open_required_model_setup(
                    self._on_voice_setup_download_ready,
                )
        if hotkeys_registered:
            self._register_search_hotkeys()

    def _open_required_model_setup(self, on_ready):
        try:
            missing = missing_models_for_tier(self._voice_tier)
        except Exception as error:
            log_voice_exception(f"required model lookup failed: {error}")
            self._on_voice_error("Could not check the required local models.")
            return
        if not missing:
            on_ready()
            return
        free_disk = None
        try:
            import psutil
            free_disk = psutil.disk_usage(
                str(get_config_path("settings.json").parent)
            ).free / 1e9
        except Exception:
            log_voice_exception("could not read free disk for model setup")
        self._voice_setup_window = SetupWindow(
            self._voice_tier,
            missing,
            free_disk,
            selection_reason=self._tier_reason,
            selection_mode=self._tier_mode,
        )
        self._voice_setup_window.status_changed.connect(
            self.status_window.show_status
        )
        self._voice_setup_window.ready.connect(on_ready)
        self._voice_setup_window.cancelled.connect(
            self._on_voice_setup_cancelled
        )
        self._voice_setup_window.show()

    def _on_voice_setup_download_ready(self):
        self._voice_setup_window = None

    def _detect_hardware_in_background(self):
        profile = {}
        override = "auto"
        tier = "small"
        reason = "small: hardware detection did not complete"
        try:
            profile = detect_hardware()
            override = get_performance_tier_override()
            if get_big_vision_model_setting() == "gemma-4-E4B-it":
                log(
                    "WARNING: Gemma E4B is enabled for the big tier; "
                    "this vision model is untested."
                )
            tier, reason = choose_tier(profile, override)
        except Exception as error:
            log(f"Hardware profile or tier selection failed: {error}")

        report = {
            "hardware": profile,
            "performance_tier": tier,
            "override": override,
            "reason": reason,
        }
        if profile:
            report["status_intro_shown"] = True
        try:
            profile_path = get_hardware_profile_path()
            temporary_path = profile_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(profile_path)
        except Exception as error:
            log(f"Could not save hardware profile: {error}")

        log(f"Hardware profile: {json.dumps(profile, sort_keys=True)}")
        log(f"Performance tier selected: {tier} ({reason})")
        normalized_override = str(override or "auto").strip().casefold()
        mode = (
            "auto"
            if normalized_override in ("", "auto")
            or normalized_override not in ("small", "mid", "big")
            else "override"
        )
        self.hardware_tier_bridge.ready.emit(
            tier, mode, profile.get("free_disk_gb"), reason
        )

    def _on_hardware_tier_ready(
        self, tier, mode, free_disk_gb, selection_reason=""
    ):
        set_current_tier(tier)
        self._voice_tier = tier
        self._tier_reason = selection_reason
        self._tier_mode = mode
        self.status_window.set_performance_tier(tier, mode)
        self.status_window.show_status("Checking required models...")
        self.status_window.show()
        try:
            model_entry = get_voice_model_for_tier(tier)
            if model_entry is None:
                log_voice(f"voice model path not found tier={tier}")
            else:
                from app_paths import find_model
                model_path = find_model(model_entry["repo_id"])
                log_voice(
                    f"voice model path found path={model_path}"
                    if model_path is not None
                    else "voice model path not found "
                    f"repo_id={model_entry['repo_id']}"
                )
        except Exception:
            log_voice_exception(f"voice model lookup failed for tier={tier}")
        display_tier = get_tier_display_name(tier)
        self.performance_tier_action.setText(
            f"Performance tier: {display_tier} ({mode})"
        )
        self.performance_tier_action.setToolTip(selection_reason)
        self.tray_icon.showMessage(
            "Smart File Finder performance tier",
            f"Using the {display_tier} tier ({'automatic' if mode == 'auto' else 'manual'}). "
            "The tier controls which local models are required.",
            QSystemTrayIcon.MessageIcon.Information,
            5000,
        )
        try:
            missing = missing_models_for_tier(tier)
            minimum_free_disk = (
                get_min_free_disk_gb_for_tier(tier) if missing else None
            )
        except Exception as error:
            log(f"Model registry check failed: {error}")
            QMessageBox.critical(
                None,
                "Model setup error",
                f"Could not check the required local models: {error}",
            )
            self.app.quit()
            return

        if missing:
            if (
                minimum_free_disk is not None
                and free_disk_gb is not None
                and free_disk_gb < minimum_free_disk
            ):
                message = (
                    f"The {get_tier_display_name(tier)} performance setup "
                    f"needs at least {minimum_free_disk} GB free disk space. "
                    f"This PC has {free_disk_gb:.1f} GB available."
                )
                log(f"Model setup blocked: {message}")
                QMessageBox.warning(
                    None,
                    "Not enough free disk space",
                    message,
                )
                self.app.quit()
                return
            set_huggingface_offline(False)
            self.status_window.show_status("Waiting for model setup...")
            self.setup_window = SetupWindow(
                tier,
                missing,
                free_disk_gb,
                selection_reason=selection_reason,
                selection_mode=mode,
            )
            self.setup_window.status_changed.connect(
                self.status_window.show_status
            )
            self.setup_window.ready.connect(self._on_models_ready)
            self.setup_window.cancelled.connect(self.app.quit)
            self.setup_window.show()
            return

        self._on_models_ready()

    def _on_models_ready(self):
        set_huggingface_offline(True)
        log("All required models are local; Hugging Face offline mode enabled.")
        if self._normal_startup_started:
            return
        self._normal_startup_started = True
        self.status_window.show_status("Preparing to load local models...")
        self.status_window.show()
        self._indexing_thread = threading.Thread(
            target=self._run_initial_backfill,
            daemon=True,
        )
        self._indexing_thread.start()

    def _run_initial_backfill(self):
        self._preload_embedding_model()
        self.progress_bridge.model_loading.emit("Starting initial indexing...")

        backfill(
            progress_callback=lambda folder, filename, done, total:
                self.progress_bridge.progress_updated.emit(folder, filename, done, total),
            error_callback=lambda filename:
                self.progress_bridge.error_occurred.emit(filename),
            status_callback=lambda message:
                self.progress_bridge.notice.emit(message),
            model_status_callback=lambda message:
                self.progress_bridge.model_loading.emit(message),
        )
        print("Indexing complete.")
        self.indexing_done_bridge.finished.emit()

    def _on_indexing_finished(self):
        self._indexing_complete = True
        self._start_watchers()
        self.status_window.mark_complete()
        self.status_window.show()
        self._register_search_hotkeys()
        self._search_hotkeys_registered = True
        print(f"Smart File Finder running. Press {DEFAULT_HOTKEY} to search.")

    def _register_search_hotkeys(self):
        hold_ms = int(self._load_voice_setting("voice_hold_ms", 500))
        if self.voice_enabled:
            voice_key = str(
                self._load_voice_setting("voice_key", "left alt")
            ).strip().casefold()
            if voice_key not in {"left alt", "right alt"}:
                log(
                    f"WARNING: Invalid voice_key {voice_key!r}; "
                    "using left alt"
                )
                voice_key = "left alt"
            log_voice(
                f"registering voice key={voice_key!r} "
                f"hold_threshold_ms={hold_ms}"
            )
            self._hotkey_unregister = register_tap_hold_hotkey(
                self._on_voice_hotkey_event,
                hold_threshold_ms=hold_ms,
                max_hold_ms=int(
                    self._load_voice_setting("voice_max_seconds", 15)
                ) * 1000,
                voice_key=voice_key,
                cancel_event=self._voice_overlay_active,
            )
        else:
            log_voice(
                f"voice hook not installed; enabled={self.voice_enabled}"
            )
            handle = register_hotkey(self._show_input_bar, DEFAULT_HOTKEY)
            self._hotkey_unregister = lambda: unregister_hotkey(handle)

    def _start_watchers(self):
        folders = load_indexed_folders()
        handler = LiveIndexHandler(
            status_callback=self.progress_bridge.notice.emit,
            model_status_callback=self.progress_bridge.model_loading.emit,
        )
        for folder in folders:
            if not Path(folder).is_dir():
                log(f"WARNING: Not watching unreachable folder: {folder}")
                continue
            observer = Observer()
            observer.schedule(handler, folder, recursive=True)
            observer.daemon = True
            observer.start()
            print(f"Watching for changes in: {folder}")

    def _on_voice_hotkey_event(self, gesture):
        log_voice(f"controller received gesture={gesture.value}")
        if gesture == VoiceGesture.TAP:
            self.hotkey_bridge.triggered.emit()
        elif gesture == VoiceGesture.HOLD_START:
            self.voice_bridge.state.emit("hold_start")
        elif gesture == VoiceGesture.HOLD_END:
            self.voice_bridge.state.emit("hold_end")
        elif gesture == VoiceGesture.CANCEL:
            self.voice_bridge.state.emit("cancel")

    def _on_voice_state(self, state):
        log_voice(f"controller state={state}")
        if state == "hold_start":
            self._start_voice_recording()
        elif state == "hold_end":
            self._finish_voice_recording()
        elif state == "cancel":
            self._cancel_voice_session()

    def _start_voice_recording(self):
        if not self.voice_enabled or self._recording:
            log_voice(
                "recorder start skipped "
                f"enabled={self.voice_enabled} recording={self._recording}"
            )
            return
        if self._voice_setup_window is not None:
            return
        if self._voice_tier is None:
            self._on_voice_error("Voice search is not available yet")
            return
        model = missing_voice_model_for_tier(self._voice_tier)
        if model is not None:
            log_voice(
                f"voice model missing tier={self._voice_tier}; "
                "opening model setup"
            )
            self._open_required_model_setup(
                self._on_voice_model_ready,
            )
            return
        try:
            model_entry = get_voice_model_for_tier(self._voice_tier)
            model_path = None
            if model_entry is not None:
                from app_paths import find_model
                model_path = find_model(model_entry["repo_id"])
            if model_path is None:
                log_voice(
                    f"voice model unavailable tier={self._voice_tier}"
                )
                raise VoiceRecorderError(
                    "Voice model is missing. Voice stays off until it is downloaded."
                )
            log_voice(f"voice model path found path={model_path}")
            if self.voice_transcriber is None:
                self.voice_transcriber = VoiceTranscriber(
                    model_path=model_path,
                    model_name=model_entry["name"],
                    model_status_callback=(
                        self.progress_bridge.model_loading.emit
                    ),
                )
            recorder_started_at = time.perf_counter()
            self.voice_recorder.start()
            log_voice(
                "recorder start "
                f"seconds={time.perf_counter() - recorder_started_at:.3f} "
                f"samples=0 sample_rate={self.voice_recorder.SAMPLE_RATE}"
            )
            self._recording = True
            self._start_live_transcription()
            self.input_bar.hide()
            self._voice_overlay_active.set()
            self._pending_voice_transcript = None
            self.voice_overlay.show_listening()
        except VoiceRecorderError as error:
            log_voice_exception(f"recorder start failed: {error}")
            self._on_voice_error(str(error))
        except Exception as error:
            log_voice_exception(f"recorder start failed: {error}")
            self._on_voice_error(
                "Could not open the microphone. Check your audio device."
            )

    def _on_voice_model_ready(self):
        self._voice_setup_window = None
        self._start_voice_recording()

    def _on_voice_setup_cancelled(self):
        self._voice_setup_window = None
        self.voice_enabled = False
        self._save_voice_setting("voice_enabled", False)
        self._update_voice_tray_action()
        self._on_voice_error("Voice search: cancelled")

    def _finish_voice_recording(self):
        if not self._recording:
            log_voice("recorder stop skipped; no active recording")
            return
        self._recording = False
        self._live_transcription_stop.set()
        recorder_stopped_at = time.perf_counter()
        try:
            audio = self.voice_recorder.stop()
        except Exception as error:
            log_voice_exception(f"recorder stop failed: {error}")
            self._on_voice_error(str(error))
            return
        if audio is None:
            log_voice(
                "recorder stop "
                f"seconds={time.perf_counter() - recorder_stopped_at:.3f} "
                "samples=0"
            )
            self._show_voice_message("Didn't catch that", 1500)
            return
        sample_count = len(audio)
        sample_rate = self.voice_recorder.SAMPLE_RATE
        log_voice(
            "recorder stop "
            f"seconds={time.perf_counter() - recorder_stopped_at:.3f} "
            f"samples={sample_count} "
            f"recorded_seconds={sample_count / sample_rate:.3f}"
        )
        self.voice_overlay.show_transcribing()
        worker = threading.Thread(
            target=self._transcribe_voice_audio,
            args=(audio, self._live_transcription_thread),
            daemon=True,
            name="voice-transcription",
        )
        worker.start()

    def _start_live_transcription(self):
        stop_event = threading.Event()
        self._live_transcription_stop = stop_event
        self._live_transcription_thread = threading.Thread(
            target=self._transcribe_voice_live,
            args=(stop_event,),
            daemon=True,
            name="voice-live-transcription",
        )
        self._live_transcription_thread.start()

    def _transcribe_voice_live(self, stop_event):
        last_sample_count = 0
        minimum_samples = int(
            self.voice_recorder.SAMPLE_RATE
            * LIVE_TRANSCRIPTION_MIN_NEW_SECONDS
        )
        while not stop_event.wait(
            LIVE_TRANSCRIPTION_INTERVAL_SECONDS
        ):
            try:
                audio = self.voice_recorder.snapshot()
                if audio is None:
                    continue
                sample_count = len(audio)
                if sample_count - last_sample_count < minimum_samples:
                    audio.fill(0)
                    continue
                last_sample_count = sample_count
                try:
                    text = self.voice_transcriber.transcribe_partial(audio)
                finally:
                    audio.fill(0)
                if text and not stop_event.is_set():
                    self.voice_bridge.partial_text.emit(text)
            except Exception:
                log_voice_exception("live transcription update failed")
                break

    def _on_voice_partial_text(self, text):
        if not self._recording or not text:
            return
        self.voice_overlay.show_live_transcript(text)
        self._restore_background_status()

    def _transcribe_voice_audio(self, audio, live_thread=None):
        transcription_started_at = time.perf_counter()
        log_voice(f"transcription start samples={len(audio)}")
        try:
            if (
                live_thread is not None
                and live_thread is not threading.current_thread()
            ):
                live_thread.join()
            text = self.voice_transcriber.transcribe(audio)
        except Exception:
            log_voice_exception("transcription failed")
            self.voice_bridge.error.emit(
                "Could not transcribe that recording."
            )
            return
        finally:
            try:
                audio.fill(0)
            except Exception:
                log_voice_exception("could not clear in-memory voice audio")
        log_voice(
            f"transcription end "
            f"seconds={time.perf_counter() - transcription_started_at:.3f} "
            f"text={text!r}"
        )
        self.voice_bridge.text.emit(text)

    def _on_voice_text(self, text):
        text = str(text or "").strip()
        log_voice(f"recognized text={text!r}")
        self._restore_background_status()
        if not text:
            self._show_voice_message("Didn't catch that", 1500)
            return
        self._pending_voice_transcript = text
        self._pending_voice_auto_search = bool(
            self._load_voice_setting("voice_auto_search", True)
        )
        self.voice_overlay.show_transcript(text)

    def _on_voice_error(self, message):
        self._cancel_recording()
        self._restore_background_status()
        log_voice(f"voice error message={message!r}")
        self._show_voice_message(message, 2000)

    def _restore_background_status(self):
        if getattr(self, "_indexing_complete", False):
            self.status_window.mark_complete()

    def _cancel_voice(self):
        self._cancel_voice_session()

    def _cancel_recording(self):
        stop_event = getattr(self, "_live_transcription_stop", None)
        if stop_event is not None:
            stop_event.set()
        if not self._recording:
            return
        self._recording = False
        try:
            self.voice_recorder.cancel()
        except Exception:
            log_voice_exception("could not discard cancelled voice recording")

    def _cancel_voice_session(self):
        log_voice("voice session cancelled")
        self._cancel_recording()
        self._pending_voice_transcript = None
        self._voice_overlay_active.clear()
        finish_session = getattr(
            self._hotkey_unregister, "finish_voice_session", None
        )
        if finish_session is not None:
            finish_session()
        self.voice_overlay.fade_out()

    def _show_voice_message(self, message, duration_ms):
        log_voice(
            f"voice message shown text={message!r} duration_ms={duration_ms}"
        )
        self._pending_voice_transcript = None
        self._voice_overlay_active.set()
        self.voice_overlay.show_message(message, duration_ms)

    def _on_voice_overlay_dismissed(self):
        transcript = self._pending_voice_transcript
        log_voice(f"overlay dismissed pending_transcript={transcript!r}")
        if transcript is None:
            self._voice_overlay_active.clear()
            finish_session = getattr(
                self._hotkey_unregister, "finish_voice_session", None
            )
            if finish_session is not None:
                finish_session()
            return

        auto_search = self._pending_voice_auto_search
        self._pending_voice_transcript = None
        parsed = parse_voice_command(transcript)
        log_voice(
            f"parsed command={parsed.command.value} query={parsed.query!r} "
            f"auto_search={auto_search}"
        )
        if parsed.command == Command.OPEN and not parsed.query:
            log_voice("action=prompt_for_open_query")
            self._show_voice_message(
                "Say what to open, for example: open corrective rag",
                2000,
            )
            return

        if not auto_search:
            log_voice("action=show_editable_input_bar")
            self._pending_voice_transcript = transcript
            self.input_bar.text_input.setText(transcript)
            self.input_bar.show_and_focus()
            self._voice_overlay_active.clear()
            finish_session = getattr(
                self._hotkey_unregister, "finish_voice_session", None
            )
            if finish_session is not None:
                finish_session()
            return

        self._voice_overlay_active.clear()
        finish_session = getattr(
            self._hotkey_unregister, "finish_voice_session", None
        )
        if finish_session is not None:
            finish_session()
        log_voice(f"action=dispatch_{parsed.command.value}")
        self._dispatch_command(parsed, voice_feedback=True)

    def _show_input_bar(self):
        self.hotkey_bridge.triggered.emit()

    def _actually_show_input_bar(self):
        self.input_bar.show_and_focus()

    def _on_query_submitted(self, raw_text):
        if (
            self._pending_voice_transcript is not None
            and raw_text == self._pending_voice_transcript
        ):
            self._pending_voice_transcript = None
            parsed = parse_voice_command(raw_text)
            log_voice(
                f"parsed command={parsed.command.value} query={parsed.query!r} "
                "source=editable_voice_transcript"
            )
            self._dispatch_command(parsed, voice_feedback=True)
            return
        self._pending_voice_transcript = None
        try:
            parsed = parse_command(raw_text)
        except ValueError as e:
            QMessageBox.warning(self.input_bar, "Invalid command", str(e))
            return

        self._dispatch_command(parsed)

    def _dispatch_command(self, parsed, voice_feedback=False):
        if not parsed.query:
            return

        result = handle_query(parsed.query)

        if result.get("action") == "options":
            log_voice("action result=options; awaiting a user choice")
            self.options_group = OptionsPopupGroup(
                result["options"], anchor=self.input_bar
            )
            self.options_group.choice_made.connect(self._on_option_chosen)
            self.options_group.show()
            if voice_feedback:
                self._show_voice_feedback("Not sure which file - pick one")
            return

        if result.get("action") != "answer" or not result.get("found"):
            log_voice("action result=no_match")
            if voice_feedback:
                self._show_voice_feedback("No match found")
            QMessageBox.information(
                self.input_bar,
                "No match found",
                result.get(
                    "message",
                    f"No confident match found. ({result['reason']})",
                )
            )
            return

        self.result_card = ResultCard(
            file_path=Path(result["path"]).resolve(),
            score=result["score"],
            explanation=result["explanation"],
            anchor=self.input_bar,
        )

        filename = Path(result["path"]).name
        if parsed.command == Command.OPEN:
            log_voice("action invoked=open")
            self.result_card._handle_open()
            if voice_feedback:
                self._show_voice_feedback(f"Opening {filename}")
        elif parsed.command == Command.SHOW:
            log_voice("action invoked=show")
            self.result_card._handle_show()
            if voice_feedback:
                self._show_voice_feedback(
                    f"Showing {filename} in Explorer"
                )
        else:
            log_voice("action invoked=find")
            self.result_card.show()

    def _show_voice_feedback(self, text):
        log_voice(f"voice action feedback text={text!r}")
        self.voice_overlay.show_feedback(text, duration_ms=1000)

    def _on_option_chosen(self, action, file_path):
        self.input_bar.hide()
        target = Path(file_path).resolve()
        try:
            if action == "open":
                open_file(target)
            else:
                reveal_in_explorer(target)
        except Exception as error:
            QMessageBox.warning(
                self.input_bar,
                "Error",
                f"Could not {action} {target}: {error}",
            )

    def run(self):
        print("Starting Smart File Finder...")
        sys.exit(self.app.exec())


if __name__ == "__main__":
    print(f"[TIMING] About to create app at {time.time()}")
    SmartFileFinderApp().run()