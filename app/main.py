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

from app.hotkey_listener import register_hotkey, DEFAULT_HOTKEY
from app.input_bar import InputBar
from app.command_parser import parse_command, Command, should_auto_open
from app.query_handler import handle_query
from app.result_card import OptionsPopupGroup, ResultCard
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
    get_resource_path,
)
from app.hardware_profile import (
    choose_tier,
    detect_hardware,
    get_performance_tier_override,
    set_current_tier,
)
from app.setup_window import SetupWindow
from indexing.model_downloader import (
    missing_models_for_tier,
    set_huggingface_offline,
)


from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon

from app.indexing_status import IndexingStatusWidget
from indexing.embedder import preload_model
from observability.logger import log

import time
print(f"[TIMING] Script started at {time.time()}")

class HotkeyBridge(QObject):
    triggered = pyqtSignal()


class IndexingDoneBridge(QObject):
    finished = pyqtSignal()

class ProgressBridge(QObject):
    progress_updated = pyqtSignal(str, str, int, int)
    error_occurred = pyqtSignal(str)
    model_loading = pyqtSignal()
    notice = pyqtSignal(str)


class HardwareTierBridge(QObject):
    ready = pyqtSignal(str, str, object)


class LiveIndexHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            self._index(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._index(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_supported_path(event.src_path):
            remove_file_index(event.src_path, reason="file deleted")

    def on_moved(self, event):
        if event.is_directory:
            return
        if self._is_supported_path(event.src_path):
            remove_file_index(event.src_path, reason="file moved")
        self._index(event.dest_path, force=True)

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
        try:
            status = index_one_file(file_path, force=force)
            print(status)
        except Exception as e:
            print(f"[ERROR - live watch] {file_path} -> {e}")

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
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self.app.quit)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self.input_bar = InputBar(on_submit=self._on_query_submitted)
        self.result_card = None
        self.options_group = None

        self.hotkey_bridge = HotkeyBridge()
        self.hotkey_bridge.triggered.connect(self._actually_show_input_bar)

        self.indexing_done_bridge = IndexingDoneBridge()
        self.indexing_done_bridge.finished.connect(self._on_indexing_finished)

        self.status_window = IndexingStatusWidget()

        self.progress_bridge = ProgressBridge()
        self.progress_bridge.progress_updated.connect(self.status_window.show_progress)
        self.progress_bridge.error_occurred.connect(self.status_window.show_error)
        self.progress_bridge.model_loading.connect(self.status_window.show_loading_model)
        self.progress_bridge.notice.connect(self.status_window.show_notice)

        self.setup_window = None
        self._normal_startup_started = False

        self.hardware_tier_bridge = HardwareTierBridge()
        self.hardware_tier_bridge.ready.connect(self._on_hardware_tier_ready)
        self._hardware_detection_thread = threading.Thread(
            target=self._detect_hardware_in_background,
            daemon=True,
            name="hardware-profile",
        )
        self._hardware_detection_thread.start()

    @staticmethod
    def _preload_embedding_model():
        try:
            preload_model()
        except Exception as error:
            log(f"Embedding model preload failed; it will retry on use: {error}")

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
            tier, mode, profile.get("free_disk_gb")
        )

    def _on_hardware_tier_ready(self, tier, mode, free_disk_gb):
        set_current_tier(tier)
        display_tier = tier.capitalize()
        self.performance_tier_action.setText(
            f"Performance tier: {display_tier} ({mode})"
        )
        try:
            missing = missing_models_for_tier(tier)
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
            set_huggingface_offline(False)
            self.setup_window = SetupWindow(tier, missing, free_disk_gb)
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
        self.status_window.show()
        self._embedding_warmup_thread = threading.Thread(
            target=self._preload_embedding_model,
            daemon=True,
        )
        self._embedding_warmup_thread.start()
        self._indexing_thread = threading.Thread(
            target=self._run_initial_backfill,
            daemon=True,
        )
        self._indexing_thread.start()

    def _run_initial_backfill(self):
        print("Indexing files, please wait...")
        self.progress_bridge.model_loading.emit()

        backfill(
            progress_callback=lambda folder, filename, done, total:
                self.progress_bridge.progress_updated.emit(folder, filename, done, total),
            error_callback=lambda filename:
                self.progress_bridge.error_occurred.emit(filename),
            status_callback=lambda message:
                self.progress_bridge.notice.emit(message),
        )
        print("Indexing complete.")
        self.indexing_done_bridge.finished.emit()

    def _on_indexing_finished(self):
        self.status_window.mark_complete()
        self.status_window.hide()
        self._start_watchers()
        register_hotkey(self._show_input_bar, DEFAULT_HOTKEY)
        print(f"Smart File Finder running. Press {DEFAULT_HOTKEY} to search.")

    def _start_watchers(self):
        folders = load_indexed_folders()
        handler = LiveIndexHandler()
        for folder in folders:
            if not Path(folder).is_dir():
                log(f"WARNING: Not watching unreachable folder: {folder}")
                continue
            observer = Observer()
            observer.schedule(handler, folder, recursive=True)
            observer.daemon = True
            observer.start()
            print(f"Watching for changes in: {folder}")

    def _show_input_bar(self):
        self.hotkey_bridge.triggered.emit()

    def _actually_show_input_bar(self):
        self.input_bar.show_and_focus()

    def _on_query_submitted(self, raw_text):
        try:
            parsed = parse_command(raw_text)
        except ValueError as e:
            QMessageBox.warning(self.input_bar, "Invalid command", str(e))
            return

        if not parsed.query:
            return

        result = handle_query(parsed.query)

        if result.get("action") == "options":
            self.options_group = OptionsPopupGroup(
                result["options"], anchor=self.input_bar
            )
            self.options_group.choice_made.connect(self._on_option_chosen)
            self.options_group.show()
            return

        if not result["found"]:
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
            file_path=result["path"],
            score=result["score"],
            explanation=result["explanation"],
            anchor=self.input_bar,
        )

        if should_auto_open(parsed.command, result.get("action")):
            self.result_card._handle_open()
        elif parsed.command == Command.SHOW:
            self.result_card._handle_show()
        else:
            self.result_card.show()

    def _on_option_chosen(self, action, file_path):
        self.input_bar.hide()
        target = (
            Path(file_path).resolve()
            if action == "open"
            else Path(file_path).resolve().parent
        )
        try:
            os.startfile(str(target))
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