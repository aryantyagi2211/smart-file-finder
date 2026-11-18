"""Consent-gated first-run setup for required local model files."""

import threading

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app_paths import find_model
from indexing.model_downloader import (
    ModelDownloadCancelled,
    download_model,
)


def _format_mb(byte_count):
    return f"{byte_count / 1_000_000:.0f} MB"


class _DownloadSignals(QObject):
    progress = pyqtSignal(int, int, int, float)
    finished = pyqtSignal(bool, str)


class SetupWindow(QWidget):
    ready = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, tier, models, free_disk_gb=None, parent=None):
        super().__init__(parent)
        self.tier = tier
        self.models = list(models)
        self._progress_values = [0 for _ in self.models]
        self._totals = [
            int(model.get("download_size_bytes") or sum(model.get("files", {}).values()))
            for model in self.models
        ]
        self._cancel_event = None
        self._worker = None
        self._busy = False
        self._cancel_after_worker = False

        self.setWindowTitle("Setting up Smart File Finder")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setMinimumWidth(560)
        self.setStyleSheet("""
            QWidget {
                color: #e9edf2;
                background-color: #171b22;
                font-family: Segoe UI, sans-serif;
                font-size: 13px;
            }
            QLabel#heading {
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#detail {
                color: #b7c0cc;
            }
            QLabel#modelName {
                font-weight: 600;
            }
            QProgressBar {
                background: #262d37;
                border: 0;
                border-radius: 5px;
                color: #e9edf2;
                height: 12px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #71c98b;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #2d3541;
                border: 1px solid #485363;
                border-radius: 7px;
                padding: 8px 18px;
            }
            QPushButton#download {
                background-color: #42885a;
                border-color: #61a876;
                font-weight: 700;
            }
            QPushButton:disabled {
                color: #86909d;
                background-color: #252a32;
            }
        """)

        self._signals = _DownloadSignals(self)
        self._signals.progress.connect(self._on_progress)
        self._signals.finished.connect(self._on_finished)
        self._model_bars = []
        self._model_details = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(12)
        heading = QLabel("Setting up Smart File Finder")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        names = ", ".join(model["name"] for model in self.models)
        total_bytes = sum(self._totals)
        free_text = (
            f"{free_disk_gb:.1f} GB"
            if free_disk_gb is not None
            else "unknown"
        )
        detail = QLabel(
            f"The {tier} tier needs these model files: {names}. "
            f"Total download: {_format_mb(total_bytes)}. "
            f"Free disk space: {free_text}."
        )
        detail.setObjectName("detail")
        detail.setWordWrap(True)
        layout.addWidget(detail)

        privacy = QLabel(
            "The only network use is downloading public model files from "
            "Hugging Face once. No user file, query or telemetry is ever "
            "sent. After setup the app works fully offline."
        )
        privacy.setObjectName("detail")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)

        for index, model in enumerate(self.models):
            name_label = QLabel(
                f"{model['name']}  ·  {_format_mb(self._totals[index])}"
            )
            name_label.setObjectName("modelName")
            layout.addWidget(name_label)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            layout.addWidget(bar)
            progress_label = QLabel(
                f"0 MB / {_format_mb(self._totals[index])} · waiting"
            )
            progress_label.setObjectName("detail")
            layout.addWidget(progress_label)
            self._model_bars.append(bar)
            self._model_details.append(progress_label)

        self.total_bar = QProgressBar()
        self.total_bar.setRange(0, 1000)
        self.total_bar.setValue(0)
        layout.addWidget(QLabel("Total setup progress"))
        layout.addWidget(self.total_bar)
        self.total_detail = QLabel(
            f"0 MB / {_format_mb(total_bytes)} · waiting"
        )
        self.total_detail.setObjectName("detail")
        layout.addWidget(self.total_detail)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ff8585; background: transparent;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)
        buttons.addWidget(self.cancel_button)
        self.download_button = QPushButton("Download")
        self.download_button.setObjectName("download")
        self.download_button.clicked.connect(self._start_downloads)
        buttons.addWidget(self.download_button)
        layout.addLayout(buttons)
        self.resize(620, self.sizeHint().height())

    def _start_downloads(self):
        if self._busy:
            return
        self.error_label.clear()
        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading...")
        self.cancel_button.setEnabled(True)
        self._busy = True
        self._cancel_after_worker = False
        self._cancel_event = threading.Event()
        self._worker = threading.Thread(
            target=self._download_models,
            daemon=True,
            name="model-download",
        )
        self._worker.start()

    def _download_models(self):
        try:
            for index, model in enumerate(self.models):
                if self._cancel_event.is_set():
                    raise ModelDownloadCancelled("Model download cancelled.")
                if find_model(model["repo_id"]) is not None:
                    self._signals.progress.emit(
                        index, self._totals[index], self._totals[index], 0.0
                    )
                    continue
                download_model(
                    model,
                    progress_cb=lambda done, total, speed, model_index=index:
                        self._signals.progress.emit(
                            model_index, done, total, speed
                        ),
                    cancel_event=self._cancel_event,
                )
            self._signals.finished.emit(True, "")
        except ModelDownloadCancelled:
            self._signals.finished.emit(False, "cancelled")
        except Exception as error:
            self._signals.finished.emit(False, str(error))

    def _on_progress(self, index, done, total, speed):
        if index < 0 or index >= len(self.models):
            return
        self._progress_values[index] = max(
            self._progress_values[index], min(int(done), self._totals[index])
        )
        current = self._progress_values[index]
        expected = max(1, self._totals[index])
        self._model_bars[index].setValue(int(current * 1000 / expected))
        speed_text = (
            f"{speed / 1_000_000:.1f} MB/s"
            if speed > 0 else "finishing"
        )
        self._model_details[index].setText(
            f"{_format_mb(current)} / {_format_mb(self._totals[index])} · "
            f"{speed_text}"
        )
        current_total = sum(self._progress_values)
        total_expected = max(1, sum(self._totals))
        self.total_bar.setValue(int(current_total * 1000 / total_expected))
        self.total_detail.setText(
            f"{_format_mb(current_total)} / {_format_mb(sum(self._totals))} · "
            f"{speed_text}"
        )

    def _on_finished(self, success, message):
        self._busy = False
        self._worker = None
        if self._cancel_after_worker or message == "cancelled":
            self.cancelled.emit()
            self.close()
            return
        if success:
            self.ready.emit()
            self.close()
            return
        self.error_label.setText(message)
        self.download_button.setEnabled(True)
        self.download_button.setText("Retry")

    def _cancel(self):
        if self._busy:
            self._cancel_after_worker = True
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Cancelling...")
            self._cancel_event.set()
            return
        self.cancelled.emit()
        self.close()
