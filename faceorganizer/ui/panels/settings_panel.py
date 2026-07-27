"""Settings panel — detection, clustering, and performance configuration."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from faceorganizer.app_settings import AppSettings
from faceorganizer.hardware import RuntimeProfile


class SettingsPanel(QWidget):
    """Shows configurable settings and read-only hardware info."""

    settings_saved = Signal()

    def __init__(
        self,
        settings: AppSettings,
        profile: RuntimeProfile,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._profile = profile
        self.setObjectName("SettingsPanel")
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("settingsScroll")
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)
        scroll.setWidget(content)

        # ── Hardware ────────────────────────────────────────────────────────
        hw_group = QGroupBox("Detected Hardware")
        hw_form = QFormLayout(hw_group)
        hw_label = QLabel(self._profile.summary())
        hw_label.setObjectName("hwSummaryLabel")
        hw_label.setWordWrap(True)
        hw_form.addRow(hw_label)
        layout.addWidget(hw_group)

        # ── Detection ───────────────────────────────────────────────────────
        det_group = QGroupBox("Detection")
        det_form = QFormLayout(det_group)

        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.1, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setDecimals(2)
        self._confidence_spin.setValue(self._settings.detection_confidence)
        det_form.addRow("Min confidence:", self._confidence_spin)

        self._face_size_spin = QSpinBox()
        self._face_size_spin.setRange(10, 500)
        self._face_size_spin.setValue(self._settings.min_face_size)
        det_form.addRow("Min face size (px):", self._face_size_spin)

        layout.addWidget(det_group)

        # ── Clustering ──────────────────────────────────────────────────────
        cl_group = QGroupBox("Clustering")
        cl_form = QFormLayout(cl_group)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.05, 0.99)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(self._settings.cluster_threshold)
        cl_form.addRow("Cluster threshold (eps):", self._threshold_spin)

        self._incremental_check = QCheckBox("Incremental (preserve existing clusters)")
        self._incremental_check.setChecked(self._settings.incremental_clustering)
        cl_form.addRow(self._incremental_check)

        layout.addWidget(cl_group)

        # ── Performance ─────────────────────────────────────────────────────
        perf_group = QGroupBox("Performance")
        perf_form = QFormLayout(perf_group)

        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 32)
        self._workers_spin.setValue(
            self._settings.worker_count or self._profile.recommended_workers
        )
        perf_form.addRow(
            f"Worker threads (recommended: {self._profile.recommended_workers}):",
            self._workers_spin,
        )

        layout.addWidget(perf_group)

        # ── Save ────────────────────────────────────────────────────────────
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("saveSettingsButton")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()

    def _save(self) -> None:
        self._settings.detection_confidence = self._confidence_spin.value()
        self._settings.min_face_size = self._face_size_spin.value()
        self._settings.cluster_threshold = self._threshold_spin.value()
        self._settings.incremental_clustering = self._incremental_check.isChecked()
        self._settings.worker_count = self._workers_spin.value()
        self._settings.save()
        self.settings_saved.emit()

    def refresh(self, settings: AppSettings) -> None:
        """Refresh the UI from updated settings."""
        self._settings = settings
        self._confidence_spin.setValue(settings.detection_confidence)
        self._face_size_spin.setValue(settings.min_face_size)
        self._threshold_spin.setValue(settings.cluster_threshold)
        self._incremental_check.setChecked(settings.incremental_clustering)
        self._workers_spin.setValue(settings.worker_count or self._profile.recommended_workers)
