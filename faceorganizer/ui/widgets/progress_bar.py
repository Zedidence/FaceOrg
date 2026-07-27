"""Status-bar progress widget shown during long operations."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget


class OperationProgressBar(QWidget):
    """Compact progress display for the main window status bar."""

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)

        self._label = QLabel("")
        self._label.setObjectName("progressLabel")

        self._bar = QProgressBar()
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setFixedWidth(180)
        self._bar.setTextVisible(False)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("cancelButton")
        self._cancel_btn.setFixedWidth(60)
        self._cancel_btn.clicked.connect(self.cancel_requested)

        layout.addWidget(self._label)
        layout.addWidget(self._bar)
        layout.addWidget(self._cancel_btn)

        self.hide()

    def start(self, label: str = "") -> None:
        self._label.setText(label)
        self._bar.setValue(0)
        self._bar.setMaximum(0)   # indeterminate initially
        self.show()

    def update_progress(self, processed: int, total: int, message: str = "") -> None:
        if message:
            self._label.setText(message)
        if processed > 0 and total > 0:
            # Switch to determinate mode only once real work has started.
            self._bar.setMaximum(total)
            self._bar.setValue(processed)
        elif total > 0 and processed == 0:
            # Discovery finished but first image not yet processed (models
            # downloading / DML shaders compiling).  Stay indeterminate so
            # the animation keeps running and the app doesn't look frozen.
            self._bar.setMaximum(0)
        else:
            self._bar.setMaximum(0)

    def stop(self) -> None:
        self.hide()
        self._bar.setValue(0)
        self._bar.setMaximum(100)
        self._label.setText("")
