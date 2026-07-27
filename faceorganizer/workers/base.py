"""Base worker class for Qt background operations.

Pattern: instantiate the worker, move it to a QThread, connect signals, start the thread.
The worker never touches any QWidget — only emits signals back to the UI thread.

Usage::

    thread = QThread()
    worker = ScanWorker(...)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    thread.start()
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot


class BaseWorker(QObject):
    """QObject-based worker designed to be moved to a QThread."""

    started = Signal()
    progress = Signal(int, int, str)   # processed, total, current_file
    finished = Signal(dict)            # result payload
    error = Signal(str)                # error message

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    @Slot()
    def run(self) -> None:
        raise NotImplementedError

    @Slot()
    def cancel(self) -> None:
        """Request cancellation. Thread-safe."""
        self._stop_event.set()
