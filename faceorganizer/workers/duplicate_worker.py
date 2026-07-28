"""Background worker that finds duplicate photos."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot

from faceorganizer.config import DEFAULT_DUPLICATE_HAMMING_THRESHOLD, get_db_path
from faceorganizer.database.schema import init_db
from faceorganizer.workers.base import BaseWorker


class DuplicateWorker(BaseWorker):
    """Wraps backfill_phashes() + run_duplicate_detection() for a QThread."""

    def __init__(
        self,
        scan_root: Path,
        hamming_threshold: int = DEFAULT_DUPLICATE_HAMMING_THRESHOLD,
    ) -> None:
        super().__init__()
        self._scan_root = scan_root
        self._hamming_threshold = hamming_threshold

    @Slot()
    def run(self) -> None:
        self.started.emit()
        self.progress.emit(0, 0, "Finding duplicate photos…")
        conn = None
        try:
            from faceorganizer.duplicates import run_duplicate_detection
            from faceorganizer.scanner.scan_runner import backfill_phashes

            db_path = get_db_path(self._scan_root)
            conn = init_db(db_path)

            def on_backfill_progress(done: int, total: int) -> None:
                self.progress.emit(done, total, "Computing photo hashes…")

            backfill_phashes(
                conn, on_progress=on_backfill_progress, stop_event=self._stop_event
            )
            num_groups = run_duplicate_detection(
                conn, hamming_threshold=self._hamming_threshold
            )

            self.finished.emit({"num_groups": num_groups})
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if conn:
                conn.close()
