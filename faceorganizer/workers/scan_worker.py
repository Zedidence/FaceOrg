"""Background worker that runs a face-detection scan."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot

from faceorganizer.config import get_db_path
from faceorganizer.database.schema import init_db
from faceorganizer.scanner.scan_runner import ScanProgress, run_scan
from faceorganizer.workers.base import BaseWorker


class ScanWorker(BaseWorker):
    """Wraps run_scan() for execution in a QThread."""

    def __init__(
        self,
        scan_root: Path,
        worker_count: int,
        detection_confidence: float,
        min_face_size: int,
    ) -> None:
        super().__init__()
        self._scan_root = scan_root
        self._worker_count = worker_count
        self._detection_confidence = detection_confidence
        self._min_face_size = min_face_size

    @Slot()
    def run(self) -> None:
        self.started.emit()
        conn = None
        try:
            # Apply detection settings before any detector is created.
            # Safe: single-user app, scan tasks are not run concurrently.
            import faceorganizer.scanner.face_detector as _fd

            _fd.MIN_DETECTION_CONFIDENCE = self._detection_confidence
            _fd.MIN_FACE_SIZE = self._min_face_size

            db_path = get_db_path(self._scan_root)
            conn = init_db(db_path)

            def on_progress(p: ScanProgress) -> None:
                self.progress.emit(p.processed, p.total, p.current_file)

            result = run_scan(
                conn,
                self._scan_root,
                max_workers=self._worker_count,
                on_progress=on_progress,
                stop_event=self._stop_event,
            )
            self.finished.emit(
                {
                    "processed": result.processed,
                    "faces_found": result.faces_found,
                    "errors": result.errors,
                    "cancelled": result.cancelled,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if conn:
                conn.close()
