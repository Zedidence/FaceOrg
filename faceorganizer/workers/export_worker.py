"""Background worker that exports photos to per-person folders."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot

from faceorganizer.config import get_db_path
from faceorganizer.database.schema import init_db
from faceorganizer.organizer.export import export_by_person
from faceorganizer.workers.base import BaseWorker


class ExportWorker(BaseWorker):
    """Wraps export_by_person() for execution in a QThread."""

    def __init__(
        self,
        scan_root: Path,
        output_dir: Path,
        symlink: bool = False,
    ) -> None:
        super().__init__()
        self._scan_root = scan_root
        self._output_dir = output_dir
        self._symlink = symlink

    @Slot()
    def run(self) -> None:
        self.started.emit()
        self.progress.emit(0, 0, "Exporting photos…")
        conn = None
        try:
            db_path = get_db_path(self._scan_root)
            conn = init_db(db_path)

            def on_progress(done: int, total: int) -> None:
                self.progress.emit(done, total, "")

            summary = export_by_person(
                conn,
                self._output_dir,
                symlink=self._symlink,
                on_progress=on_progress,
                stop_event=self._stop_event,
            )
            total_exported = sum(summary.values())
            self.finished.emit(
                {
                    "total_exported": total_exported,
                    "people": len(summary),
                    "output_dir": str(self._output_dir),
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if conn:
                conn.close()
