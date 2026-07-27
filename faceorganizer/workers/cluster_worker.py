"""Background worker that runs face clustering."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot

from faceorganizer.clustering.cluster import run_clustering, run_incremental_clustering
from faceorganizer.config import DEFAULT_CLUSTER_THRESHOLD, get_db_path
from faceorganizer.database.schema import init_db
from faceorganizer.workers.base import BaseWorker


class ClusterWorker(BaseWorker):
    """Wraps run_clustering() / run_incremental_clustering() for a QThread."""

    def __init__(
        self,
        scan_root: Path,
        incremental: bool = True,
        eps: float = DEFAULT_CLUSTER_THRESHOLD,
    ) -> None:
        super().__init__()
        self._scan_root = scan_root
        self._incremental = incremental
        self._eps = eps

    @Slot()
    def run(self) -> None:
        self.started.emit()
        self.progress.emit(0, 0, "Clustering faces…")
        conn = None
        try:
            db_path = get_db_path(self._scan_root)
            conn = init_db(db_path)

            if self._incremental:
                result = run_incremental_clustering(conn, eps=self._eps)
                num_clusters = result.get("new_clusters", 0)
            else:
                num_clusters = run_clustering(conn, eps=self._eps)
                result = {"new_clusters": num_clusters}

            self.finished.emit(
                {
                    "num_clusters": num_clusters,
                    "incremental": self._incremental,
                    **result,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if conn:
                conn.close()
