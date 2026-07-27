"""Review panel — side-by-side cluster comparison."""

from __future__ import annotations

import sqlite3

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from faceorganizer import actions
from faceorganizer.database.core import get_clusters
from faceorganizer.models import PersonCluster
from faceorganizer.ui.widgets.face_grid import FaceGrid
from faceorganizer.ui.widgets.thumbnail_cache import ThumbnailCache


class ReviewPanel(QWidget):
    """Side-by-side view of cluster pairs for merge/dismiss decisions."""

    clusters_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ReviewPanel")
        self._conn: sqlite3.Connection | None = None
        self._cache: ThumbnailCache | None = None
        self._clusters: list[PersonCluster] = []
        self._pair_index = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Navigation header
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Previous")
        self._prev_btn.clicked.connect(self._prev_pair)
        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._next_pair)
        self._pair_label = QLabel("")
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._pair_label)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        # Side-by-side panes
        self._splitter = QSplitter()
        self._left_pane = ClusterPane()
        self._right_pane = ClusterPane()
        self._splitter.addWidget(self._left_pane)
        self._splitter.addWidget(self._right_pane)
        layout.addWidget(self._splitter, stretch=1)

        # Merge button
        merge_btn = QPushButton("Merge Left into Right")
        merge_btn.clicked.connect(self._merge)
        layout.addWidget(merge_btn)

    def load(self, conn: sqlite3.Connection, cache: ThumbnailCache) -> None:
        self._conn = conn
        self._cache = cache
        self._clusters = get_clusters(conn)
        self._pair_index = 0
        self._show_pair()

    def _show_pair(self) -> None:
        if len(self._clusters) < 2:
            self._pair_label.setText("Not enough clusters to compare")
            return
        i = self._pair_index
        c1 = self._clusters[i % len(self._clusters)]
        c2 = self._clusters[(i + 1) % len(self._clusters)]
        self._pair_label.setText(f"Pair {i + 1} / {len(self._clusters)}")

        from faceorganizer.database.core import get_faces_for_cluster
        faces1 = get_faces_for_cluster(self._conn, c1.id)
        faces2 = get_faces_for_cluster(self._conn, c2.id)
        self._left_pane.load(c1, faces1[:20], self._cache)
        self._right_pane.load(c2, faces2[:20], self._cache)

    def _prev_pair(self) -> None:
        self._pair_index = max(0, self._pair_index - 1)
        self._show_pair()

    def _next_pair(self) -> None:
        self._pair_index = min(len(self._clusters) - 2, self._pair_index + 1)
        self._show_pair()

    def _merge(self) -> None:
        if self._conn is None or len(self._clusters) < 2:
            return
        i = self._pair_index
        c1 = self._clusters[i % len(self._clusters)]
        c2 = self._clusters[(i + 1) % len(self._clusters)]
        try:
            actions.merge_people(self._conn, c2.id, c1.id)
        except actions.ActionError as e:
            QMessageBox.warning(self, "Merge Failed", str(e))
            return
        self.clusters_changed.emit()
        self._clusters = get_clusters(self._conn)
        self._pair_index = min(self._pair_index, max(0, len(self._clusters) - 2))
        self._show_pair()


class ClusterPane(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title = QLabel("")
        self._title.setObjectName("clusterPaneTitle")
        layout.addWidget(self._title)
        self._grid = FaceGrid(tile_size=80)
        layout.addWidget(self._grid, stretch=1)

    def load(self, cluster: PersonCluster, faces: list[dict], cache: ThumbnailCache) -> None:
        self._title.setText(f"{cluster.name}  ({cluster.face_count} faces)")
        self._grid.load_faces(faces, cache)
