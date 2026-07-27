"""Person detail panel — face grid + photo viewer for a single person."""

from __future__ import annotations

import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from faceorganizer import actions
from faceorganizer.app_settings import AppSettings
from faceorganizer.database.core import get_cluster_by_id, get_faces_for_cluster
from faceorganizer.ui.dialogs.merge_dialog import MergeDialog
from faceorganizer.ui.dialogs.rename_dialog import RenameDialog
from faceorganizer.ui.widgets.face_grid import FaceGrid
from faceorganizer.ui.widgets.photo_viewer import PhotoViewer
from faceorganizer.ui.widgets.thumbnail_cache import ThumbnailCache


class PersonDetailPanel(QWidget):
    """Deep-dive view for one person cluster."""

    back_requested = Signal()
    clusters_changed = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PersonDetailPanel")
        self._settings = settings
        self._conn: sqlite3.Connection | None = None
        self._cache: ThumbnailCache | None = None
        self._cluster_id: int | None = None
        self._faces: list[dict] = []
        self._all_faces_in_photo: dict[str, list[dict]] = {}  # photo_path → faces
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Header ────────────────────────────────────────────────────────
        header = QHBoxLayout()
        back_btn = QPushButton("← People")
        back_btn.setObjectName("backButton")
        back_btn.clicked.connect(self.back_requested)
        header.addWidget(back_btn)

        self._name_label = QLabel("")
        self._name_label.setObjectName("personDetailName")
        header.addWidget(self._name_label)
        header.addStretch()

        rename_btn = QPushButton("Rename…")
        rename_btn.clicked.connect(self._rename)
        merge_btn = QPushButton("Merge into…")
        merge_btn.clicked.connect(self._merge)
        recluster_btn = QPushButton("Recluster")
        recluster_btn.clicked.connect(self._recluster)

        for btn in (rename_btn, merge_btn, recluster_btn):
            header.addWidget(btn)

        layout.addLayout(header)

        # ── Main splitter: face grid (left) + photo viewer (right) ────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: face grid
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._face_count_label = QLabel("")
        self._face_count_label.setObjectName("faceCountLabel")
        left_layout.addWidget(self._face_count_label)
        self._grid = FaceGrid(tile_size=90)
        self._grid.face_clicked.connect(self._on_face_clicked)
        left_layout.addWidget(self._grid, stretch=1)
        splitter.addWidget(left)

        # Right: photo viewer
        self._viewer = PhotoViewer()
        splitter.addWidget(self._viewer)
        splitter.setSizes([300, 600])

        layout.addWidget(splitter, stretch=1)

        # Context menu for face grid
        self._grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._grid.customContextMenuRequested.connect(self._show_face_context_menu)

    def load(
        self,
        conn: sqlite3.Connection,
        cache: ThumbnailCache,
        cluster_id: int,
    ) -> None:
        self._conn = conn
        self._cache = cache
        self._cluster_id = cluster_id
        self._refresh()

    def _refresh(self) -> None:
        if self._conn is None or self._cluster_id is None:
            return

        cluster = get_cluster_by_id(self._conn, self._cluster_id)
        if cluster is None:
            self.back_requested.emit()
            return

        self._name_label.setText(cluster.name)
        self._faces = get_faces_for_cluster(self._conn, self._cluster_id)
        self._face_count_label.setText(f"{len(self._faces)} faces")
        self._grid.load_faces(self._faces, self._cache)
        self._viewer.clear()

    def _on_face_clicked(self, face_id: int) -> None:
        face = next((f for f in self._faces if f["face_id"] == face_id), None)
        if face is None:
            return

        # Load all faces for this photo (for bbox overlay context)
        photo_path = face["photo_path"]
        if photo_path not in self._all_faces_in_photo:
            cur = self._conn.execute(
                """SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h
                   FROM faces f JOIN photos p ON p.id = f.photo_id
                   WHERE p.path = ? AND f.dismissed = 0""",
                (photo_path,),
            )
            self._all_faces_in_photo[photo_path] = [
                {"face_id": r[0], "bbox_x": r[1], "bbox_y": r[2], "bbox_w": r[3], "bbox_h": r[4]}
                for r in cur.fetchall()
            ]

        self._viewer.load_photo(
            photo_path,
            self._all_faces_in_photo[photo_path],
            selected_face_id=face_id,
        )

    def _show_face_context_menu(self, pos) -> None:
        selected_id = self._grid._selected_face_id
        if selected_id is None:
            return
        menu = QMenu(self)
        menu.addAction("Dismiss face", lambda: self._dismiss_face(selected_id))
        menu.addAction("Split to new person", lambda: self._split_face(selected_id))
        menu.exec(self._grid.mapToGlobal(pos))

    def _dismiss_face(self, face_id: int) -> None:
        if self._conn is None:
            return
        try:
            actions.dismiss_face(self._conn, face_id)
        except actions.ActionError as e:
            QMessageBox.warning(self, "Dismiss Failed", str(e))
            return
        self.clusters_changed.emit()
        self._refresh()

    def _split_face(self, face_id: int) -> None:
        if self._conn is None:
            return
        dlg = RenameDialog("New Person", self)
        dlg.setWindowTitle("New Person Name")
        if dlg.exec():
            try:
                actions.split_face(self._conn, face_id, dlg.new_name())
            except actions.ActionError as e:
                QMessageBox.warning(self, "Split Failed", str(e))
                return
            self.clusters_changed.emit()
            self._refresh()

    def _rename(self) -> None:
        if self._conn is None or self._cluster_id is None:
            return
        cluster = get_cluster_by_id(self._conn, self._cluster_id)
        if cluster is None:
            return
        dlg = RenameDialog(cluster.name, self)
        if dlg.exec() and dlg.new_name():
            if not actions.rename_person(self._conn, self._cluster_id, dlg.new_name()):
                QMessageBox.warning(self, "Rename Failed", "Cluster not found.")
                return
            self.clusters_changed.emit()
            self._refresh()

    def _merge(self) -> None:
        if self._conn is None or self._cluster_id is None:
            return
        from faceorganizer.database.core import get_clusters
        clusters = get_clusters(self._conn)
        cluster = get_cluster_by_id(self._conn, self._cluster_id)
        dlg = MergeDialog(cluster.name if cluster else "", clusters, self._cluster_id, self)
        if dlg.exec():
            target = dlg.target_cluster_id()
            if target is not None:
                try:
                    actions.merge_people(self._conn, target, self._cluster_id)
                except actions.ActionError as e:
                    QMessageBox.warning(self, "Merge Failed", str(e))
                    return
                self.clusters_changed.emit()
                self.back_requested.emit()

    def _recluster(self) -> None:
        if self._conn is None or self._cluster_id is None:
            return
        try:
            result = actions.recluster_person(
                self._conn, self._cluster_id, eps=self._settings.cluster_threshold
            )
        except actions.ActionError as e:
            QMessageBox.warning(self, "Recluster Failed", str(e))
            return
        self.clusters_changed.emit()
        self._refresh()
        if result["new_clusters"] == 0 and result["noise"] == 0:
            QMessageBox.information(self, "Recluster", "Cluster is already cohesive — no changes made.")
        else:
            QMessageBox.information(
                self, "Recluster",
                f"Split into {result['new_clusters']} new sub-cluster(s), "
                f"{result['noise']} face(s) became unassigned.",
            )
