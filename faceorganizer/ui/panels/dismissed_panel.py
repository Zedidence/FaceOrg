"""Dismissed panel — view and restore false-positive detections."""

from __future__ import annotations

import sqlite3

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from faceorganizer.database.core import get_dismissed_faces, restore_face
from faceorganizer.ui.widgets.face_grid import FaceGrid
from faceorganizer.ui.widgets.thumbnail_cache import ThumbnailCache


class DismissedPanel(QWidget):
    """Shows all dismissed faces with a Restore button."""

    faces_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DismissedPanel")
        self._conn: sqlite3.Connection | None = None
        self._cache: ThumbnailCache | None = None
        self._selected_face_id: int | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        header = QHBoxLayout()
        self._count_label = QLabel("Dismissed faces")
        header.addWidget(self._count_label)
        header.addStretch()
        restore_btn = QPushButton("Restore Selected")
        restore_btn.clicked.connect(self._restore_selected)
        header.addWidget(restore_btn)
        layout.addLayout(header)

        self._grid = FaceGrid(tile_size=90)
        self._grid.face_clicked.connect(self._on_face_clicked)
        layout.addWidget(self._grid, stretch=1)

    def load(self, conn: sqlite3.Connection, cache: ThumbnailCache) -> None:
        self._conn = conn
        self._cache = cache
        self._refresh()

    def _refresh(self) -> None:
        if self._conn is None:
            return
        faces = get_dismissed_faces(self._conn)
        self._count_label.setText(f"Dismissed faces  ({len(faces)})")
        self._grid.load_faces(faces, self._cache)

    def _on_face_clicked(self, face_id: int) -> None:
        self._selected_face_id = face_id

    def _restore_selected(self) -> None:
        if self._conn is None or self._selected_face_id is None:
            return
        restore_face(self._conn, self._selected_face_id)
        self._selected_face_id = None
        self.faces_changed.emit()
        self._refresh()
