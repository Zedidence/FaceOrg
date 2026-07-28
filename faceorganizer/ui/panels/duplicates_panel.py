"""Duplicate photos panel — browse duplicate-photo groups and delete unwanted copies."""

from __future__ import annotations

import sqlite3

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from faceorganizer import actions
from faceorganizer.database.core import get_duplicate_groups, get_photos_in_duplicate_group
from faceorganizer.ui.widgets.face_grid import FaceGrid
from faceorganizer.ui.widgets.thumbnail_cache import ThumbnailCache


class _PhotoThumbnailAdapter:
    """Adapts ThumbnailCache.get_photo() to FaceGrid's expected .get(...) signature.

    FaceGrid/FaceTile were built for face crops (keyed by face_id + bbox);
    reused here for whole-photo tiles rather than duplicating the grid/tile/
    paging logic for what is otherwise an identical widget. "face_id" in the
    dicts passed to load_faces() below is really a photo_id, and the bbox
    fields are unused placeholders this adapter discards.
    """

    def __init__(self, cache: ThumbnailCache) -> None:
        self._cache = cache

    def get(self, photo_id, photo_path, *_bbox):
        return self._cache.get_photo(photo_id, photo_path)


class DuplicatesPanel(QWidget):
    """Browse duplicate-photo groups (found via the Find Duplicates action)."""

    photos_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DuplicatesPanel")
        self._conn: sqlite3.Connection | None = None
        self._cache: ThumbnailCache | None = None
        self._groups: list[dict] = []
        self._group_index = 0
        self._selected_photo_id: int | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Previous")
        self._prev_btn.clicked.connect(self._prev_group)
        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._next_group)
        self._group_label = QLabel("")
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._group_label)
        nav.addStretch()
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._delete_selected)
        nav.addWidget(delete_btn)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        hint = QLabel(
            "Click a photo to select it, then Delete Selected to send it to the "
            "Recycle Bin. Use Operations → Find Duplicates to scan for groups."
        )
        hint.setObjectName("duplicatesHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._grid = FaceGrid(tile_size=120)
        self._grid.face_clicked.connect(self._on_photo_clicked)
        layout.addWidget(self._grid, stretch=1)

    def load(self, conn: sqlite3.Connection, cache: ThumbnailCache) -> None:
        self._conn = conn
        self._cache = cache
        self._groups = get_duplicate_groups(conn)
        self._group_index = 0
        self._show_group()

    def refresh(self) -> None:
        if self._conn is None:
            return
        self._groups = get_duplicate_groups(self._conn)
        if self._group_index >= len(self._groups):
            self._group_index = max(0, len(self._groups) - 1)
        self._show_group()

    def _show_group(self) -> None:
        self._selected_photo_id = None

        if not self._groups:
            self._group_label.setText("No duplicate groups found")
            self._grid.clear()
            return

        i = self._group_index % len(self._groups)
        group = self._groups[i]
        self._group_label.setText(
            f"Group {i + 1} / {len(self._groups)}  ({group['photo_count']} photos)"
        )

        photos = get_photos_in_duplicate_group(self._conn, group["id"])
        face_shaped = [
            {
                "face_id": p["photo_id"], "photo_path": p["path"],
                "bbox_x": 0, "bbox_y": 0, "bbox_w": 0, "bbox_h": 0,
            }
            for p in photos
        ]
        self._grid.load_faces(face_shaped, _PhotoThumbnailAdapter(self._cache))

    def _prev_group(self) -> None:
        if self._groups:
            self._group_index = (self._group_index - 1) % len(self._groups)
        self._show_group()

    def _next_group(self) -> None:
        if self._groups:
            self._group_index = (self._group_index + 1) % len(self._groups)
        self._show_group()

    def _on_photo_clicked(self, photo_id: int) -> None:
        self._selected_photo_id = photo_id

    def _delete_selected(self) -> None:
        if self._conn is None or self._selected_photo_id is None:
            return
        try:
            actions.delete_photo(self._conn, self._selected_photo_id)
        except actions.ActionError as e:
            QMessageBox.warning(self, "Delete Failed", str(e))
            return
        self.photos_changed.emit()
        self.refresh()
