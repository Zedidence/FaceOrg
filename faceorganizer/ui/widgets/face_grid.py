"""FaceGrid — a scrollable, flow-layout grid of FaceTile widgets."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget, QScrollArea

from faceorganizer.ui.widgets.face_tile import FaceTile
from faceorganizer.ui.widgets.flow_layout import FlowLayout

_PAGE_SIZE = 200  # tiles rendered per page; keeps Qt widget count bounded


class FaceGrid(QScrollArea):
    """Scrollable flow grid of face thumbnails.

    Emits face_clicked(face_id) when a tile is clicked.

    To avoid creating thousands of Qt widgets for large clusters, tiles are
    rendered in pages of _PAGE_SIZE.  All face metadata is stored in memory
    (cheap — plain dicts); only the visible widgets are created.  A "Load
    more" button appears at the bottom when more pages are available.
    """

    face_clicked = Signal(int)

    def __init__(self, tile_size: int = 100, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tile_size = tile_size
        self._tiles: dict[int, FaceTile] = {}
        self._selected_face_id: int | None = None
        self._all_face_data: list[dict] = []
        self._cache = None
        self._loaded_count: int = 0

        self.setWidgetResizable(True)
        self.setObjectName("FaceGrid")

        # Outer widget: tiles container + load-more button stacked vertically.
        outer = QWidget()
        outer.setObjectName("FaceGridOuter")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(4)

        self._container = QWidget()
        self._container.setObjectName("FaceGridContainer")
        self._layout = FlowLayout(self._container, h_spacing=4, v_spacing=4)
        self._container.setLayout(self._layout)
        outer_layout.addWidget(self._container)

        self._load_more_btn = QPushButton("")
        self._load_more_btn.setObjectName("FaceGridLoadMore")
        self._load_more_btn.hide()
        self._load_more_btn.clicked.connect(self._load_next_page)
        outer_layout.addWidget(self._load_more_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        outer_layout.addStretch()

        self.setWidget(outer)

    # ── Public API ──────────────────────────────────────────────────────────

    def load_faces(self, face_data: list[dict], cache) -> None:
        """Populate the grid from a list of face dicts.

        Each dict must have: face_id, photo_path, bbox_x, bbox_y, bbox_w, bbox_h.
        Only the first page of tiles is created immediately; the rest are
        rendered on demand via the "Load more" button.
        """
        self.clear()
        self._all_face_data = face_data
        self._cache = cache
        self._load_next_page()

    def clear(self) -> None:
        """Remove all tiles and reset paging state."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._tiles.clear()
        self._selected_face_id = None
        self._all_face_data = []
        self._cache = None
        self._loaded_count = 0
        self._load_more_btn.hide()

    def select_face(self, face_id: int | None) -> None:
        if self._selected_face_id is not None and self._selected_face_id in self._tiles:
            self._tiles[self._selected_face_id].set_selected(False)
        self._selected_face_id = face_id
        if face_id is not None and face_id in self._tiles:
            self._tiles[face_id].set_selected(True)

    # ── Internal ────────────────────────────────────────────────────────────

    def _load_next_page(self) -> None:
        batch = self._all_face_data[self._loaded_count:self._loaded_count + _PAGE_SIZE]
        for face in batch:
            fid = face["face_id"]
            px = self._cache.get(
                fid,
                face["photo_path"],
                face["bbox_x"],
                face["bbox_y"],
                face["bbox_w"],
                face["bbox_h"],
            )
            tile = FaceTile(fid, px, self._tile_size)
            tile.clicked.connect(self._on_tile_clicked)
            self._tiles[fid] = tile
            self._layout.addWidget(tile)

        self._loaded_count += len(batch)
        self._container.adjustSize()
        self._update_load_more_btn()

    def _update_load_more_btn(self) -> None:
        remaining = len(self._all_face_data) - self._loaded_count
        if remaining > 0:
            next_batch = min(_PAGE_SIZE, remaining)
            self._load_more_btn.setText(
                f"Load {next_batch:,} more  ({remaining:,} remaining)"
            )
            self._load_more_btn.show()
        else:
            self._load_more_btn.hide()

    def _on_tile_clicked(self, face_id: int) -> None:
        self.select_face(face_id)
        self.face_clicked.emit(face_id)
