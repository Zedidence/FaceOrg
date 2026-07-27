"""People panel — DigiKam-style grid of person cards."""

from __future__ import annotations

import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QScrollArea, QVBoxLayout, QWidget,
)

from faceorganizer import actions
from faceorganizer.database.core import get_clusters, get_representative_face
from faceorganizer.models import PersonCluster
from faceorganizer.ui.dialogs.merge_dialog import MergeDialog
from faceorganizer.ui.dialogs.rename_dialog import RenameDialog
from faceorganizer.ui.widgets.flow_layout import FlowLayout
from faceorganizer.ui.widgets.thumbnail_cache import ThumbnailCache


class PeoplePanel(QWidget):
    """DigiKam People tab — flow grid of PersonCards."""

    person_selected = Signal(int)   # cluster_id
    clusters_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PeoplePanel")
        self._conn: sqlite3.Connection | None = None
        self._cache: ThumbnailCache | None = None
        self._clusters: list[PersonCluster] = []
        self._cards: list[PersonCard] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Search bar
        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search people…")
        self._search.setObjectName("peopleSearch")
        self._search.textChanged.connect(self._filter)
        search_row.addWidget(self._search)
        layout.addLayout(search_row)

        # Scroll area with flow layout
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setObjectName("peopleScroll")
        layout.addWidget(self._scroll, stretch=1)

        self._container = QWidget()
        self._container.setObjectName("peopleContainer")
        self._flow = FlowLayout(self._container, h_spacing=8, v_spacing=8)
        self._container.setLayout(self._flow)
        self._scroll.setWidget(self._container)

    def load(self, conn: sqlite3.Connection, cache: ThumbnailCache) -> None:
        self._conn = conn
        self._cache = cache
        self._refresh()

    def _refresh(self) -> None:
        if self._conn is None:
            return

        # Remove old cards
        while self._flow.count():
            item = self._flow.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        self._clusters = get_clusters(self._conn)
        filter_text = self._search.text().lower()

        tile_size = 120

        for cluster in self._clusters:
            if filter_text and filter_text not in cluster.name.lower():
                continue

            # Representative face thumbnail — single LIMIT 1 query per cluster,
            # not a full get_faces_for_cluster() load.
            face = get_representative_face(
                self._conn, cluster.id, cluster.representative_face_id
            )
            px: QPixmap | None = None
            if face:
                px = self._cache.get(
                    face["face_id"],
                    face["photo_path"],
                    face["bbox_x"], face["bbox_y"],
                    face["bbox_w"], face["bbox_h"],
                )

            card = PersonCard(cluster, px, tile_size)
            card.clicked.connect(self._on_card_clicked)
            card.rename_requested.connect(self._rename)
            card.merge_requested.connect(self._merge)
            card.dismiss_requested.connect(self._dismiss_cluster)
            self._flow.addWidget(card)
            self._cards.append(card)

        self._container.adjustSize()

    def _filter(self, text: str) -> None:
        text = text.lower()
        for card in self._cards:
            card.setVisible(not text or text in card.cluster_name.lower())

    def _on_card_clicked(self, cluster_id: int) -> None:
        self.person_selected.emit(cluster_id)

    def _rename(self, cluster_id: int) -> None:
        cluster = next((c for c in self._clusters if c.id == cluster_id), None)
        if cluster is None:
            return
        dlg = RenameDialog(cluster.name, self)
        if dlg.exec() and dlg.new_name():
            if not actions.rename_person(self._conn, cluster_id, dlg.new_name()):
                QMessageBox.warning(self, "Rename Failed", "Cluster not found.")
                return
            self.clusters_changed.emit()
            self._refresh()

    def _merge(self, cluster_id: int) -> None:
        dlg = MergeDialog(
            next((c.name for c in self._clusters if c.id == cluster_id), ""),
            self._clusters,
            cluster_id,
            self,
        )
        if dlg.exec():
            target = dlg.target_cluster_id()
            if target is not None:
                try:
                    actions.merge_people(self._conn, target, cluster_id)
                except actions.ActionError as e:
                    QMessageBox.warning(self, "Merge Failed", str(e))
                    return
                self.clusters_changed.emit()
                self._refresh()

    def _dismiss_cluster(self, cluster_id: int) -> None:
        try:
            actions.dismiss_cluster(self._conn, cluster_id)
        except actions.ActionError as e:
            QMessageBox.warning(self, "Dismiss Failed", str(e))
            return
        self.clusters_changed.emit()
        self._refresh()

    def refresh(self) -> None:
        self._refresh()


class PersonCard(QFrame):
    """Single person card: face thumbnail + name + face count."""

    clicked = Signal(int)           # cluster_id
    rename_requested = Signal(int)
    merge_requested = Signal(int)
    dismiss_requested = Signal(int)

    def __init__(
        self,
        cluster: PersonCluster,
        pixmap: QPixmap | None,
        tile_size: int = 120,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cluster_id = cluster.id
        self._cluster_name = cluster.name
        self.setObjectName("PersonCard")
        self.setFixedWidth(tile_size + 16)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Face thumbnail
        img = QLabel()
        img.setFixedSize(tile_size, tile_size)
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setObjectName("personCardThumb")
        if pixmap and not pixmap.isNull():
            img.setPixmap(
                pixmap.scaled(
                    tile_size, tile_size,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            img.setText("?")
        layout.addWidget(img)

        # Name
        name_label = QLabel(cluster.name)
        name_label.setObjectName("personCardName")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(False)
        name_label.setMaximumWidth(tile_size)
        layout.addWidget(name_label)

        # Count
        count_label = QLabel(f"{cluster.face_count} face{'s' if cluster.face_count != 1 else ''}")
        count_label.setObjectName("personCardCount")
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(count_label)

    @property
    def cluster_name(self) -> str:
        return self._cluster_name

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._cluster_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Rename…", lambda: self.rename_requested.emit(self._cluster_id))
        menu.addAction("Merge into…", lambda: self.merge_requested.emit(self._cluster_id))
        menu.addSeparator()
        menu.addAction("Dismiss all faces", lambda: self.dismiss_requested.emit(self._cluster_id))
        menu.exec(event.globalPos())
