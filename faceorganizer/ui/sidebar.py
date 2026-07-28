"""Sidebar panel with People tree and Views navigation."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from faceorganizer.models import PersonCluster

# Tree item data roles
_ITEM_TYPE_ROLE = Qt.ItemDataRole.UserRole        # "person" | "view" | "section"
_ITEM_ID_ROLE = Qt.ItemDataRole.UserRole + 1      # cluster_id or view name


class SidebarPanel(QWidget):
    """Left sidebar: People tree + Views navigation."""

    person_selected = Signal(int)    # cluster_id
    view_selected = Signal(str)      # "people" | "review" | "timeline" | "dismissed" | "settings"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SidebarPanel")
        self.setMinimumWidth(180)
        self.setMaximumWidth(320)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(0)

        title = QLabel("FaceOrganizer")
        title.setObjectName("sidebarTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._tree = QTreeWidget()
        self._tree.setObjectName("sidebarTree")
        self._tree.setHeaderHidden(True)
        self._tree.setAnimated(True)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree, stretch=1)

        self._stats_label = QLabel("")
        self._stats_label.setObjectName("sidebarStats")
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        self._populate_views()

    def _populate_views(self) -> None:
        self._tree.clear()

        # People section (populated by refresh_clusters)
        self._people_root = QTreeWidgetItem(["People"])
        self._people_root.setData(0, _ITEM_TYPE_ROLE, "section")
        self._people_root.setData(0, _ITEM_ID_ROLE, "people")
        self._tree.addTopLevelItem(self._people_root)
        self._people_root.setExpanded(True)

        # Views section
        views_root = QTreeWidgetItem(["Views"])
        views_root.setData(0, _ITEM_TYPE_ROLE, "section")
        self._tree.addTopLevelItem(views_root)
        views_root.setExpanded(True)

        for label, view_id in [
            ("All People", "people"),
            ("Review", "review"),
            ("Timeline", "timeline"),
            ("Dismissed", "dismissed"),
            ("Duplicates", "duplicates"),
            ("Settings", "settings"),
        ]:
            item = QTreeWidgetItem(views_root, [label])
            item.setData(0, _ITEM_TYPE_ROLE, "view")
            item.setData(0, _ITEM_ID_ROLE, view_id)

    def refresh_clusters(self, clusters: list[PersonCluster], stats: dict | None = None) -> None:
        """Rebuild the People subtree from the current cluster list."""
        # Remove all existing person children
        while self._people_root.childCount():
            self._people_root.removeChild(self._people_root.child(0))

        for cluster in clusters:
            item = QTreeWidgetItem(
                self._people_root,
                [f"{cluster.name}  ({cluster.face_count})"],
            )
            item.setData(0, _ITEM_TYPE_ROLE, "person")
            item.setData(0, _ITEM_ID_ROLE, cluster.id)

        self._people_root.setText(0, f"People  ({len(clusters)})")

        if stats:
            self._stats_label.setText(
                f"{stats.get('photos', 0)} photos  •  "
                f"{stats.get('faces', 0)} faces  •  "
                f"{stats.get('clusters', 0)} people"
            )

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        item_type = item.data(0, _ITEM_TYPE_ROLE)
        item_id = item.data(0, _ITEM_ID_ROLE)

        if item_type == "person":
            self.person_selected.emit(int(item_id))
        elif item_type == "view":
            self.view_selected.emit(str(item_id))

    def select_person(self, cluster_id: int) -> None:
        """Highlight the sidebar item for the given cluster."""
        for i in range(self._people_root.childCount()):
            child = self._people_root.child(i)
            if child.data(0, _ITEM_ID_ROLE) == cluster_id:
                self._tree.setCurrentItem(child)
                return
