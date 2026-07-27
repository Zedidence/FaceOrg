"""Dialog for merging one cluster into another."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget,
)

from faceorganizer.models import PersonCluster


class MergeDialog(QDialog):
    """Pick a target cluster to merge the current one into."""

    def __init__(
        self,
        source_name: str,
        clusters: list[PersonCluster],
        source_id: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Person")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Merge <b>{source_name}</b> into:"))

        self._combo = QComboBox()
        self._cluster_ids: list[int] = []
        for c in clusters:
            if c.id == source_id:
                continue
            self._combo.addItem(f"{c.name}  ({c.face_count} faces)")
            self._cluster_ids.append(c.id)

        layout.addWidget(self._combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def target_cluster_id(self) -> int | None:
        idx = self._combo.currentIndex()
        if idx < 0 or idx >= len(self._cluster_ids):
            return None
        return self._cluster_ids[idx]
