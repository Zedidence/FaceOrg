"""FaceTile — a clickable face thumbnail widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class FaceTile(QWidget):
    """Displays a single face thumbnail. Emits clicked(face_id) on click."""

    clicked = Signal(int)        # face_id
    double_clicked = Signal(int)

    def __init__(self, face_id: int, pixmap: QPixmap | None, tile_size: int = 100, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._face_id = face_id
        self._selected = False
        self._tile_size = tile_size

        self.setFixedSize(tile_size, tile_size)
        self.setObjectName("FaceTile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._img_label = QLabel(self)
        self._img_label.setFixedSize(tile_size, tile_size)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setScaledContents(True)

        if pixmap and not pixmap.isNull():
            self._img_label.setPixmap(
                pixmap.scaled(tile_size, tile_size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self._img_label.setText("?")

    @property
    def face_id(self) -> int:
        return self._face_id

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._selected:
            painter = QPainter(self)
            painter.setPen(QColor("#3d7ab5"))
            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
            painter.drawRect(1, 1, self.width() - 3, self.height() - 3)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._face_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self._face_id)
        super().mouseDoubleClickEvent(event)
