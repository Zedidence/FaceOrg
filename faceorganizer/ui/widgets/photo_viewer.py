"""PhotoViewer — displays a full photo with face bounding box overlays."""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class PhotoViewer(QWidget):
    """Shows a scaled photo with face bounding boxes drawn via QPainter.

    Selected face: accent blue box.
    Other faces: white box at 60% opacity.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._orig_w: int = 0
        self._orig_h: int = 0
        self._faces: list[dict] = []        # list of face dicts: bbox_x/y/w/h, face_id
        self._selected_face_id: int | None = None

        self.setObjectName("PhotoViewer")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 200)

    def load_photo(self, photo_path: str, faces: list[dict], selected_face_id: int | None = None) -> None:
        """Load a photo and its faces. faces is a list of dicts with bbox_* keys."""
        px = QPixmap(photo_path)
        if px.isNull():
            self._pixmap = None
            self._faces = []
        else:
            self._pixmap = px
            self._orig_w = px.width()
            self._orig_h = px.height()
            self._faces = faces
        self._selected_face_id = selected_face_id
        self.update()

    def highlight_face(self, face_id: int | None) -> None:
        self._selected_face_id = face_id
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._faces = []
        self._selected_face_id = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if self._pixmap is None or self._pixmap.isNull():
            painter.fillRect(self.rect(), QColor("#1a1a1a"))
            painter.setPen(QColor("#606060"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No photo selected")
            return

        # Scale pixmap to fit widget while keeping aspect ratio
        widget_rect = self.rect()
        scaled = self._pixmap.scaled(
            widget_rect.width(),
            widget_rect.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Center the image
        x_off = (widget_rect.width() - scaled.width()) // 2
        y_off = (widget_rect.height() - scaled.height()) // 2

        painter.fillRect(widget_rect, QColor("#1a1a1a"))
        painter.drawPixmap(x_off, y_off, scaled)

        if not self._faces or self._orig_w == 0:
            return

        # Scale factors from original → displayed image
        sx = scaled.width() / self._orig_w
        sy = scaled.height() / self._orig_h

        for face in self._faces:
            fid = face.get("face_id")
            bx = int(face["bbox_x"] * sx) + x_off
            by = int(face["bbox_y"] * sy) + y_off
            bw = int(face["bbox_w"] * sx)
            bh = int(face["bbox_h"] * sy)

            if fid == self._selected_face_id:
                pen = QPen(QColor("#3d7ab5"), 2)
            else:
                col = QColor(255, 255, 255, 153)   # white, 60% opacity
                pen = QPen(col, 1)

            painter.setPen(pen)
            painter.drawRect(bx, by, bw, bh)
