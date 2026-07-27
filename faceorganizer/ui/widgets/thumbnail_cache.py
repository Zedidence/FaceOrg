"""Face thumbnail cache — returns QPixmap objects.

Ported from web/thumbnails.py. Adds an in-memory LRU QPixmap cache on top
of the existing on-disk JPEG cache.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PIL import Image
from PySide6.QtGui import QPixmap

from faceorganizer.config import get_thumbnail_dir
from faceorganizer.logging_config import get_logger

log = get_logger("ui.thumbnail_cache")

_MEMORY_CACHE_LIMIT = 500   # max QPixmap objects kept in RAM
_PADDING_FACTOR = 0.2        # 20% padding around each face crop


class ThumbnailCache:
    """Disk-backed, LRU-memory-cached face thumbnail provider."""

    def __init__(self, scan_root: Path, thumb_size: tuple[int, int] = (150, 150)) -> None:
        self._scan_root = scan_root
        self._thumb_size = thumb_size
        self._mem: OrderedDict[int, QPixmap] = OrderedDict()
        self._thumb_dir = get_thumbnail_dir(scan_root)
        self._thumb_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        face_id: int,
        photo_path: str,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
    ) -> QPixmap | None:
        """Return a QPixmap for the face, generating and caching it as needed."""
        # 1. In-memory LRU hit
        if face_id in self._mem:
            self._mem.move_to_end(face_id)
            return self._mem[face_id]

        # 2. On-disk JPEG hit
        disk_path = self._thumb_dir / f"face_{face_id}.jpg"
        if disk_path.exists():
            px = QPixmap(str(disk_path))
            if not px.isNull():
                self._put(face_id, px)
                return px

        # 3. Generate from source photo
        px = self._generate(face_id, photo_path, bbox_x, bbox_y, bbox_w, bbox_h, disk_path)
        if px is not None:
            self._put(face_id, px)
        return px

    def invalidate(self) -> None:
        """Clear the in-memory cache (e.g., after a new scan)."""
        self._mem.clear()

    # ── private ──────────────────────────────────────────────────────────────

    def _put(self, face_id: int, px: QPixmap) -> None:
        self._mem[face_id] = px
        self._mem.move_to_end(face_id)
        if len(self._mem) > _MEMORY_CACHE_LIMIT:
            self._mem.popitem(last=False)

    def _generate(
        self,
        face_id: int,
        photo_path: str,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        disk_path: Path,
    ) -> QPixmap | None:
        src = Path(photo_path)
        if not src.exists():
            log.warning("Source photo missing for thumbnail: %s", src)
            return None

        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                pad_w = int(bbox_w * _PADDING_FACTOR)
                pad_h = int(bbox_h * _PADDING_FACTOR)
                left = max(0, bbox_x - pad_w)
                top = max(0, bbox_y - pad_h)
                right = min(img.width, bbox_x + bbox_w + pad_w)
                bottom = min(img.height, bbox_y + bbox_h + pad_h)

                crop = img.crop((left, top, right, bottom))
                crop.thumbnail(self._thumb_size, Image.LANCZOS)
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(str(disk_path), "JPEG", quality=85)

            px = QPixmap(str(disk_path))
            return px if not px.isNull() else None
        except Exception:
            log.exception("Failed to generate thumbnail for face %d", face_id)
            if disk_path.exists():
                try:
                    disk_path.unlink()
                except OSError:
                    pass
            return None
