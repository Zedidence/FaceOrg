"""Face thumbnail cropping and caching."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from faceorganizer.config import THUMBNAIL_SIZE, get_thumbnail_dir
from faceorganizer.logging_config import get_logger

log = get_logger("web.thumbnails")


def _thumb_path(thumb_dir: Path, face_id: int) -> Path:
    return thumb_dir / f"face_{face_id}.jpg"


def _photo_thumb_path(thumb_dir: Path, photo_id: int) -> Path:
    return thumb_dir / f"photo_{photo_id}.jpg"


def get_or_create_thumbnail(
    scan_root: Path,
    face_id: int,
    photo_path: str,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
) -> Path | None:
    """Return the path to a cached face thumbnail, creating it if needed."""
    thumb_dir = get_thumbnail_dir(scan_root)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    dest = _thumb_path(thumb_dir, face_id)

    if dest.exists():
        return dest

    src = Path(photo_path)
    if not src.exists():
        log.warning("Source photo missing for thumbnail: %s", src)
        return None

    try:
        with Image.open(src) as img:
            img = img.convert("RGB")

            # Add 20% padding around the face for context
            pad_w = int(bbox_w * 0.2)
            pad_h = int(bbox_h * 0.2)
            left = max(0, bbox_x - pad_w)
            top = max(0, bbox_y - pad_h)
            right = min(img.width, bbox_x + bbox_w + pad_w)
            bottom = min(img.height, bbox_y + bbox_h + pad_h)

            crop = img.crop((left, top, right, bottom))
            crop.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            crop.save(dest, "JPEG", quality=85)
        return dest
    except Exception:
        log.exception("Failed to create thumbnail for face %d", face_id)
        # Remove partially-written file so it won't be served as corrupt
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return None


def get_or_create_photo_thumbnail(
    scan_root: Path, photo_id: int, photo_path: str
) -> Path | None:
    """Return the path to a cached whole-photo thumbnail, creating it if needed.

    Used for duplicate-group review, where photos are shown uncropped rather
    than as a face crop. Cached under a distinct "photo_<id>" key so it can't
    collide with the face-crop cache above.
    """
    thumb_dir = get_thumbnail_dir(scan_root)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    dest = _photo_thumb_path(thumb_dir, photo_id)

    if dest.exists():
        return dest

    src = Path(photo_path)
    if not src.exists():
        log.warning("Source photo missing for thumbnail: %s", src)
        return None

    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            img.save(dest, "JPEG", quality=85)
        return dest
    except Exception:
        log.exception("Failed to create thumbnail for photo %d", photo_id)
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return None
