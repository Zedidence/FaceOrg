"""Constants and configuration for FaceOrganizer."""

from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif",
})

# Clustering defaults (tuned for ArcFace 512-dim embeddings)
DEFAULT_CLUSTER_THRESHOLD = 0.30  # cosine distance eps for DBSCAN on ArcFace embeddings
MIN_CLUSTER_SIZE = 3  # minimum faces to form a cluster (DBSCAN min_samples)
DEFAULT_MAYBE_THRESHOLD = 0.45  # cosine distance for "maybe X" suggestions (looser than eps)


def similarity_to_eps(similarity_pct: float) -> float:
    """Convert user-facing similarity % to cosine distance eps."""
    return 1.0 - (similarity_pct / 100.0)


def eps_to_similarity(eps: float) -> float:
    """Convert cosine distance eps to user-facing similarity %."""
    return (1.0 - eps) * 100.0

# Face detection confidence threshold
MIN_DETECTION_CONFIDENCE = 0.9

# Minimum face bounding box size in pixels (width or height).
# Detections smaller than this are almost always false positives.
MIN_FACE_SIZE = 40

# Internal data directory name (created inside the scanned folder)
DATA_DIR_NAME = ".faceorganizer"
DB_FILENAME = "faces.db"
THUMBNAIL_DIR_NAME = "thumbnails"
THUMBNAIL_SIZE = (150, 150)


def get_data_dir(scan_root: Path) -> Path:
    """Return the .faceorganizer data directory for a given scan root."""
    return scan_root / DATA_DIR_NAME


def get_db_path(scan_root: Path) -> Path:
    """Return the SQLite database path for a given scan root."""
    return get_data_dir(scan_root) / DB_FILENAME


def get_thumbnail_dir(scan_root: Path) -> Path:
    """Return the thumbnail cache directory for a given scan root."""
    return get_data_dir(scan_root) / THUMBNAIL_DIR_NAME
