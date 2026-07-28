"""Core data models for FaceOrganizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class PhotoInfo:
    """Metadata about a scanned photo."""

    path: Path
    file_size: int
    width: int
    height: int
    format: str
    exif_date: datetime | None = None
    num_faces: int = 0
    id: int | None = None
    phash: str | None = None


@dataclass
class FaceInfo:
    """A single detected face within a photo."""

    photo_path: Path
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    embedding: np.ndarray
    detection_confidence: float
    cluster_id: int | None = None
    photo_id: int | None = None
    id: int | None = None


@dataclass
class PersonCluster:
    """A group of faces identified as the same person."""

    id: int
    name: str
    face_count: int = 0
    representative_face_id: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
