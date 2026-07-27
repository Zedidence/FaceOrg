"""Shared pytest fixtures and helpers for FaceOrganizer's test suite."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from faceorganizer.database.core import insert_faces_batch, insert_photo
from faceorganizer.database.schema import init_db
from faceorganizer.models import FaceInfo, PhotoInfo


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    """A fresh, empty database connection backed by a temp file."""
    c = init_db(tmp_path / "test.db")
    yield c
    c.close()


def make_embedding(seed: int, dim: int = 512) -> np.ndarray:
    """Deterministic unit-length embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def make_similar_embedding(base: np.ndarray, noise: float = 0.02) -> np.ndarray:
    """An embedding close to *base* (small cosine distance)."""
    rng = np.random.RandomState(abs(hash(base.tobytes())) % (2**31))
    perturbed = base + rng.randn(*base.shape).astype(np.float32) * noise
    return perturbed / np.linalg.norm(perturbed)


def add_photo(conn: sqlite3.Connection, path: str = "/fake/photo.jpg") -> int:
    return insert_photo(conn, PhotoInfo(
        path=path, file_size=1, width=100, height=100, format="JPEG",
    ))


def add_faces(conn: sqlite3.Connection, photo_id: int, embeddings, cluster_id=None) -> list[int]:
    """Insert faces for *photo_id* and return their new ids in insertion order."""
    before = {r[0] for r in conn.execute("SELECT id FROM faces").fetchall()}
    insert_faces_batch(conn, [
        FaceInfo(photo_path="/fake/photo.jpg", bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10,
                 embedding=e, detection_confidence=0.9, cluster_id=cluster_id)
        for e in embeddings
    ], photo_id)
    rows = conn.execute("SELECT id FROM faces ORDER BY id").fetchall()
    return [r[0] for r in rows if r[0] not in before]
