"""Load and normalize face embeddings from the database."""

from __future__ import annotations

import sqlite3

import numpy as np

from faceorganizer.database.core import get_all_embeddings
from faceorganizer.logging_config import get_logger

log = get_logger("clustering.embeddings")


def load_normalized_embeddings(
    conn: sqlite3.Connection,
) -> tuple[list[int], np.ndarray]:
    """Load all face embeddings from the DB and L2-normalize them.

    Returns (face_ids, normalized_embeddings) where normalized_embeddings
    has shape (N, D) with unit-length rows. Returns empty arrays when no
    faces are stored.
    """
    face_ids, embeddings = get_all_embeddings(conn)
    if len(face_ids) == 0:
        log.info("No embeddings in database")
        return face_ids, embeddings

    log.info("Loaded %d embeddings (dim=%d)", len(face_ids), embeddings.shape[1])

    # L2-normalize each row so cosine distance == 1 - dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    embeddings = embeddings / norms

    return face_ids, embeddings
