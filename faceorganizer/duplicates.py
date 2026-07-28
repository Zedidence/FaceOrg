"""Perceptual near-duplicate photo detection.

Mirrors faceorganizer/clustering/cluster.py's run_clustering() shape almost
exactly, but groups *photos* by perceptual-hash Hamming distance instead of
*faces* by embedding cosine distance.
"""

from __future__ import annotations

import sqlite3

import imagehash
import numpy as np
from sklearn.cluster import DBSCAN

from faceorganizer.config import DEFAULT_DUPLICATE_HAMMING_THRESHOLD, MIN_DUPLICATE_GROUP_SIZE
from faceorganizer.database.core import (
    clear_duplicate_groups,
    insert_duplicate_group,
    update_photo_duplicate_groups_batch,
)
from faceorganizer.logging_config import get_logger

log = get_logger("duplicates")


def run_duplicate_detection(
    conn: sqlite3.Connection,
    hamming_threshold: int = DEFAULT_DUPLICATE_HAMMING_THRESHOLD,
    min_group_size: int = MIN_DUPLICATE_GROUP_SIZE,
) -> int:
    """Group photos into duplicate groups by perceptual-hash similarity.

    Clears any existing duplicate-group assignments before running (mirrors
    run_clustering). *hamming_threshold* is out of 64 bits (imagehash.phash's
    hash size); a photo pair within that Hamming distance is considered a
    likely duplicate. Returns the number of groups found.
    """
    rows = conn.execute("SELECT id, phash FROM photos WHERE phash IS NOT NULL").fetchall()

    clear_duplicate_groups(conn)

    if len(rows) < min_group_size:
        log.info("Not enough hashed photos to detect duplicates (%d found)", len(rows))
        return 0

    photo_ids = [r[0] for r in rows]
    # Unpack each 64-bit hex hash into a 64-element boolean vector.  sklearn's
    # BallTree supports 'hamming' natively, so this gets the same O(N log N)
    # scaling that run_clustering() uses for face embeddings (vs. O(N^2)
    # brute-force), which matters for large photo libraries.
    bits = np.array([imagehash.hex_to_hash(phash).hash.flatten() for _, phash in rows])
    hash_bits = bits.shape[1]
    eps = hamming_threshold / hash_bits

    log.info(
        "Running DBSCAN (hamming threshold=%d/%d bits -> eps=%.4f, min_samples=%d) on %d photos",
        hamming_threshold, hash_bits, eps, min_group_size, len(photo_ids),
    )
    labels = DBSCAN(
        eps=eps, min_samples=min_group_size, metric="hamming", algorithm="ball_tree"
    ).fit_predict(bits)

    unique_labels = set(labels)
    unique_labels.discard(-1)  # -1 means noise / no duplicate found
    num_groups = len(unique_labels)

    if num_groups == 0:
        log.info("No duplicate groups found among %d hashed photos", len(photo_ids))
        return 0

    label_to_group_id: dict[int, int] = {
        label: insert_duplicate_group(conn) for label in sorted(unique_labels)
    }

    assignments: list[tuple[int | None, int]] = [
        (label_to_group_id.get(int(label)), photo_id)
        for photo_id, label in zip(photo_ids, labels, strict=True)
    ]
    update_photo_duplicate_groups_batch(conn, assignments)

    log.info("Found %d duplicate group(s) among %d hashed photos", num_groups, len(photo_ids))
    return num_groups
