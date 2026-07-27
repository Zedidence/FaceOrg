"""Tests for incremental clustering functionality."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from faceorganizer.clustering.cluster import (
    run_clustering,
    run_incremental_clustering,
)
from faceorganizer.database.core import (
    get_cluster_centroids,
    get_clusters,
    get_scan_stats,
    get_unassigned_embeddings,
    insert_cluster,
    insert_faces_batch,
    insert_photo,
    merge_clusters,
    rename_cluster,
    update_face_clusters_batch,
)
from faceorganizer.database.schema import init_db
from faceorganizer.models import FaceInfo, PhotoInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(seed: int, dim: int = 512) -> np.ndarray:
    """Create a deterministic unit-length embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_similar_embedding(base: np.ndarray, noise: float = 0.02) -> np.ndarray:
    """Return an embedding close to *base* (small cosine distance)."""
    rng = np.random.RandomState(abs(hash(base.tobytes())) % (2**31))
    perturbed = base + rng.randn(*base.shape).astype(np.float32) * noise
    perturbed /= np.linalg.norm(perturbed)
    return perturbed


def _setup_db(tmp_path) -> sqlite3.Connection:
    """Initialise a fresh in-memory-like DB in tmp_path."""
    db_path = tmp_path / "test.db"
    return init_db(db_path)


def _add_photo(conn: sqlite3.Connection, name: str = "photo.jpg") -> int:
    photo = PhotoInfo(
        path=f"/fake/{name}",
        file_size=1000,
        width=640,
        height=480,
        format="JPEG",
        exif_date=None,
        num_faces=0,
    )
    return insert_photo(conn, photo)


def _add_faces(
    conn: sqlite3.Connection, photo_id: int, embeddings: list[np.ndarray],
) -> None:
    faces = [
        FaceInfo(
            photo_path="/fake/photo.jpg",
            bbox_x=0, bbox_y=0, bbox_w=100, bbox_h=100,
            embedding=emb, detection_confidence=0.99, cluster_id=None,
        )
        for emb in embeddings
    ]
    insert_faces_batch(conn, faces, photo_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetUnassignedEmbeddings:
    def test_all_unassigned(self, tmp_path):
        conn = _setup_db(tmp_path)
        pid = _add_photo(conn)
        _add_faces(conn, pid, [_make_embedding(1), _make_embedding(2)])
        ids, embs = get_unassigned_embeddings(conn)
        assert len(ids) == 2
        conn.close()

    def test_none_unassigned(self, tmp_path):
        conn = _setup_db(tmp_path)
        pid = _add_photo(conn)
        _add_faces(conn, pid, [_make_embedding(1)])
        cid = insert_cluster(conn, "Alice")
        update_face_clusters_batch(conn, [(cid, 1)])
        ids, _ = get_unassigned_embeddings(conn)
        assert len(ids) == 0
        conn.close()


class TestGetClusterCentroids:
    def test_returns_centroids(self, tmp_path):
        conn = _setup_db(tmp_path)
        pid = _add_photo(conn)
        emb = _make_embedding(10)
        _add_faces(conn, pid, [emb, _make_similar_embedding(emb)])
        cid = insert_cluster(conn, "Bob")
        update_face_clusters_batch(conn, [(cid, 1), (cid, 2)])
        centroids = get_cluster_centroids(conn)
        assert cid in centroids
        assert centroids[cid].shape == (512,)
        # Centroid should be unit-length
        assert abs(np.linalg.norm(centroids[cid]) - 1.0) < 1e-5
        conn.close()


class TestIncrementalClustering:
    def test_assigns_new_faces_to_existing_clusters(self, tmp_path):
        """New faces similar to an existing cluster get assigned to it."""
        conn = _setup_db(tmp_path)

        # Create an initial cluster "Alice" with 3 similar faces
        base = _make_embedding(42)
        pid1 = _add_photo(conn, "batch1.jpg")
        _add_faces(conn, pid1, [base, _make_similar_embedding(base), _make_similar_embedding(base)])
        run_clustering(conn, eps=0.55)

        clusters_before = get_clusters(conn)
        assert len(clusters_before) >= 1
        alice_cluster = clusters_before[0]
        rename_cluster(conn, alice_cluster.id, "Alice")

        # Add a new face similar to Alice
        pid2 = _add_photo(conn, "batch2.jpg")
        new_face = _make_similar_embedding(base, noise=0.01)
        _add_faces(conn, pid2, [new_face])

        result = run_incremental_clustering(conn, eps=0.55)
        assert result["assigned"] == 1
        assert result["new_clusters"] == 0

        # Verify Alice's name is preserved
        clusters_after = get_clusters(conn)
        names = {c.name for c in clusters_after}
        assert "Alice" in names
        conn.close()

    def test_creates_new_clusters_from_novel_faces(self, tmp_path):
        """Faces far from existing clusters form new clusters via DBSCAN."""
        conn = _setup_db(tmp_path)

        # Seed cluster (need >= 3 faces to satisfy MIN_CLUSTER_SIZE=3)
        base = _make_embedding(1)
        pid1 = _add_photo(conn, "seed.jpg")
        _add_faces(conn, pid1, [base, _make_similar_embedding(base), _make_similar_embedding(base)])
        run_clustering(conn, eps=0.55)
        rename_cluster(conn, get_clusters(conn)[0].id, "Alice")

        # Add a group of novel faces (very different from Alice)
        novel_base = _make_embedding(9999)
        pid2 = _add_photo(conn, "novel.jpg")
        _add_faces(conn, pid2, [novel_base, _make_similar_embedding(novel_base), _make_similar_embedding(novel_base)])

        result = run_incremental_clustering(conn, eps=0.55)
        assert result["new_clusters"] >= 1

        # Alice should still exist
        names = {c.name for c in get_clusters(conn)}
        assert "Alice" in names
        conn.close()

    def test_preserves_merges(self, tmp_path):
        """Merged clusters are preserved during incremental clustering."""
        conn = _setup_db(tmp_path)

        base_a = _make_embedding(100)
        base_b = _make_embedding(200)
        pid1 = _add_photo(conn, "p1.jpg")
        _add_faces(conn, pid1, [
            base_a, _make_similar_embedding(base_a), _make_similar_embedding(base_a),
            base_b, _make_similar_embedding(base_b), _make_similar_embedding(base_b),
        ])
        run_clustering(conn, eps=0.55)

        clusters = get_clusters(conn)
        if len(clusters) >= 2:
            rename_cluster(conn, clusters[0].id, "Merged")
            merge_clusters(conn, clusters[0].id, clusters[1].id)

        stats_before = get_scan_stats(conn)

        # Add a new unrelated face (will be noise)
        pid2 = _add_photo(conn, "p2.jpg")
        _add_faces(conn, pid2, [_make_embedding(7777)])

        run_incremental_clustering(conn, eps=0.55)

        # The "Merged" cluster should still exist with its name intact
        names = {c.name for c in get_clusters(conn)}
        assert "Merged" in names
        conn.close()

    def test_noop_when_no_unassigned(self, tmp_path):
        """Does nothing when all faces are already assigned."""
        conn = _setup_db(tmp_path)
        pid = _add_photo(conn)
        base = _make_embedding(5)
        _add_faces(conn, pid, [base, _make_similar_embedding(base), _make_similar_embedding(base)])
        run_clustering(conn, eps=0.55)

        result = run_incremental_clustering(conn, eps=0.55)
        assert result == {"assigned": 0, "new_clusters": 0, "maybe_clusters": 0, "new_noise": 0}
        conn.close()

    def test_noop_on_empty_db(self, tmp_path):
        conn = _setup_db(tmp_path)
        result = run_incremental_clustering(conn, eps=0.55)
        assert result == {"assigned": 0, "new_clusters": 0, "maybe_clusters": 0, "new_noise": 0}
        conn.close()
