"""Tests for face dismissal (marking false-positive detections)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from faceorganizer.clustering.cluster import run_clustering, run_incremental_clustering
from faceorganizer.database.core import (
    dismiss_face,
    get_all_embeddings,
    get_scan_stats,
    get_unassigned_embeddings,
    insert_cluster,
    insert_faces_batch,
    insert_photo,
    restore_face,
    update_face_clusters_batch,
)
from faceorganizer.database.schema import init_db
from faceorganizer.models import FaceInfo, PhotoInfo


def _emb(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _similar(base: np.ndarray) -> np.ndarray:
    rng = np.random.RandomState(abs(hash(base.tobytes())) % (2**31))
    v = base + rng.randn(512).astype(np.float32) * 0.02
    return v / np.linalg.norm(v)


def _db(tmp_path) -> sqlite3.Connection:
    return init_db(tmp_path / "test.db")


def _photo(conn, name="p.jpg") -> int:
    return insert_photo(conn, PhotoInfo(
        path=f"/fake/{name}", file_size=1, width=1, height=1,
        format="JPEG", exif_date=None, num_faces=0,
    ))


def _faces(conn, photo_id, embeddings):
    insert_faces_batch(conn, [
        FaceInfo(photo_path="/fake/p.jpg", bbox_x=0, bbox_y=0, bbox_w=1, bbox_h=1,
                 embedding=e, detection_confidence=0.9)
        for e in embeddings
    ], photo_id)


class TestDismissRestore:
    def test_dismiss_removes_from_cluster(self, tmp_path):
        conn = _db(tmp_path)
        pid = _photo(conn)
        base = _emb(1)
        _faces(conn, pid, [base, _similar(base)])
        run_clustering(conn, eps=0.55)

        # Face 1 should be in a cluster
        stats = get_scan_stats(conn)
        assert stats["faces"] == 2
        assert stats["dismissed_faces"] == 0

        dismiss_face(conn, 1)
        stats = get_scan_stats(conn)
        assert stats["faces"] == 1  # dismissed not counted
        assert stats["dismissed_faces"] == 1
        conn.close()

    def test_restore_undoes_dismiss(self, tmp_path):
        conn = _db(tmp_path)
        pid = _photo(conn)
        _faces(conn, pid, [_emb(1)])
        dismiss_face(conn, 1)
        assert get_scan_stats(conn)["dismissed_faces"] == 1

        restore_face(conn, 1)
        assert get_scan_stats(conn)["dismissed_faces"] == 0
        assert get_scan_stats(conn)["faces"] == 1
        conn.close()

    def test_dismissed_excluded_from_get_all_embeddings(self, tmp_path):
        conn = _db(tmp_path)
        pid = _photo(conn)
        _faces(conn, pid, [_emb(1), _emb(2)])
        dismiss_face(conn, 1)

        ids, _ = get_all_embeddings(conn)
        assert 1 not in ids
        assert 2 in ids
        conn.close()

    def test_dismissed_excluded_from_unassigned_embeddings(self, tmp_path):
        conn = _db(tmp_path)
        pid = _photo(conn)
        _faces(conn, pid, [_emb(1), _emb(2)])
        dismiss_face(conn, 1)

        ids, _ = get_unassigned_embeddings(conn)
        assert 1 not in ids
        assert 2 in ids
        conn.close()

    def test_dismissed_excluded_from_clustering(self, tmp_path):
        conn = _db(tmp_path)
        pid = _photo(conn)
        base = _emb(10)
        _faces(conn, pid, [base, _similar(base), _emb(999)])
        dismiss_face(conn, 3)  # dismiss the outlier

        run_clustering(conn, eps=0.55)
        stats = get_scan_stats(conn)
        # Only the 2 non-dismissed faces should be considered
        assert stats["dismissed_faces"] == 1
        conn.close()

    def test_dismiss_nonexistent_returns_false(self, tmp_path):
        conn = _db(tmp_path)
        assert dismiss_face(conn, 9999) is False
        conn.close()
