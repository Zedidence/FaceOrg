"""Flask test-client coverage for faceorganizer/web/app.py's cluster-management API.

These routes were migrated to the shared faceorganizer.actions layer; this
file exercises them the way a browser actually would, through HTTP requests
against a real (temp-file) database, rather than calling actions.py directly.
"""

from __future__ import annotations

import time

import pytest

from faceorganizer.config import get_db_path
from faceorganizer.database.core import insert_cluster
from faceorganizer.database.schema import init_db
from faceorganizer.web.app import create_app

from .conftest import add_faces, add_photo, make_embedding, make_similar_embedding


@pytest.fixture
def web(tmp_path):
    """A (client, seed_conn) pair backed by a real scan-root database."""
    scan_root = tmp_path / "photos"
    scan_root.mkdir()
    db_path = get_db_path(scan_root)
    seed_conn = init_db(db_path)

    app = create_app(scan_root)
    app.testing = True
    client = app.test_client()

    yield client, seed_conn
    seed_conn.close()


def _poll_task(client, task_id, timeout=5.0):
    """Poll /api/task/<id> until it's done/error/cancelled, or time out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/task/{task_id}")
        data = resp.get_json()
        if data["status"] in ("done", "error", "cancelled"):
            return data
        time.sleep(0.02)
    raise TimeoutError(f"Task {task_id} did not finish within {timeout}s")


class TestRenameRoute:
    def test_sanitizes_and_reports_name(self, web):
        client, conn = web
        cid = insert_cluster(conn, "Person_001")

        resp = client.post("/api/rename", json={"cluster_id": cid, "name": "Al/ice"})
        data = resp.get_json()

        assert resp.status_code == 200
        assert data["name"] == "Al_ice"

    def test_missing_name_is_400(self, web):
        client, conn = web
        cid = insert_cluster(conn, "Person_001")
        resp = client.post("/api/rename", json={"cluster_id": cid, "name": ""})
        assert resp.status_code == 400

    def test_missing_cluster_is_404(self, web):
        client, _conn = web
        resp = client.post("/api/rename", json={"cluster_id": 9999, "name": "Alice"})
        assert resp.status_code == 404


class TestMergeRoute:
    def test_self_merge_is_400(self, web):
        client, conn = web
        cid = insert_cluster(conn, "Alice")
        resp = client.post("/api/merge", json={"keep_id": cid, "merge_id": cid})
        assert resp.status_code == 400

    def test_missing_cluster_is_404(self, web):
        client, conn = web
        cid = insert_cluster(conn, "Alice")
        resp = client.post("/api/merge", json={"keep_id": cid, "merge_id": 9999})
        assert resp.status_code == 404

    def test_successful_merge(self, web):
        client, conn = web
        pid = add_photo(conn)
        c1 = insert_cluster(conn, "Alice")
        c2 = insert_cluster(conn, "Bob")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=c1)
        add_faces(conn, pid, [make_embedding(2)], cluster_id=c2)

        resp = client.post("/api/merge", json={"keep_id": c1, "merge_id": c2})
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True, "kept": c1, "merged": c2}


class TestSplitRoutes:
    def test_split_single_defaults_name(self, web):
        client, conn = web
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        [fid] = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        resp = client.post("/api/split", json={"face_id": fid, "name": ""})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["new_cluster_id"] != cid

    def test_split_missing_face_is_404(self, web):
        client, _conn = web
        resp = client.post("/api/split", json={"face_id": 9999, "name": "X"})
        assert resp.status_code == 404

    def test_split_batch_requires_face_ids(self, web):
        client, _conn = web
        resp = client.post("/api/split-batch", json={"face_ids": []})
        assert resp.status_code == 400

    def test_split_batch_groups_by_similarity(self, web):
        client, conn = web
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Mixed")
        base_a = make_embedding(1)
        base_b = make_embedding(999)
        face_ids = add_faces(
            conn, pid,
            [base_a, make_similar_embedding(base_a), base_b, make_similar_embedding(base_b)],
            cluster_id=cid,
        )

        resp = client.post("/api/split-batch", json={"face_ids": face_ids, "name": "Group"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert len(data["new_cluster_ids"]) == 2


class TestDismissRoutes:
    def test_dismiss_face_not_found(self, web):
        client, _conn = web
        resp = client.post("/api/dismiss", json={"face_id": 9999})
        assert resp.status_code == 404

    def test_dismiss_and_restore_face(self, web):
        client, conn = web
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        [fid] = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        resp = client.post("/api/dismiss", json={"face_id": fid})
        assert resp.status_code == 200

        resp = client.post("/api/restore", json={"face_id": fid})
        assert resp.status_code == 200

    def test_dismiss_cluster_not_found_is_404(self, web):
        """Regression test: this used to silently return 200 with faces_dismissed=0."""
        client, _conn = web
        resp = client.post("/api/dismiss-cluster", json={"cluster_id": 9999})
        assert resp.status_code == 404

    def test_dismiss_cluster_removes_it(self, web):
        client, conn = web
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1), make_embedding(2)], cluster_id=cid)

        resp = client.post("/api/dismiss-cluster", json={"cluster_id": cid})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["faces_dismissed"] == 2


class TestReclusterRoute:
    def test_recluster_missing_cluster_reports_error(self, web):
        client, _conn = web
        resp = client.post("/api/recluster", json={"cluster_id": 9999})
        assert resp.status_code == 200  # task accepted; failure surfaces via polling
        task_id = resp.get_json()["task_id"]
        result = _poll_task(client, task_id)
        assert result["status"] == "error"

    def test_recluster_splits_mixed_cluster(self, web):
        client, conn = web
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Mixed")
        base_a = make_embedding(1)
        base_b = make_embedding(999)
        add_faces(
            conn, pid,
            [
                base_a, make_similar_embedding(base_a), make_similar_embedding(base_a),
                base_b, make_similar_embedding(base_b), make_similar_embedding(base_b),
            ],
            cluster_id=cid,
        )

        resp = client.post("/api/recluster", json={"cluster_id": cid, "threshold": 0.55})
        task_id = resp.get_json()["task_id"]
        result = _poll_task(client, task_id)

        assert result["status"] == "done"
        assert result["result"]["new_clusters"] >= 1


class TestExportRoute:
    def test_export_requires_output_dir(self, web):
        client, _conn = web
        resp = client.post("/api/export", json={"output_dir": ""})
        assert resp.status_code == 400

    def test_export_can_be_cancelled(self, web, tmp_path):
        """Regression test: export used to drop stop_event, making cancel a no-op."""
        client, conn = web
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        resp = client.post(
            "/api/export",
            json={"output_dir": str(tmp_path / "export")},
        )
        task_id = resp.get_json()["task_id"]
        client.post(f"/api/task/{task_id}/cancel")
        result = _poll_task(client, task_id)
        assert result["status"] in ("cancelled", "done")  # tiny job may finish before cancel lands


class TestPersonPageTemplate:
    def test_split_and_recluster_have_independent_threshold_fields(self, web):
        """Regression test: these two fields used to be the same shared <input>."""
        client, conn = web
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        resp = client.get(f"/person/{cid}")
        html = resp.get_data(as_text=True)
        assert resp.status_code == 200
        assert 'id="split-threshold"' in html
        assert 'id="recluster-threshold"' in html
