"""Tests for the shared cluster-management actions layer (faceorganizer/actions.py).

These actions are the single place the CLI, web app, and desktop UI all call
into for rename/merge/split/dismiss/recluster/export, so this file exercises
the validation and side effects that must hold no matter which interface
triggers them.
"""

from __future__ import annotations

import pytest

from faceorganizer import actions
from faceorganizer.database.core import (
    get_cluster_by_id,
    get_clusters,
    get_faces_for_cluster,
    insert_cluster,
)

from .conftest import add_faces, add_photo, make_embedding, make_similar_embedding


class TestRename:
    def test_sanitizes_unsafe_characters(self, conn):
        cid = insert_cluster(conn, "Person_001")
        result = actions.rename_person_full(conn, cid, "Al/ice<>")
        assert result["name"] == "Al_ice__"
        assert get_cluster_by_id(conn, cid).name == "Al_ice__"

    def test_absorbs_unassigned_faces_near_centroid(self, conn):
        pid = add_photo(conn)
        base = make_embedding(1)
        cid = insert_cluster(conn, "Person_001")
        add_faces(conn, pid, [base, make_similar_embedding(base)], cluster_id=cid)
        # An unassigned face close to the cluster centroid
        add_faces(conn, pid, [make_similar_embedding(base)], cluster_id=None)

        result = actions.rename_person_full(conn, cid, "Alice")
        assert result["absorbed_faces"] == 1
        assert len(get_faces_for_cluster(conn, cid)) == 3

    def test_does_not_absorb_for_auto_generated_name(self, conn):
        pid = add_photo(conn)
        base = make_embedding(1)
        cid = insert_cluster(conn, "OldName")
        add_faces(conn, pid, [base], cluster_id=cid)
        add_faces(conn, pid, [make_similar_embedding(base)], cluster_id=None)

        result = actions.rename_person_full(conn, cid, "Person_042")
        # Auto-generated/"maybe X" names never trigger absorption, so the
        # absorb-related keys aren't present in the result at all.
        assert "absorbed_faces" not in result
        assert len(get_faces_for_cluster(conn, cid)) == 1

    def test_returns_none_for_missing_cluster(self, conn):
        assert actions.rename_person_full(conn, 9999, "Alice") is None
        assert actions.rename_person(conn, 9999, "Alice") is False

    def test_rename_person_bool_wrapper(self, conn):
        cid = insert_cluster(conn, "Person_001")
        assert actions.rename_person(conn, cid, "Alice") is True


class TestMerge:
    def test_self_merge_raises(self, conn):
        cid = insert_cluster(conn, "Alice")
        with pytest.raises(actions.ActionError) as exc:
            actions.merge_people(conn, cid, cid)
        assert exc.value.status == 400

    def test_missing_cluster_raises_404(self, conn):
        cid = insert_cluster(conn, "Alice")
        with pytest.raises(actions.ActionError) as exc:
            actions.merge_people(conn, cid, 9999)
        assert exc.value.status == 404

    def test_merges_faces_and_deletes_source(self, conn):
        pid = add_photo(conn)
        c1 = insert_cluster(conn, "Alice")
        c2 = insert_cluster(conn, "Bob")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=c1)
        add_faces(conn, pid, [make_embedding(2)], cluster_id=c2)

        actions.merge_people(conn, c1, c2)

        assert get_cluster_by_id(conn, c2) is None
        assert len(get_faces_for_cluster(conn, c1)) == 2


class TestSplit:
    def test_missing_face_raises_404(self, conn):
        with pytest.raises(actions.ActionError) as exc:
            actions.split_face(conn, 9999, "New Person")
        assert exc.value.status == 404

    def test_defaults_name_when_blank(self, conn):
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        [fid] = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        result = actions.split_face(conn, fid, "")
        assert result["name"] == f"Split_{fid}"

    def test_sanitizes_name(self, conn):
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        [fid] = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        result = actions.split_face(conn, fid, "Weird/Name")
        assert result["name"] == "Weird_Name"
        assert get_faces_for_cluster(conn, result["cluster_id"])[0]["face_id"] == fid

    def test_batch_requires_face_ids(self, conn):
        with pytest.raises(actions.ActionError):
            actions.split_faces_batch(conn, [], "Split")

    def test_batch_defaults_base_name(self, conn):
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        face_ids = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        result = actions.split_faces_batch(conn, face_ids, "")
        assert len(result["cluster_ids"]) == 1
        assert get_cluster_by_id(conn, result["cluster_ids"][0]).name == "Split"


class TestDismissRestore:
    def test_dismiss_face_not_found(self, conn):
        with pytest.raises(actions.ActionError) as exc:
            actions.dismiss_face(conn, 9999)
        assert exc.value.status == 404

    def test_restore_face_not_found(self, conn):
        with pytest.raises(actions.ActionError):
            actions.restore_face(conn, 9999)

    def test_dismiss_then_restore(self, conn):
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        [fid] = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        actions.dismiss_face(conn, fid)
        assert get_faces_for_cluster(conn, cid) == []

        actions.restore_face(conn, fid)
        # Restored faces are unassigned, not automatically re-clustered
        assert get_faces_for_cluster(conn, cid) == []

    def test_dismiss_cluster_not_found(self, conn):
        with pytest.raises(actions.ActionError) as exc:
            actions.dismiss_cluster(conn, 9999)
        assert exc.value.status == 404

    def test_dismiss_cluster_removes_it_and_counts_faces(self, conn):
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1), make_embedding(2)], cluster_id=cid)

        count = actions.dismiss_cluster(conn, cid)
        assert count == 2
        assert get_cluster_by_id(conn, cid) is None


class TestRecluster:
    def test_missing_cluster_raises_404(self, conn):
        with pytest.raises(actions.ActionError) as exc:
            actions.recluster_person(conn, 9999)
        assert exc.value.status == 404

    def test_cohesive_cluster_is_a_noop(self, conn):
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        base = make_embedding(1)
        add_faces(
            conn, pid,
            [base, make_similar_embedding(base), make_similar_embedding(base)],
            cluster_id=cid,
        )
        result = actions.recluster_person(conn, cid, eps=0.55)
        assert result == {"original_faces": 3, "new_clusters": 0, "noise": 0}

    def test_splits_mixed_cluster(self, conn):
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
        result = actions.recluster_person(conn, cid, eps=0.55)
        assert result["new_clusters"] >= 1
        assert len(get_clusters(conn)) == 2


class TestExport:
    def test_export_copies_photos_into_person_folders(self, conn, tmp_path):
        src = tmp_path / "src.jpg"
        src.write_bytes(b"fake image data")
        pid = add_photo(conn, path=str(src))
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        out_dir = tmp_path / "export"
        summary = actions.export_people(conn, out_dir, symlink=False)

        assert summary == {"Alice": 1}
        assert (out_dir / "Alice" / "src.jpg").read_bytes() == b"fake image data"
