"""Tests for faceorganizer/cli/commands.py.

Calls main(argv) in-process (fast, no subprocess) against a real temp-file
database seeded directly via the database helpers, since `scan` itself needs
real images and downloaded ONNX models and is out of scope here.
"""

from __future__ import annotations

import pytest

from faceorganizer import __version__
from faceorganizer.cli.commands import main
from faceorganizer.config import get_db_path
from faceorganizer.database.core import insert_cluster, insert_photo
from faceorganizer.database.schema import init_db
from faceorganizer.models import PhotoInfo

from .conftest import add_faces, add_photo, make_embedding, make_similar_embedding


@pytest.fixture
def scanned_folder(tmp_path):
    """A folder with an initialized (but otherwise empty) database, plus its conn."""
    folder = tmp_path / "photos"
    folder.mkdir()
    conn = init_db(get_db_path(folder))
    yield folder, conn
    conn.close()


class TestVersionAndStats:
    def test_version_prints_and_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_stats_without_scan_exits_nonzero(self, tmp_path, capsys):
        folder = tmp_path / "never_scanned"
        folder.mkdir()
        with pytest.raises(SystemExit) as exc:
            main(["stats", str(folder)])
        assert exc.value.code == 1

    def test_stats_on_seeded_db(self, scanned_folder, capsys):
        folder, conn = scanned_folder
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        main(["stats", str(folder)])
        out = capsys.readouterr().out
        assert "Faces:" in out
        assert "1" in out


class TestRename:
    def test_rename_success(self, scanned_folder, capsys):
        folder, conn = scanned_folder
        cid = insert_cluster(conn, "Person_001")

        main(["rename", str(folder), str(cid), "Alice"])
        assert "renamed to 'Alice'" in capsys.readouterr().out

    def test_rename_missing_cluster_exits_nonzero(self, scanned_folder):
        folder, _conn = scanned_folder
        with pytest.raises(SystemExit) as exc:
            main(["rename", str(folder), "9999", "Alice"])
        assert exc.value.code == 1


class TestDismiss:
    def test_dismiss_and_restore(self, scanned_folder, capsys):
        folder, conn = scanned_folder
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        [fid] = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        main(["dismiss", str(folder), str(fid)])
        assert "dismissed" in capsys.readouterr().out

        main(["dismiss", str(folder), str(fid), "--restore"])
        assert "restored" in capsys.readouterr().out

    def test_dismiss_missing_face_exits_nonzero(self, scanned_folder):
        folder, _conn = scanned_folder
        with pytest.raises(SystemExit) as exc:
            main(["dismiss", str(folder), "9999"])
        assert exc.value.code == 1

    def test_dismiss_cluster_success(self, scanned_folder, capsys):
        folder, conn = scanned_folder
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1), make_embedding(2)], cluster_id=cid)

        main(["dismiss-cluster", str(folder), str(cid)])
        assert "2 face(s)" in capsys.readouterr().out

    def test_dismiss_cluster_missing_exits_nonzero(self, scanned_folder):
        folder, _conn = scanned_folder
        with pytest.raises(SystemExit) as exc:
            main(["dismiss-cluster", str(folder), "9999"])
        assert exc.value.code == 1


class TestMerge:
    def test_merge_success(self, scanned_folder, capsys):
        folder, conn = scanned_folder
        pid = add_photo(conn)
        c1 = insert_cluster(conn, "Alice")
        c2 = insert_cluster(conn, "Bob")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=c1)
        add_faces(conn, pid, [make_embedding(2)], cluster_id=c2)

        main(["merge", str(folder), str(c1), str(c2)])
        assert "merged into" in capsys.readouterr().out

    def test_self_merge_exits_nonzero(self, scanned_folder):
        folder, conn = scanned_folder
        cid = insert_cluster(conn, "Alice")
        with pytest.raises(SystemExit) as exc:
            main(["merge", str(folder), str(cid), str(cid)])
        assert exc.value.code == 1

    def test_merge_missing_cluster_exits_nonzero(self, scanned_folder):
        folder, conn = scanned_folder
        cid = insert_cluster(conn, "Alice")
        with pytest.raises(SystemExit) as exc:
            main(["merge", str(folder), str(cid), "9999"])
        assert exc.value.code == 1


class TestSplit:
    def test_split_success(self, scanned_folder, capsys):
        folder, conn = scanned_folder
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        [fid] = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        main(["split", str(folder), str(fid), "New Person"])
        assert "New Person" in capsys.readouterr().out

    def test_split_default_name(self, scanned_folder, capsys):
        folder, conn = scanned_folder
        pid = add_photo(conn)
        cid = insert_cluster(conn, "Alice")
        [fid] = add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        main(["split", str(folder), str(fid)])
        assert f"Split_{fid}" in capsys.readouterr().out

    def test_split_missing_face_exits_nonzero(self, scanned_folder):
        folder, _conn = scanned_folder
        with pytest.raises(SystemExit) as exc:
            main(["split", str(folder), "9999"])
        assert exc.value.code == 1


class TestRecluster:
    def test_recluster_missing_cluster_exits_nonzero(self, scanned_folder):
        folder, _conn = scanned_folder
        with pytest.raises(SystemExit) as exc:
            main(["recluster", str(folder), "9999"])
        assert exc.value.code == 1

    def test_recluster_splits_mixed_cluster(self, scanned_folder, capsys):
        folder, conn = scanned_folder
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

        main(["recluster", str(folder), str(cid), "--threshold", "0.55"])
        assert "Recluster complete" in capsys.readouterr().out


class TestFindDuplicatesAndDuplicates:
    def test_no_hashed_photos_reports_none_found(self, scanned_folder, capsys):
        folder, _conn = scanned_folder
        main(["find-duplicates", str(folder)])
        assert "No duplicate groups found" in capsys.readouterr().out

    def test_finds_and_lists_duplicate_groups(self, scanned_folder, capsys, tmp_path):
        folder, conn = scanned_folder
        insert_photo(conn, PhotoInfo(
            path=str(tmp_path / "a1.jpg"), file_size=100, width=200, height=200,
            format="JPEG", phash="8f8f8f8f8f8f8f8f",
        ))
        insert_photo(conn, PhotoInfo(
            path=str(tmp_path / "a2.jpg"), file_size=100, width=200, height=200,
            format="JPEG", phash="8f8f8f8f8f8f8f8d",
        ))

        main(["find-duplicates", str(folder)])
        assert "Found 1 duplicate group(s)" in capsys.readouterr().out

        main(["duplicates", str(folder)])
        out = capsys.readouterr().out
        assert "1 duplicate group(s)" in out
        assert "a1.jpg" in out
        assert "a2.jpg" in out

    def test_duplicates_without_prior_detection(self, scanned_folder, capsys):
        folder, _conn = scanned_folder
        main(["duplicates", str(folder)])
        assert "Run 'find-duplicates' first" in capsys.readouterr().out


class TestDeletePhoto:
    def test_delete_photo_success(self, scanned_folder, capsys, tmp_path):
        folder, conn = scanned_folder
        src = tmp_path / "photo.jpg"
        src.write_bytes(b"data")
        photo_id = insert_photo(conn, PhotoInfo(
            path=str(src), file_size=4, width=10, height=10, format="JPEG",
        ))

        main(["delete-photo", str(folder), str(photo_id)])

        assert "sent to the Recycle Bin" in capsys.readouterr().out
        assert not src.exists()

    def test_delete_photo_missing_exits_nonzero(self, scanned_folder):
        folder, _conn = scanned_folder
        with pytest.raises(SystemExit) as exc:
            main(["delete-photo", str(folder), "9999"])
        assert exc.value.code == 1
