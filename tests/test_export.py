"""Tests for organizer/export.py — copying/symlinking photos into per-person folders."""

from __future__ import annotations

import threading

import pytest

from faceorganizer.database.core import insert_cluster
from faceorganizer.organizer.export import export_by_person

from .conftest import add_faces, add_photo, make_embedding


def _symlinks_supported(tmp_path) -> bool:
    """Windows requires Developer Mode or admin rights to create symlinks."""
    probe_target = tmp_path / "_probe_target"
    probe_link = tmp_path / "_probe_link"
    probe_target.write_bytes(b"")
    try:
        probe_link.symlink_to(probe_target)
    except OSError:
        return False
    return True


class TestExportByPerson:
    def test_no_clusters_returns_empty_summary(self, conn, tmp_path):
        out_dir = tmp_path / "export"
        summary = export_by_person(conn, out_dir)
        assert summary == {}

    def test_copies_photo_into_sanitized_person_folder(self, conn, tmp_path):
        src = tmp_path / "vacation.jpg"
        src.write_bytes(b"photo bytes")
        pid = add_photo(conn, path=str(src))
        cid = insert_cluster(conn, "Al/ice")  # unsafe folder-name character
        add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        out_dir = tmp_path / "export"
        summary = export_by_person(conn, out_dir, symlink=False)

        assert summary == {"Al/ice": 1}
        exported = out_dir / "Al_ice" / "vacation.jpg"
        assert exported.exists()
        assert exported.read_bytes() == b"photo bytes"
        # Original file is untouched (copy, not move)
        assert src.read_bytes() == b"photo bytes"

    def test_symlink_mode_creates_symlink_not_copy(self, conn, tmp_path):
        if not _symlinks_supported(tmp_path):
            pytest.skip("Symlink creation not permitted (needs Developer Mode or admin on Windows)")
        src = tmp_path / "photo.jpg"
        src.write_bytes(b"data")
        pid = add_photo(conn, path=str(src))
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        out_dir = tmp_path / "export"
        export_by_person(conn, out_dir, symlink=True)

        exported = out_dir / "Alice" / "photo.jpg"
        assert exported.is_symlink()

    def test_duplicate_filenames_get_suffixed(self, conn, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        src_a = dir_a / "photo.jpg"
        src_b = dir_b / "photo.jpg"
        src_a.write_bytes(b"AAA")
        src_b.write_bytes(b"BBB")

        cid = insert_cluster(conn, "Alice")
        pid_a = add_photo(conn, path=str(src_a))
        add_faces(conn, pid_a, [make_embedding(1)], cluster_id=cid)
        pid_b = add_photo(conn, path=str(src_b))
        add_faces(conn, pid_b, [make_embedding(2)], cluster_id=cid)

        out_dir = tmp_path / "export"
        summary = export_by_person(conn, out_dir)

        assert summary == {"Alice": 2}
        person_dir = out_dir / "Alice"
        names = sorted(p.name for p in person_dir.iterdir())
        assert names == ["photo.jpg", "photo_1.jpg"]

    def test_missing_source_file_is_skipped_not_crashed(self, conn, tmp_path):
        missing = tmp_path / "gone.jpg"  # never created
        pid = add_photo(conn, path=str(missing))
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, pid, [make_embedding(1)], cluster_id=cid)

        out_dir = tmp_path / "export"
        summary = export_by_person(conn, out_dir)

        assert summary == {"Alice": 0}

    def test_on_progress_called_once_per_cluster(self, conn, tmp_path):
        calls = []
        for i, seed in enumerate((1, 2)):
            src = tmp_path / f"p{i}.jpg"
            src.write_bytes(b"x")
            pid = add_photo(conn, path=str(src))
            cid = insert_cluster(conn, f"Person{i}")
            add_faces(conn, pid, [make_embedding(seed)], cluster_id=cid)

        export_by_person(conn, tmp_path / "export", on_progress=lambda done, total: calls.append((done, total)))

        assert calls == [(1, 2), (2, 2)]

    def test_stop_event_halts_early(self, conn, tmp_path):
        for i, seed in enumerate((1, 2, 3)):
            src = tmp_path / f"p{i}.jpg"
            src.write_bytes(b"x")
            pid = add_photo(conn, path=str(src))
            cid = insert_cluster(conn, f"Person{i}")
            add_faces(conn, pid, [make_embedding(seed)], cluster_id=cid)

        stop_event = threading.Event()
        stop_event.set()  # already stopped before we even start
        summary = export_by_person(conn, tmp_path / "export", stop_event=stop_event)

        assert summary == {}
