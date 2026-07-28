"""Tests for faceorganizer/duplicates.py and the duplicate-group DB helpers."""

from __future__ import annotations

from faceorganizer.database.core import (
    clear_duplicate_groups,
    delete_photo,
    get_duplicate_groups,
    get_photo_by_id,
    get_photos_in_duplicate_group,
    insert_photo,
)
from faceorganizer.duplicates import run_duplicate_detection
from faceorganizer.models import PhotoInfo

# A base hash and near/far variants, used across multiple tests.
_BASE_HASH = "8f8f8f8f8f8f8f8f"
_NEAR_HASH = "8f8f8f8f8f8f8f8d"  # 1 bit different from _BASE_HASH
_FAR_HASH = "0000000000000000"  # very different from _BASE_HASH


def _add_photo(conn, path, phash=None, file_size=100):
    return insert_photo(
        conn,
        PhotoInfo(path=path, file_size=file_size, width=200, height=200,
                  format="JPEG", phash=phash),
    )


class TestRunDuplicateDetection:
    def test_groups_near_duplicates(self, conn):
        id_a = _add_photo(conn, "/fake/a1.jpg", _BASE_HASH)
        id_b = _add_photo(conn, "/fake/a2.jpg", _NEAR_HASH)

        num_groups = run_duplicate_detection(conn, hamming_threshold=10, min_group_size=2)

        assert num_groups == 1
        groups = get_duplicate_groups(conn)
        assert len(groups) == 1
        assert groups[0]["photo_count"] == 2
        photo_ids = {p["photo_id"] for p in get_photos_in_duplicate_group(conn, groups[0]["id"])}
        assert photo_ids == {id_a, id_b}

    def test_leaves_dissimilar_photos_ungrouped(self, conn):
        _add_photo(conn, "/fake/a1.jpg", _BASE_HASH)
        id_c = _add_photo(conn, "/fake/b1.jpg", _FAR_HASH)

        run_duplicate_detection(conn, hamming_threshold=10, min_group_size=2)

        assert get_photo_by_id(conn, id_c)["duplicate_group_id"] is None

    def test_ignores_photos_without_a_hash(self, conn):
        _add_photo(conn, "/fake/a1.jpg", _BASE_HASH)
        _add_photo(conn, "/fake/a2.jpg", _NEAR_HASH)
        id_no_hash = _add_photo(conn, "/fake/none.jpg", None)

        run_duplicate_detection(conn, hamming_threshold=10, min_group_size=2)

        assert get_photo_by_id(conn, id_no_hash)["duplicate_group_id"] is None

    def test_respects_hamming_threshold(self, conn):
        # These two hashes are exactly 4 bits apart.
        _add_photo(conn, "/fake/a1.jpg", "8f8f8f8f8f8f8f8f")
        _add_photo(conn, "/fake/a2.jpg", "8f8f8f8f8f8f8f80")

        # A stricter threshold than the actual 4-bit distance excludes the pair.
        num_groups = run_duplicate_detection(conn, hamming_threshold=2, min_group_size=2)

        assert num_groups == 0

    def test_no_photos_returns_zero(self, conn):
        assert run_duplicate_detection(conn) == 0

    def test_fewer_than_min_group_size_returns_zero(self, conn):
        _add_photo(conn, "/fake/a1.jpg", _BASE_HASH)
        assert run_duplicate_detection(conn, min_group_size=2) == 0

    def test_rerunning_clears_previous_groups(self, conn):
        id_a = _add_photo(conn, "/fake/a1.jpg", _BASE_HASH)
        id_b = _add_photo(conn, "/fake/a2.jpg", _NEAR_HASH)
        run_duplicate_detection(conn, hamming_threshold=10, min_group_size=2)
        assert get_photo_by_id(conn, id_a)["duplicate_group_id"] is not None

        # Delete one photo from the pair so it's now a singleton; re-run.
        delete_photo(conn, id_b)
        run_duplicate_detection(conn, hamming_threshold=10, min_group_size=2)

        assert get_photo_by_id(conn, id_a)["duplicate_group_id"] is None
        assert get_duplicate_groups(conn) == []


class TestDuplicateGroupHelpers:
    def test_clear_duplicate_groups(self, conn):
        _add_photo(conn, "/fake/a1.jpg", _BASE_HASH)
        _add_photo(conn, "/fake/a2.jpg", _NEAR_HASH)
        run_duplicate_detection(conn, hamming_threshold=10, min_group_size=2)
        assert get_duplicate_groups(conn) != []

        clear_duplicate_groups(conn)

        assert get_duplicate_groups(conn) == []

    def test_photos_in_group_ordered_largest_file_first(self, conn):
        id_small = _add_photo(conn, "/fake/small.jpg", _BASE_HASH, file_size=50)
        id_large = _add_photo(conn, "/fake/large.jpg", _NEAR_HASH, file_size=500)
        run_duplicate_detection(conn, hamming_threshold=10, min_group_size=2)

        group_id = get_photo_by_id(conn, id_small)["duplicate_group_id"]
        photos = get_photos_in_duplicate_group(conn, group_id)

        assert [p["photo_id"] for p in photos] == [id_large, id_small]


class TestDeletePhoto:
    def test_delete_nonexistent_returns_false(self, conn):
        assert delete_photo(conn, 9999) is False

    def test_delete_removes_photo_and_its_faces(self, conn):
        from faceorganizer.database.core import get_faces_for_cluster, insert_cluster
        from tests.conftest import add_faces, make_embedding

        photo_id = _add_photo(conn, "/fake/a.jpg")
        cid = insert_cluster(conn, "Alice")
        add_faces(conn, photo_id, [make_embedding(1)], cluster_id=cid)

        assert delete_photo(conn, photo_id) is True

        assert get_photo_by_id(conn, photo_id) is None
        assert get_faces_for_cluster(conn, cid) == []
