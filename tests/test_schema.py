"""Tests for faceorganizer/database/schema.py's migration system."""

from __future__ import annotations

import sqlite3

from faceorganizer.database.schema import SCHEMA_VERSION, configure_connection, init_db


class TestFreshDatabase:
    def test_photos_table_has_duplicate_detection_columns(self, tmp_path):
        conn = init_db(tmp_path / "fresh.db")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(photos)").fetchall()}
        assert "phash" in cols
        assert "duplicate_group_id" in cols
        conn.close()

    def test_duplicate_groups_table_exists(self, tmp_path):
        conn = init_db(tmp_path / "fresh.db")
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='duplicate_groups'"
        ).fetchone()
        assert row is not None
        conn.close()

    def test_version_is_current(self, tmp_path):
        conn = init_db(tmp_path / "fresh.db")
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        conn.close()


class TestMigrationFromV4:
    """Regression test: a v4-shaped DB (pre-duplicate-detection) must migrate
    to v5 without crashing. This previously failed because CREATE_TABLES ran
    an index on photos.duplicate_group_id before the ALTER TABLE that adds it
    had a chance to run against an existing (pre-v5) photos table.
    """

    def _make_v4_db(self, path):
        conn = sqlite3.connect(str(path))
        configure_connection(conn)
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                format TEXT NOT NULL, exif_date TEXT, num_faces INTEGER NOT NULL DEFAULT 0,
                scanned_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT, photo_id INTEGER NOT NULL,
                bbox_x INTEGER NOT NULL, bbox_y INTEGER NOT NULL,
                bbox_w INTEGER NOT NULL, bbox_h INTEGER NOT NULL,
                embedding BLOB NOT NULL, detection_confidence REAL NOT NULL,
                cluster_id INTEGER, dismissed INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                representative_face_id INTEGER, created_at TEXT, updated_at TEXT
            );
            INSERT INTO schema_version (version) VALUES (4);
            INSERT INTO photos (path, file_size, width, height, format)
                VALUES ('/fake/a.jpg', 100, 200, 200, 'JPEG');
        """)
        conn.commit()
        conn.close()

    def test_migrates_without_error_and_preserves_data(self, tmp_path):
        db_path = tmp_path / "old.db"
        self._make_v4_db(db_path)

        conn = init_db(db_path)
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 5
        cols = {r[1] for r in conn.execute("PRAGMA table_info(photos)").fetchall()}
        assert "phash" in cols
        assert "duplicate_group_id" in cols

        row = conn.execute("SELECT path, phash, duplicate_group_id FROM photos").fetchone()
        assert row == ("/fake/a.jpg", None, None)
        conn.close()

    def test_reopening_migrated_database_is_idempotent(self, tmp_path):
        db_path = tmp_path / "old.db"
        self._make_v4_db(db_path)

        init_db(db_path).close()
        conn = init_db(db_path)  # second open must not error or re-run migrations
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 5
        conn.close()
