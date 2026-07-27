"""SQLite schema definition and initialization."""

import sqlite3
from pathlib import Path

from faceorganizer.logging_config import get_logger

log = get_logger("database.schema")

SCHEMA_VERSION = 4

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS photos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    file_size   INTEGER NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    format      TEXT    NOT NULL,
    exif_date   TEXT,
    num_faces   INTEGER NOT NULL DEFAULT 0,
    scanned_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS faces (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id              INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    bbox_x                INTEGER NOT NULL,
    bbox_y                INTEGER NOT NULL,
    bbox_w                INTEGER NOT NULL,
    bbox_h                INTEGER NOT NULL,
    embedding             BLOB    NOT NULL,
    detection_confidence  REAL    NOT NULL,
    cluster_id            INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    dismissed             INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS clusters (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT    NOT NULL,
    representative_face_id INTEGER,
    created_at            TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at            TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(path);
CREATE INDEX IF NOT EXISTS idx_photos_exif_date ON photos(exif_date);
-- Composite index: speeds up the ROW_NUMBER window queries that partition by
-- cluster_id and order by detection_confidence on every dashboard/review page.
CREATE INDEX IF NOT EXISTS idx_faces_cluster_conf
    ON faces(cluster_id, detection_confidence DESC);

-- Covering index for dismissed-column filters (get_all_embeddings, stats,
-- get_dismissed_faces).  A single column index is enough because dismissed is
-- very low-cardinality and SQLite will use it for equality and IS NULL scans.
CREATE INDEX IF NOT EXISTS idx_faces_dismissed ON faces(dismissed);

-- Composite index: (cluster_id, dismissed) covers the most common join pattern
-- used by get_faces_for_cluster and get_cluster_centroids.
CREATE INDEX IF NOT EXISTS idx_faces_cluster_dismissed
    ON faces(cluster_id, dismissed);
"""


def configure_connection(conn: sqlite3.Connection) -> None:
    """Apply recommended PRAGMA settings for performance and safety.

    Safe defaults for a consumer Windows machine:
    - WAL journal: concurrent reads while writing, less fsync pressure on SSD.
    - synchronous=NORMAL: one fsync per WAL checkpoint instead of per commit;
      safe with WAL (no data loss on OS crash, only on power-cut mid-checkpoint).
    - cache_size: 32 MB page cache in RAM — avoids repeated disk reads for hot
      pages (cluster list, face thumbnails) without over-committing memory.
    - temp_store=MEMORY: sorting/grouping temp tables stay in RAM; avoids
      writing temporary files to disk for the window-function queries.
    - busy_timeout: retry up to 5 s on a locked DB instead of failing
      immediately; prevents spurious errors when the web UI and a background
      scan task race for the write lock.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32768")   # 32 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA busy_timeout=5000")   # ms


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the database and tables if they don't exist. Return a connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug("Opening database: %s", db_path)
    conn = sqlite3.connect(str(db_path))
    configure_connection(conn)
    conn.executescript(CREATE_TABLES)

    # Track schema version
    cur = conn.execute("SELECT COUNT(*) FROM schema_version")
    if cur.fetchone()[0] == 0:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

    # Apply migrations
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply incremental schema migrations."""
    cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
    row = cur.fetchone()
    current = row[0] if row else 1

    if current < 2:
        # Add dismissed column for marking false-positive detections
        try:
            conn.execute(
                "ALTER TABLE faces ADD COLUMN dismissed INTEGER NOT NULL DEFAULT 0"
            )
            log.info("Migration v2: added faces.dismissed column")
        except sqlite3.OperationalError:
            pass  # column already exists (e.g. from re-init)
        conn.execute("UPDATE schema_version SET version = 2")
        current = 2

    if current < 3:
        # Embedding model changed from SFace (128-dim) to ArcFace (512-dim).
        # Old embeddings are incompatible — clear faces and clusters so
        # the user re-scans with the new model.  Photos are kept so
        # incremental scan knows which files exist but will re-detect faces
        # because there are no face rows left.
        face_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        cluster_count = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        conn.execute("DELETE FROM faces")
        conn.execute("DELETE FROM clusters")
        # Reset scanned_at so every photo is re-scanned
        conn.execute("UPDATE photos SET num_faces = 0, scanned_at = '1970-01-01 00:00:00'")
        conn.execute("UPDATE schema_version SET version = 3")
        current = 3
        if face_count > 0:
            log.info(
                "Migration v3: cleared %d faces and %d clusters — embedding model "
                "upgraded from SFace (128-dim) to ArcFace (512-dim). "
                "Please re-scan to regenerate embeddings.",
                face_count, cluster_count,
            )

    if current < 4:
        # Add indexes on dismissed and (cluster_id, dismissed).  These were
        # missing from earlier schema versions; CREATE INDEX IF NOT EXISTS is
        # safe to run on a DB that already has them (e.g. freshly created).
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_dismissed ON faces(dismissed)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_cluster_dismissed "
            "ON faces(cluster_id, dismissed)"
        )
        conn.execute("UPDATE schema_version SET version = 4")
        log.info("Migration v4: created idx_faces_dismissed and idx_faces_cluster_dismissed")
