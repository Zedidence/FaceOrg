"""CRUD operations for the FaceOrganizer database."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from faceorganizer.logging_config import get_logger
from faceorganizer.models import FaceInfo, PersonCluster, PhotoInfo

log = get_logger("database.core")

_AUTO_OR_MAYBE_RE = re.compile(r"^(Person_\d{3,}|maybe .*)$")


def insert_photo(conn: sqlite3.Connection, photo: PhotoInfo) -> int:
    """Insert a photo record. Returns the new row id."""
    cur = conn.execute(
        """INSERT INTO photos (path, file_size, width, height, format, exif_date,
                               num_faces, phash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(photo.path),
            photo.file_size,
            photo.width,
            photo.height,
            photo.format,
            photo.exif_date.isoformat() if photo.exif_date else None,
            photo.num_faces,
            photo.phash,
        ),
    )
    conn.commit()
    log.debug("Inserted photo %s (id=%d)", photo.path, cur.lastrowid)
    return cur.lastrowid


def photo_exists(conn: sqlite3.Connection, path: Path) -> bool:
    """Check if a photo has already been scanned."""
    cur = conn.execute("SELECT 1 FROM photos WHERE path = ?", (str(path),))
    return cur.fetchone() is not None


def insert_face(conn: sqlite3.Connection, face: FaceInfo, photo_id: int) -> int:
    """Insert a face record. Returns the new row id."""
    cur = conn.execute(
        """INSERT INTO faces (photo_id, bbox_x, bbox_y, bbox_w, bbox_h,
                              embedding, detection_confidence, cluster_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            photo_id,
            face.bbox_x,
            face.bbox_y,
            face.bbox_w,
            face.bbox_h,
            face.embedding.astype(np.float32).tobytes(),
            face.detection_confidence,
            face.cluster_id,
        ),
    )
    conn.commit()
    log.debug("Inserted face id=%d for photo_id=%d", cur.lastrowid, photo_id)
    return cur.lastrowid


def insert_faces_batch(conn: sqlite3.Connection, faces: list[FaceInfo], photo_id: int) -> None:
    """Insert multiple faces in a single transaction."""
    conn.executemany(
        """INSERT INTO faces (photo_id, bbox_x, bbox_y, bbox_w, bbox_h,
                              embedding, detection_confidence, cluster_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                photo_id,
                f.bbox_x,
                f.bbox_y,
                f.bbox_w,
                f.bbox_h,
                f.embedding.astype(np.float32).tobytes(),
                f.detection_confidence,
                f.cluster_id,
            )
            for f in faces
        ],
    )
    conn.commit()
    log.debug("Inserted %d face(s) for photo_id=%d", len(faces), photo_id)


def get_all_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Load all non-dismissed face embeddings from the database.

    Returns (face_ids, embeddings_matrix) where embeddings_matrix is shape (N, D).
    """
    cur = conn.execute("SELECT id, embedding FROM faces WHERE dismissed = 0")
    rows = cur.fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    face_ids = [r[0] for r in rows]
    embeddings = np.array(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32
    )
    return face_ids, embeddings


def get_embeddings_by_ids(
    conn: sqlite3.Connection, face_ids: list[int]
) -> tuple[list[int], np.ndarray]:
    """Load embeddings for a specific list of face IDs (non-dismissed only).

    Returns (face_ids, embeddings_matrix) — order matches the DB query, not the input list.
    """
    if not face_ids:
        return [], np.empty((0, 0), dtype=np.float32)
    placeholders = ",".join("?" * len(face_ids))
    cur = conn.execute(
        f"SELECT id, embedding FROM faces WHERE id IN ({placeholders}) AND dismissed = 0",
        face_ids,
    )
    rows = cur.fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    out_ids = [r[0] for r in rows]
    embeddings = np.array(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32
    )
    return out_ids, embeddings


def get_unassigned_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Load face embeddings for non-dismissed faces with no cluster assignment.

    Returns (face_ids, embeddings_matrix) for faces where cluster_id IS NULL.
    """
    cur = conn.execute(
        "SELECT id, embedding FROM faces WHERE cluster_id IS NULL AND dismissed = 0"
    )
    rows = cur.fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    face_ids = [r[0] for r in rows]
    embeddings = np.array(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32
    )
    return face_ids, embeddings


def get_cluster_centroids(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    """Compute the mean embedding (centroid) for each existing cluster.

    Returns {cluster_id: centroid_vector} where centroid_vector is L2-normalized.

    Uses a running-sum accumulator so memory is O(C * D) — one vector per
    cluster — rather than O(N * D) for the full embedding matrix.  For a
    300 K-face library with 500 clusters this saves ~600 MB of peak RAM.
    """
    cur = conn.execute(
        """SELECT f.cluster_id, f.embedding
           FROM faces f
           WHERE f.cluster_id IS NOT NULL AND f.dismissed = 0
           ORDER BY f.cluster_id"""
    )
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for row in cur:
        cid = row[0]
        emb = np.frombuffer(row[1], dtype=np.float32).copy()
        if cid in sums:
            sums[cid] += emb
            counts[cid] += 1
        else:
            sums[cid] = emb
            counts[cid] = 1

    centroids: dict[int, np.ndarray] = {}
    for cid, s in sums.items():
        centroid = s / counts[cid]
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid = centroid / norm
        centroids[cid] = centroid
    return centroids


def is_user_defined_name(name: str) -> bool:
    """Return True if *name* is a user-defined name (not auto-generated or 'maybe ...')."""
    return not _AUTO_OR_MAYBE_RE.match(name)


def get_named_clusters(conn: sqlite3.Connection) -> dict[int, str]:
    """Return {cluster_id: name} for user-renamed clusters.

    Excludes auto-generated names (Person_NNN) and existing "maybe ..." clusters,
    so only confirmed human-assigned names are used as references.
    """
    cur = conn.execute("SELECT id, name FROM clusters")
    return {row[0]: row[1] for row in cur if not _AUTO_OR_MAYBE_RE.match(row[1])}


def update_face_cluster(conn: sqlite3.Connection, face_id: int, cluster_id: int | None) -> None:
    """Update the cluster assignment for a face."""
    conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster_id, face_id))
    conn.commit()
    log.debug("Assigned face %d to cluster %s", face_id, cluster_id)


def update_face_clusters_batch(
    conn: sqlite3.Connection, assignments: list[tuple[int | None, int]]
) -> None:
    """Batch update cluster assignments. Each tuple is (cluster_id, face_id)."""
    conn.executemany("UPDATE faces SET cluster_id = ? WHERE id = ?", assignments)
    conn.commit()
    log.debug("Batch-assigned %d face(s) to clusters", len(assignments))


def insert_cluster(conn: sqlite3.Connection, name: str) -> int:
    """Insert a new cluster. Returns the row id."""
    cur = conn.execute("INSERT INTO clusters (name) VALUES (?)", (name,))
    conn.commit()
    log.debug("Inserted cluster %d ('%s')", cur.lastrowid, name)
    return cur.lastrowid


def get_clusters(conn: sqlite3.Connection) -> list[PersonCluster]:
    """Get all clusters with face counts (excludes dismissed faces)."""
    cur = conn.execute(
        """SELECT c.id, c.name, COUNT(f.id) as face_count,
                  c.representative_face_id, c.created_at
           FROM clusters c
           LEFT JOIN faces f ON f.cluster_id = c.id AND f.dismissed = 0
           GROUP BY c.id
           ORDER BY face_count DESC"""
    )
    results = []
    for row in cur.fetchall():
        results.append(
            PersonCluster(
                id=row[0],
                name=row[1],
                face_count=row[2],
                representative_face_id=row[3],
                created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
            )
        )
    return results


def clear_clusters(conn: sqlite3.Connection) -> None:
    """Remove all cluster assignments and cluster records (for re-clustering)."""
    conn.execute("UPDATE faces SET cluster_id = NULL")
    conn.execute("DELETE FROM clusters")
    conn.commit()
    log.info("Cleared all cluster assignments")


def get_cluster_by_id(conn: sqlite3.Connection, cluster_id: int) -> PersonCluster | None:
    """Get a single cluster by id (excludes dismissed faces from count)."""
    cur = conn.execute(
        """SELECT c.id, c.name, COUNT(f.id) as face_count,
                  c.representative_face_id, c.created_at
           FROM clusters c
           LEFT JOIN faces f ON f.cluster_id = c.id AND f.dismissed = 0
           WHERE c.id = ?
           GROUP BY c.id""",
        (cluster_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return PersonCluster(
        id=row[0],
        name=row[1],
        face_count=row[2],
        representative_face_id=row[3],
        created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
    )


def rename_cluster(conn: sqlite3.Connection, cluster_id: int, new_name: str) -> bool:
    """Rename a cluster. Returns True if the cluster existed."""
    cur = conn.execute(
        "UPDATE clusters SET name = ?, updated_at = datetime('now') WHERE id = ?",
        (new_name, cluster_id),
    )
    conn.commit()
    if cur.rowcount == 0:
        log.warning("rename_cluster: cluster %d not found", cluster_id)
        return False
    log.debug("Renamed cluster %d to '%s'", cluster_id, new_name)
    return True


def get_representative_face(
    conn: sqlite3.Connection,
    cluster_id: int,
    preferred_face_id: int | None = None,
) -> dict | None:
    """Return a single representative face dict for a cluster.

    If *preferred_face_id* is given and belongs to the cluster, that face is
    returned.  Otherwise the highest-confidence non-dismissed face is used.
    Avoids loading the entire cluster just to find one thumbnail.
    """
    if preferred_face_id is not None:
        cur = conn.execute(
            """SELECT f.id, f.photo_id, p.path,
                      f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                      f.detection_confidence
               FROM faces f JOIN photos p ON p.id = f.photo_id
               WHERE f.id = ? AND f.cluster_id = ? AND f.dismissed = 0""",
            (preferred_face_id, cluster_id),
        )
        row = cur.fetchone()
        if row:
            return {
                "face_id": row[0], "photo_id": row[1], "photo_path": row[2],
                "bbox_x": row[3], "bbox_y": row[4], "bbox_w": row[5], "bbox_h": row[6],
                "confidence": row[7],
            }

    # Fallback: highest-confidence face in the cluster
    cur = conn.execute(
        """SELECT f.id, f.photo_id, p.path,
                  f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                  f.detection_confidence
           FROM faces f JOIN photos p ON p.id = f.photo_id
           WHERE f.cluster_id = ? AND f.dismissed = 0
           ORDER BY f.detection_confidence DESC
           LIMIT 1""",
        (cluster_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "face_id": row[0], "photo_id": row[1], "photo_path": row[2],
        "bbox_x": row[3], "bbox_y": row[4], "bbox_w": row[5], "bbox_h": row[6],
        "confidence": row[7],
    }


def get_faces_for_cluster(
    conn: sqlite3.Connection, cluster_id: int
) -> list[dict]:
    """Get all faces in a cluster with their photo paths.

    Returns list of dicts with keys: face_id, photo_id, photo_path, bbox_x/y/w/h, confidence.
    """
    cur = conn.execute(
        """SELECT f.id, f.photo_id, p.path,
                  f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                  f.detection_confidence
           FROM faces f
           JOIN photos p ON p.id = f.photo_id
           WHERE f.cluster_id = ?
           ORDER BY f.detection_confidence DESC""",
        (cluster_id,),
    )
    return [
        {
            "face_id": r[0], "photo_id": r[1], "photo_path": r[2],
            "bbox_x": r[3], "bbox_y": r[4], "bbox_w": r[5], "bbox_h": r[6],
            "confidence": r[7],
        }
        for r in cur.fetchall()
    ]


def get_photos_for_cluster(conn: sqlite3.Connection, cluster_id: int) -> list[str]:
    """Get distinct photo paths for a cluster."""
    cur = conn.execute(
        """SELECT DISTINCT p.path
           FROM faces f JOIN photos p ON p.id = f.photo_id
           WHERE f.cluster_id = ?""",
        (cluster_id,),
    )
    return [r[0] for r in cur.fetchall()]


def get_face_by_id(conn: sqlite3.Connection, face_id: int) -> dict | None:
    """Get a single face with its photo path."""
    cur = conn.execute(
        """SELECT f.id, f.photo_id, p.path,
                  f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                  f.detection_confidence, f.cluster_id
           FROM faces f JOIN photos p ON p.id = f.photo_id
           WHERE f.id = ?""",
        (face_id,),
    )
    r = cur.fetchone()
    if r is None:
        return None
    return {
        "face_id": r[0], "photo_id": r[1], "photo_path": r[2],
        "bbox_x": r[3], "bbox_y": r[4], "bbox_w": r[5], "bbox_h": r[6],
        "confidence": r[7], "cluster_id": r[8],
    }


def merge_clusters(conn: sqlite3.Connection, keep_id: int, merge_id: int) -> bool:
    """Merge merge_id into keep_id: move all faces, delete the source cluster.

    Returns True if both clusters existed, False otherwise.
    """
    # Verify both clusters exist
    keep_exists = conn.execute("SELECT 1 FROM clusters WHERE id = ?", (keep_id,)).fetchone()
    merge_exists = conn.execute("SELECT 1 FROM clusters WHERE id = ?", (merge_id,)).fetchone()
    if not keep_exists or not merge_exists:
        log.warning(
            "merge_clusters: keep_id=%d or merge_id=%d not found", keep_id, merge_id
        )
        return False
    conn.execute(
        "UPDATE faces SET cluster_id = ? WHERE cluster_id = ?", (keep_id, merge_id)
    )
    conn.execute("DELETE FROM clusters WHERE id = ?", (merge_id,))
    conn.execute(
        "UPDATE clusters SET updated_at = datetime('now') WHERE id = ?", (keep_id,)
    )
    conn.commit()
    log.info("Merged cluster %d into %d", merge_id, keep_id)
    return True


def move_face_to_new_cluster(conn: sqlite3.Connection, face_id: int, new_name: str) -> int:
    """Split a face out of its current cluster into a new one. Returns the new cluster id."""
    new_id = insert_cluster(conn, new_name)
    conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (new_id, face_id))
    conn.commit()
    log.debug("Split face %d into new cluster %d ('%s')", face_id, new_id, new_name)
    return new_id


def dismiss_face(conn: sqlite3.Connection, face_id: int) -> bool:
    """Mark a face as dismissed (not a real face / false positive).

    Removes the cluster assignment and sets dismissed = 1.
    Returns True if the face existed.
    """
    cur = conn.execute(
        "UPDATE faces SET dismissed = 1, cluster_id = NULL WHERE id = ?", (face_id,)
    )
    conn.commit()
    if cur.rowcount == 0:
        log.warning("dismiss_face: face %d not found", face_id)
        return False
    log.debug("Dismissed face %d", face_id)
    return True


def restore_face(conn: sqlite3.Connection, face_id: int) -> bool:
    """Undo a dismiss — mark the face as active again (still unassigned).

    Returns True if the face existed.
    """
    cur = conn.execute("UPDATE faces SET dismissed = 0 WHERE id = ?", (face_id,))
    conn.commit()
    if cur.rowcount == 0:
        log.warning("restore_face: face %d not found", face_id)
        return False
    log.debug("Restored face %d", face_id)
    return True


def dismiss_cluster(conn: sqlite3.Connection, cluster_id: int) -> int:
    """Dismiss all faces in a cluster and delete the cluster.

    Returns the number of faces dismissed.
    """
    cur = conn.execute(
        "UPDATE faces SET dismissed = 1, cluster_id = NULL WHERE cluster_id = ?",
        (cluster_id,),
    )
    count = cur.rowcount
    conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))
    conn.commit()
    log.info("Dismissed cluster %d: %d face(s) marked as not-a-face", cluster_id, count)
    return count


def get_dismissed_faces(conn: sqlite3.Connection) -> list[dict]:
    """Get all dismissed faces with their photo paths."""
    cur = conn.execute(
        """SELECT f.id, f.photo_id, p.path,
                  f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                  f.detection_confidence
           FROM faces f
           JOIN photos p ON p.id = f.photo_id
           WHERE f.dismissed = 1
           ORDER BY f.id DESC"""
    )
    return [
        {
            "face_id": r[0], "photo_id": r[1], "photo_path": r[2],
            "bbox_x": r[3], "bbox_y": r[4], "bbox_w": r[5], "bbox_h": r[6],
            "confidence": r[7],
        }
        for r in cur.fetchall()
    ]


def get_cluster_embeddings(
    conn: sqlite3.Connection, cluster_id: int
) -> tuple[list[int], np.ndarray]:
    """Load face embeddings for all non-dismissed faces in a specific cluster.

    Returns (face_ids, embeddings_matrix) with shape (N, D).
    """
    cur = conn.execute(
        "SELECT id, embedding FROM faces WHERE cluster_id = ? AND dismissed = 0",
        (cluster_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    face_ids = [r[0] for r in rows]
    embeddings = np.array(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32
    )
    return face_ids, embeddings


def get_scan_stats(conn: sqlite3.Connection) -> dict:
    """Get summary statistics about the current database."""
    photo_count = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    cluster_count = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    # Combine three per-row face passes into one table scan.
    row = conn.execute(
        """SELECT
               SUM(CASE WHEN dismissed = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN dismissed = 0 AND cluster_id IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN dismissed = 1 THEN 1 ELSE 0 END)
           FROM faces"""
    ).fetchone()
    face_count = row[0] or 0
    unclustered = row[1] or 0
    dismissed = row[2] or 0
    return {
        "photos": photo_count,
        "faces": face_count,
        "clusters": cluster_count,
        "unclustered_faces": unclustered,
        "dismissed_faces": dismissed,
    }


# ── Duplicate photo groups ──────────────────────────────────────────────────
# Mirrors the clusters/faces.cluster_id pattern above, but groups *photos* by
# perceptual-hash similarity rather than *faces* by embedding similarity.

def insert_duplicate_group(conn: sqlite3.Connection) -> int:
    """Insert a new (empty) duplicate group. Returns the row id."""
    cur = conn.execute("INSERT INTO duplicate_groups DEFAULT VALUES")
    conn.commit()
    return cur.lastrowid


def clear_duplicate_groups(conn: sqlite3.Connection) -> None:
    """Remove all duplicate-group assignments and group records (for re-detection)."""
    conn.execute("UPDATE photos SET duplicate_group_id = NULL")
    conn.execute("DELETE FROM duplicate_groups")
    conn.commit()
    log.info("Cleared all duplicate-group assignments")


def update_photo_duplicate_groups_batch(
    conn: sqlite3.Connection, assignments: list[tuple[int | None, int]]
) -> None:
    """Batch update duplicate-group assignments. Each tuple is (group_id, photo_id)."""
    conn.executemany(
        "UPDATE photos SET duplicate_group_id = ? WHERE id = ?", assignments
    )
    conn.commit()
    log.debug("Batch-assigned %d photo(s) to duplicate groups", len(assignments))


def get_duplicate_groups(conn: sqlite3.Connection) -> list[dict]:
    """Get all duplicate groups with photo counts, largest group first."""
    cur = conn.execute(
        """SELECT dg.id, COUNT(p.id) as photo_count
           FROM duplicate_groups dg
           JOIN photos p ON p.duplicate_group_id = dg.id
           GROUP BY dg.id
           ORDER BY photo_count DESC"""
    )
    return [{"id": row[0], "photo_count": row[1]} for row in cur.fetchall()]


def get_photos_in_duplicate_group(conn: sqlite3.Connection, group_id: int) -> list[dict]:
    """Get all photos in a duplicate group, largest file first (usually the
    highest-quality copy, a reasonable default for the user to keep)."""
    cur = conn.execute(
        """SELECT id, path, width, height, file_size
           FROM photos
           WHERE duplicate_group_id = ?
           ORDER BY file_size DESC""",
        (group_id,),
    )
    return [
        {
            "photo_id": row[0], "path": row[1],
            "width": row[2], "height": row[3], "file_size": row[4],
        }
        for row in cur.fetchall()
    ]


def get_photo_by_id(conn: sqlite3.Connection, photo_id: int) -> dict | None:
    """Get a single photo by id."""
    cur = conn.execute(
        """SELECT id, path, width, height, file_size, duplicate_group_id
           FROM photos WHERE id = ?""",
        (photo_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "photo_id": row[0], "path": row[1],
        "width": row[2], "height": row[3], "file_size": row[4],
        "duplicate_group_id": row[5],
    }


def delete_photo(conn: sqlite3.Connection, photo_id: int) -> bool:
    """Delete a photo and its faces from the database. Returns True if it existed.

    Does not touch the file on disk — callers that also want the file removed
    (e.g. faceorganizer.actions.delete_photo) handle that separately.
    """
    conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
    cur = conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    conn.commit()
    if cur.rowcount == 0:
        log.warning("delete_photo: photo %d not found", photo_id)
        return False
    log.info("Deleted photo %d from database", photo_id)
    return True
