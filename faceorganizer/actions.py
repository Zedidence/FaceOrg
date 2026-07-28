"""Shared cluster-management actions used by the CLI, web app, and desktop UI.

Each interface used to call the database/clustering functions directly and grew
its own validation and side effects around them — e.g. rename used to be
sanitized in one interface and absorbed into "maybe X" clusters in another, but
never both in the same interface. This module is the single place that owns
that validation and those side effects, so the three interfaces can't drift
apart again. All three should import from here rather than reaching into
faceorganizer.database.core / faceorganizer.clustering.cluster directly for
anything a user can trigger (rename, merge, split, dismiss, recluster, export).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import send2trash

from faceorganizer.clustering.cluster import run_recluster, split_faces_by_similarity
from faceorganizer.config import DEFAULT_CLUSTER_THRESHOLD, MIN_CLUSTER_SIZE
from faceorganizer.database.core import (
    delete_photo as _db_delete_photo,
)
from faceorganizer.database.core import (
    dismiss_cluster as _db_dismiss_cluster,
)
from faceorganizer.database.core import (
    dismiss_face as _db_dismiss_face,
)
from faceorganizer.database.core import (
    get_cluster_by_id,
    get_face_by_id,
    get_photo_by_id,
    merge_clusters,
    move_face_to_new_cluster,
)
from faceorganizer.database.core import (
    restore_face as _db_restore_face,
)
from faceorganizer.logging_config import get_logger
from faceorganizer.organizer.export import export_by_person
from faceorganizer.organizer.naming import rename_person, rename_person_full, sanitize_name

log = get_logger("actions")

__all__ = [
    "ActionError",
    "rename_person",
    "rename_person_full",
    "merge_people",
    "split_face",
    "split_faces_batch",
    "dismiss_face",
    "restore_face",
    "dismiss_cluster",
    "recluster_person",
    "export_people",
    "delete_photo",
]


class ActionError(Exception):
    """A user-facing failure from a cluster-management action.

    Carries an HTTP-style *status* (400 for bad input, 404 for not-found) so
    the web layer can map it directly to a response code; the CLI and desktop
    UI only need the message.
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def merge_people(conn: sqlite3.Connection, keep_id: int, merge_id: int) -> None:
    """Merge *merge_id* into *keep_id*, deleting the source cluster."""
    if keep_id == merge_id:
        raise ActionError("cannot merge a cluster with itself")
    if not merge_clusters(conn, keep_id, merge_id):
        raise ActionError("one or both clusters not found", status=404)


def split_face(conn: sqlite3.Connection, face_id: int, new_name: str = "") -> dict:
    """Split a single face out of its cluster into a brand-new one.

    Returns {"cluster_id": int, "name": str}.
    """
    if get_face_by_id(conn, face_id) is None:
        raise ActionError("face not found", status=404)
    safe = sanitize_name(new_name) if new_name.strip() else f"Split_{face_id}"
    new_cluster_id = move_face_to_new_cluster(conn, face_id, safe)
    return {"cluster_id": new_cluster_id, "name": safe}


def split_faces_batch(
    conn: sqlite3.Connection,
    face_ids: list[int],
    base_name: str = "",
    eps: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict:
    """Split several faces into one or more similarity-grouped new clusters.

    Returns {"cluster_ids": list[int]}.
    """
    if not face_ids:
        raise ActionError("face_ids required")
    safe = sanitize_name(base_name) if base_name.strip() else "Split"
    new_ids = split_faces_by_similarity(conn, face_ids, safe, eps=eps)
    return {"cluster_ids": new_ids}


def dismiss_face(conn: sqlite3.Connection, face_id: int) -> None:
    """Mark a face as a false positive (not a real face)."""
    if not _db_dismiss_face(conn, face_id):
        raise ActionError("face not found", status=404)


def restore_face(conn: sqlite3.Connection, face_id: int) -> None:
    """Undo a face dismissal."""
    if not _db_restore_face(conn, face_id):
        raise ActionError("face not found", status=404)


def dismiss_cluster(conn: sqlite3.Connection, cluster_id: int) -> int:
    """Dismiss every face in a cluster and delete it. Returns the count dismissed."""
    if get_cluster_by_id(conn, cluster_id) is None:
        raise ActionError("cluster not found", status=404)
    return _db_dismiss_cluster(conn, cluster_id)


def recluster_person(
    conn: sqlite3.Connection,
    cluster_id: int,
    eps: float = DEFAULT_CLUSTER_THRESHOLD,
    min_samples: int = MIN_CLUSTER_SIZE,
) -> dict:
    """Re-split a cluster's faces into tighter sub-clusters."""
    try:
        return run_recluster(conn, cluster_id, eps=eps, min_samples=min_samples)
    except ValueError as e:
        raise ActionError(str(e), status=404) from e


def export_people(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    symlink: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    stop_event=None,
) -> dict:
    """Export every cluster's photos into per-person folders."""
    return export_by_person(
        conn, output_dir, symlink=symlink, on_progress=on_progress, stop_event=stop_event
    )


def delete_photo(conn: sqlite3.Connection, photo_id: int) -> None:
    """Send a photo's file to the OS Recycle Bin and remove it from the database.

    If the file no longer exists on disk (already moved/deleted outside the
    app), only the database record is removed — this is not treated as an
    error, since the end state the caller wants (photo gone) is still reached.
    """
    photo = get_photo_by_id(conn, photo_id)
    if photo is None:
        raise ActionError("photo not found", status=404)

    path = Path(photo["path"])
    if path.exists():
        try:
            send2trash.send2trash(str(path))
            log.info("Sent %s to the Recycle Bin", path)
        except OSError as e:
            # e.g. another process (an open preview/thumbnail generation) has
            # the file locked, which can happen transiently on Windows.
            raise ActionError(f"could not delete file: {e}", status=409) from e
    else:
        log.warning("delete_photo: %s no longer exists on disk", path)

    _db_delete_photo(conn, photo_id)
