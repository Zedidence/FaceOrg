"""Cluster naming utilities."""

from __future__ import annotations

import re
import sqlite3

from faceorganizer.database.core import is_user_defined_name
from faceorganizer.database.core import rename_cluster as db_rename
from faceorganizer.logging_config import get_logger

log = get_logger("organizer.naming")

# Characters not safe for directory names
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_name(name: str) -> str:
    """Make a cluster name safe for use as a directory name."""
    name = _UNSAFE.sub("_", name).strip().strip(".")
    return name or "unnamed"


def rename_person_full(
    conn: sqlite3.Connection, cluster_id: int, new_name: str
) -> dict[str, int | str] | None:
    """Rename a cluster: sanitize the name, then absorb nearby matches.

    Absorbing means pulling in unassigned faces close to the cluster and
    merging any "maybe <name>" clusters, via clustering.cluster.absorb_after_rename.
    This is the single rename path shared by the CLI, web, and desktop UIs so
    the three interfaces can't drift on rename behavior again.

    Returns a dict with the sanitized name and absorb stats (``absorbed_faces``,
    ``merged_clusters``), or None if the cluster was not found.
    """
    # Imported locally to avoid a module-load-time dependency on scikit-learn
    # (pulled in by clustering.cluster) for callers that only need sanitize_name.
    from faceorganizer.clustering.cluster import absorb_after_rename

    safe = sanitize_name(new_name)
    log.info("Renaming cluster %d -> %s", cluster_id, safe)
    if not db_rename(conn, cluster_id, safe):
        return None

    result: dict[str, int | str] = {"name": safe}
    if is_user_defined_name(safe):
        result.update(absorb_after_rename(conn, cluster_id, safe))
    return result


def rename_person(conn: sqlite3.Connection, cluster_id: int, new_name: str) -> bool:
    """Rename a cluster, sanitizing the name and absorbing nearby matches.

    Returns True if the cluster was found and renamed.
    """
    return rename_person_full(conn, cluster_id, new_name) is not None
