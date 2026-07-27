"""Cluster naming utilities."""

from __future__ import annotations

import re
import sqlite3

from faceorganizer.database.core import rename_cluster as db_rename
from faceorganizer.logging_config import get_logger

log = get_logger("organizer.naming")

# Characters not safe for directory names
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_name(name: str) -> str:
    """Make a cluster name safe for use as a directory name."""
    name = _UNSAFE.sub("_", name).strip().strip(".")
    return name or "unnamed"


def rename_person(conn: sqlite3.Connection, cluster_id: int, new_name: str) -> bool:
    """Rename a cluster, sanitizing the name first. Returns True if found."""
    safe = sanitize_name(new_name)
    log.info("Renaming cluster %d -> %s", cluster_id, safe)
    return db_rename(conn, cluster_id, safe)
