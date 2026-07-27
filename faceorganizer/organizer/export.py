"""Export organized photo folders grouped by person."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from faceorganizer.database.core import get_clusters, get_photos_for_cluster
from faceorganizer.logging_config import get_logger
from faceorganizer.organizer.naming import sanitize_name

log = get_logger("organizer.export")


def export_by_person(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    symlink: bool = False,
    on_progress=None,
    stop_event=None,
) -> dict[str, int]:
    """Create per-person folders and copy (or symlink) their photos.

    Returns a dict mapping cluster name -> number of photos exported.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = get_clusters(conn)

    if not clusters:
        log.warning("No clusters found — run 'cluster' first")
        return {}

    summary: dict[str, int] = {}
    total = len(clusters)

    for done, cluster in enumerate(clusters):
        if stop_event and stop_event.is_set():
            break

        folder_name = sanitize_name(cluster.name)
        person_dir = output_dir / folder_name
        person_dir.mkdir(exist_ok=True)

        photo_paths = get_photos_for_cluster(conn, cluster.id)
        exported = 0

        for photo_path_str in photo_paths:
            src = Path(photo_path_str)
            if not src.exists():
                log.warning("Source photo missing: %s", src)
                continue

            dest = person_dir / src.name
            # Handle duplicate filenames by appending a suffix
            if dest.exists():
                stem = src.stem
                suffix = src.suffix
                counter = 1
                while dest.exists():
                    dest = person_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            if symlink:
                dest.symlink_to(src)
            else:
                shutil.copy2(src, dest)
            exported += 1

        summary[cluster.name] = exported
        log.info("Exported %d photos for %s", exported, cluster.name)
        if on_progress:
            on_progress(done + 1, total)

    return summary
