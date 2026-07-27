"""Core scan logic shared by CLI and web UI, with parallel detection support.

Uses ThreadPoolExecutor (not ProcessPoolExecutor) so that all threads share a
single GPU context.  ONNX Runtime releases the GIL during inference, so threads
still achieve real parallelism for the GPU-bound work while image I/O overlaps
naturally.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Callable

from faceorganizer.logging_config import get_logger
from faceorganizer.models import FaceInfo, PhotoInfo

log = get_logger("scanner.runner")

# Commit accumulated DB writes to disk every N images.  Batching reduces the
# number of fsync calls ~50×, which matters for SSD write endurance and speed.
# On crash, at most _COMMIT_BATCH images need to be re-scanned on the next run.
_COMMIT_BATCH = 50

# Maximum rate at which progress signals are emitted to the UI thread.
# Emitting on every image with 4 workers can flood Qt's cross-thread event
# queue at hundreds of events/second, starving the Win32 message pump and
# causing "(Not Responding)".  100 ms gives a smooth-looking progress bar
# (~10 updates/s) without any queue build-up.
_PROGRESS_THROTTLE_S = 0.1


@dataclass
class ScanProgress:
    """Mutable progress tracker for a scan operation."""

    total: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    faces_found: int = 0
    current_file: str = ""
    done: bool = False
    cancelled: bool = False
    error_message: str = ""


def _load_scan_times(conn: sqlite3.Connection) -> dict[str, str]:
    """Load all (path, scanned_at) pairs into a dict for fast lookup."""
    cur = conn.execute("SELECT path, scanned_at FROM photos")
    return {row[0]: row[1] for row in cur}


def _needs_rescan(scan_times: dict[str, str], path: Path) -> bool:
    """Check if a photo needs (re-)scanning based on path and mtime."""
    scanned_at_str = scan_times.get(str(path))
    if scanned_at_str is None:
        return True  # never scanned
    try:
        from datetime import datetime, timezone

        scanned_at = datetime.fromisoformat(scanned_at_str)
        # SQLite datetime('now') produces UTC without tzinfo
        if scanned_at.tzinfo is None:
            scanned_at = scanned_at.replace(tzinfo=timezone.utc)
        file_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return file_mtime > scanned_at
    except Exception:
        return False  # on error, skip re-scan


def _detect_single(image_path: Path) -> tuple[PhotoInfo, list[FaceInfo]] | str:
    """Worker function for threaded detection. Returns result or error string."""
    try:
        from faceorganizer.scanner.face_detector import detect_faces

        return detect_faces(image_path)
    except Exception as e:
        return str(e)


def run_scan(
    conn: sqlite3.Connection,
    scan_root: Path,
    *,
    recursive: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
    progress: ScanProgress | None = None,
    on_progress: Callable[[ScanProgress], None] | None = None,
    stop_event: threading.Event | None = None,
) -> ScanProgress:
    """Run a full scan: discover images, detect faces, store results.

    Args:
        conn: Database connection.
        scan_root: Root folder to scan.
        recursive: Scan subdirectories.
        parallel: Use ThreadPoolExecutor for detection.
        max_workers: Number of parallel workers (default: cpu_count - 1, min 1).
        progress: Optional pre-created progress tracker.
        on_progress: Callback invoked after each image is processed.

    Returns the final ScanProgress.
    """
    from faceorganizer.scanner.file_discovery import discover_images

    if progress is None:
        progress = ScanProgress()

    images = discover_images(scan_root, recursive=recursive)
    progress.total = len(images)

    # Notify immediately so callers can show a progress bar with the correct total
    if on_progress:
        on_progress(progress)

    if not images:
        progress.done = True
        return progress

    # Filter to images that need scanning (new or modified)
    scan_times = _load_scan_times(conn)
    to_scan: list[Path] = []
    for img_path in images:
        if _needs_rescan(scan_times, img_path):
            to_scan.append(img_path)
        else:
            progress.skipped += 1

    log.info(
        "%d images to scan (%d skipped as unchanged)",
        len(to_scan), progress.skipped,
    )

    if not to_scan:
        progress.done = True
        if on_progress:
            on_progress(progress)
        return progress

    # Warm up models before the scan loop so the user sees meaningful status
    # messages instead of a silent "not responding" window during initialisation.
    from faceorganizer.scanner.face_detector import warmup_models

    def _on_warmup_status(msg: str) -> None:
        if on_progress:
            progress.current_file = msg
            on_progress(progress)

    warmup_models(_on_warmup_status)
    progress.current_file = ""

    if parallel and len(to_scan) > 1:
        workers = max_workers or min(4, max(1, (cpu_count() or 2) - 1))
        workers = min(workers, len(to_scan))
        log.info("Parallel scan with %d workers", workers)
        _scan_parallel(conn, to_scan, workers, progress, on_progress, stop_event)
    else:
        _scan_sequential(conn, to_scan, progress, on_progress, stop_event)

    if progress.cancelled:
        log.info("Scan cancelled after %d images", progress.processed)
    else:
        # Update query-planner statistics after a bulk write so subsequent reads
        # (clustering, web UI) benefit from accurate index cost estimates.
        conn.execute("PRAGMA optimize")

    progress.done = True
    if on_progress:
        on_progress(progress)
    return progress


def _scan_sequential(
    conn: sqlite3.Connection,
    images: list[Path],
    progress: ScanProgress,
    on_progress: Callable[[ScanProgress], None] | None,
    stop_event: threading.Event | None = None,
) -> None:
    from faceorganizer.scanner.face_detector import detect_faces

    last_emit = time.monotonic()

    for i, img_path in enumerate(images):
        if stop_event and stop_event.is_set():
            progress.cancelled = True
            break

        progress.current_file = img_path.name
        try:
            photo, faces = detect_faces(img_path)
        except Exception as e:
            log.warning("Failed to process %s: %s", img_path.name, e)
            progress.errors += 1
            progress.processed += 1
            continue

        _store_result(conn, photo, faces)
        progress.faces_found += len(faces)
        progress.processed += 1
        if (i + 1) % _COMMIT_BATCH == 0:
            conn.commit()

        if on_progress:
            now = time.monotonic()
            if now - last_emit >= _PROGRESS_THROTTLE_S:
                on_progress(progress)
                last_emit = now

    conn.commit()  # flush any remaining uncommitted images
    if on_progress:
        on_progress(progress)  # always emit the final state


def _scan_parallel(
    conn: sqlite3.Connection,
    images: list[Path],
    workers: int,
    progress: ScanProgress,
    on_progress: Callable[[ScanProgress], None] | None,
    stop_event: threading.Event | None = None,
) -> None:
    db_lock = threading.Lock()
    emit_lock = threading.Lock()   # serialises throttle-check + emit
    commit_counter = 0
    last_emit = time.monotonic()

    def _maybe_emit(*, force: bool = False) -> None:
        """Emit progress at most every _PROGRESS_THROTTLE_S seconds.

        Multiple worker threads call this concurrently; the emit_lock ensures
        only one thread emits per interval so the UI event queue never grows
        unbounded regardless of how many workers are running.
        """
        if on_progress is None:
            return
        nonlocal last_emit
        with emit_lock:
            now = time.monotonic()
            if force or now - last_emit >= _PROGRESS_THROTTLE_S:
                on_progress(progress)
                last_emit = now

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        future_to_path = {
            pool.submit(_detect_single, img): img for img in images
        }
        for future in as_completed(future_to_path):
            if stop_event and stop_event.is_set():
                progress.cancelled = True
                pool.shutdown(wait=False, cancel_futures=True)
                break

            img_path = future_to_path[future]
            progress.current_file = img_path.name
            try:
                result = future.result()
            except Exception as e:
                log.warning("Worker error for %s: %s", img_path.name, e)
                progress.errors += 1
                progress.processed += 1
                _maybe_emit()
                continue

            if isinstance(result, str):
                log.warning("Failed to process %s: %s", img_path.name, result)
                progress.errors += 1
            else:
                photo, faces = result
                with db_lock:
                    _store_result(conn, photo, faces)
                    commit_counter += 1
                    if commit_counter % _COMMIT_BATCH == 0:
                        conn.commit()
                progress.faces_found += len(faces)

            progress.processed += 1
            _maybe_emit()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        with db_lock:
            conn.commit()
        _maybe_emit(force=True)  # always emit final state


def _store_result(
    conn: sqlite3.Connection, photo: PhotoInfo, faces: list[FaceInfo]
) -> None:
    """Stage detection results in the current transaction (no commit).

    The caller is responsible for committing in batches via _COMMIT_BATCH.
    Grouping the delete + insert for each photo in the same open transaction
    keeps the operation atomic: a rolled-back batch leaves the previous
    on-disk state intact so the photo is simply re-scanned on the next run.
    """
    import numpy as np

    conn.execute("DELETE FROM photos WHERE path = ?", (str(photo.path),))
    cur = conn.execute(
        """INSERT INTO photos (path, file_size, width, height, format, exif_date, num_faces)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            str(photo.path),
            photo.file_size,
            photo.width,
            photo.height,
            photo.format,
            photo.exif_date.isoformat() if photo.exif_date else None,
            photo.num_faces,
        ),
    )
    photo_id = cur.lastrowid
    if faces:
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
