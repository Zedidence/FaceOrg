"""Discover image files in a directory tree."""

from __future__ import annotations

import os
from pathlib import Path

from faceorganizer.config import DATA_DIR_NAME, IMAGE_EXTENSIONS
from faceorganizer.logging_config import get_logger

log = get_logger("scanner.discovery")


def discover_images(root: Path, recursive: bool = True) -> list[Path]:
    """Find all supported image files under *root*.

    Skips the .faceorganizer data directory.
    Returns a sorted list of absolute paths.

    Uses os.walk / os.scandir instead of Path.rglob so that directory
    entries are fetched with a single scandir() syscall per folder rather
    than one stat() call per path.  On a 370 K-file library this cuts
    discovery time from ~30 s (rglob) to ~3 s.
    """
    root = root.resolve()
    images: list[Path] = []

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune the data directory in-place so os.walk never descends into it
            dirnames[:] = [d for d in dirnames if d != DATA_DIR_NAME]
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    images.append(Path(dirpath, fname))
    else:
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    if os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                        images.append(Path(entry.path))

    images.sort()
    log.info("Discovered %d image(s) in %s (recursive=%s)", len(images), root, recursive)
    return images
