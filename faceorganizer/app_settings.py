"""Persistent application settings stored in ~/.faceorganizer/app_settings.json.

This is the desktop app's per-*machine* settings store, deliberately separate
from faceorganizer.web.settings.Settings (the web app's per-*folder*
settings). The desktop app is a single long-lived install with a "recent
folders" list, window geometry, and a theme choice — all machine-level
concerns — whereas the web app is launched fresh against one folder at a time
(`faceorganizer serve <folder>`) and keeps its detection/clustering tuning
next to that folder's database instead. They are not synchronized with each
other; if both UIs are used against the same folder, detection/clustering
settings must be configured separately in each.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from faceorganizer.config import DEFAULT_CLUSTER_THRESHOLD, MIN_DETECTION_CONFIDENCE, MIN_FACE_SIZE

_SETTINGS_DIR = Path.home() / ".faceorganizer"
_SETTINGS_FILE = _SETTINGS_DIR / "app_settings.json"


@dataclass
class AppSettings:
    # Detection
    detection_confidence: float = MIN_DETECTION_CONFIDENCE
    min_face_size: int = MIN_FACE_SIZE
    # Performance (None = use RuntimeProfile.recommended)
    worker_count: int | None = None
    # Clustering
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD
    incremental_clustering: bool = True
    # UI
    theme: str = "dark"
    tile_size: int = 120
    # Recent folders (most recent first, max 10)
    recent_folders: list[str] = field(default_factory=list)
    # Window geometry
    window_width: int = 1280
    window_height: int = 800
    sidebar_width: int = 220

    @classmethod
    def load(cls) -> AppSettings:
        """Load settings from disk; return defaults on any error."""
        if not _SETTINGS_FILE.exists():
            return cls()
        try:
            data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**valid)
        except Exception:
            return cls()

    def save(self) -> None:
        """Persist settings to disk."""
        _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_recent_folder(self, path: str) -> None:
        """Add a folder to the recent list (dedup, newest first, max 10)."""
        if path in self.recent_folders:
            self.recent_folders.remove(path)
        self.recent_folders.insert(0, path)
        self.recent_folders = self.recent_folders[:10]

    def effective_workers(self, recommended: int) -> int:
        """Return the worker count to use (user override or hardware recommendation)."""
        return self.worker_count if self.worker_count is not None else recommended
