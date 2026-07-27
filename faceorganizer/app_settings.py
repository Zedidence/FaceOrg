"""Persistent application settings stored in ~/.faceorganizer/app_settings.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from faceorganizer.config import DEFAULT_CLUSTER_THRESHOLD

_SETTINGS_DIR = Path.home() / ".faceorganizer"
_SETTINGS_FILE = _SETTINGS_DIR / "app_settings.json"


@dataclass
class AppSettings:
    # Detection
    detection_confidence: float = 0.9
    min_face_size: int = 40
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
    def load(cls) -> "AppSettings":
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
