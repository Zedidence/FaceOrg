"""Persistent settings backed by <data_dir>/settings.json.

This is the web app's per-*folder* settings store, deliberately separate from
faceorganizer.app_settings.AppSettings (the desktop app's per-*machine*
settings). The web app is designed to be pointed at a different photo folder
per invocation (`faceorganizer serve <folder>`), each with its own detection
tuning, so its settings live next to that folder's database. The desktop app
instead keeps one global settings file plus a "recent folders" list, because
its settings (theme, window geometry, worker count) are about the user's
machine, not about any one folder. They are not synchronized with each other;
if both UIs are used against the same folder, detection/clustering settings
must be configured separately in each.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from faceorganizer.config import MIN_DETECTION_CONFIDENCE, MIN_FACE_SIZE

_FILENAME = "settings.json"


@dataclass
class Settings:
    detection_confidence: float = MIN_DETECTION_CONFIDENCE
    min_face_size: int = MIN_FACE_SIZE
    default_workers: int | None = None  # None = use scan_runner default

    @classmethod
    def load(cls, data_dir: Path) -> "Settings":
        path = data_dir / _FILENAME
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                detection_confidence=float(
                    raw.get("detection_confidence", MIN_DETECTION_CONFIDENCE)
                ),
                min_face_size=int(raw.get("min_face_size", MIN_FACE_SIZE)),
                default_workers=raw.get("default_workers"),
            )
        except Exception:
            return cls()

    def save(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / _FILENAME
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
