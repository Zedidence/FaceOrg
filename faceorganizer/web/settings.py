"""Persistent settings backed by <data_dir>/settings.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_FILENAME = "settings.json"


@dataclass
class Settings:
    detection_confidence: float = 0.9   # mirrors MIN_DETECTION_CONFIDENCE
    min_face_size: int = 40             # mirrors MIN_FACE_SIZE
    default_workers: int | None = None  # None = use scan_runner default

    @classmethod
    def load(cls, data_dir: Path) -> "Settings":
        path = data_dir / _FILENAME
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                detection_confidence=float(raw.get("detection_confidence", 0.9)),
                min_face_size=int(raw.get("min_face_size", 40)),
                default_workers=raw.get("default_workers"),
            )
        except Exception:
            return cls()

    def save(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / _FILENAME
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
