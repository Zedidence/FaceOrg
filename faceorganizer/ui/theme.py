"""QSS theme loader and applicator."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication


def _resource_dir() -> Path:
    """Return the path to ui/resources/, handling PyInstaller bundles."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "faceorganizer" / "ui" / "resources"
    return Path(__file__).parent / "resources"


def apply_theme(app: QApplication, dark: bool) -> None:
    """Load and apply the QSS stylesheet matching the requested theme."""
    name = "style_dark.qss" if dark else "style_light.qss"
    qss_path = _resource_dir() / name
    try:
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass  # No stylesheet yet — use Qt defaults
