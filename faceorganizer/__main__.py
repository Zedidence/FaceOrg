"""Entry point — launch the FaceOrganizer Qt desktop application."""

from __future__ import annotations

import sys


def main() -> None:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    from faceorganizer.app_settings import AppSettings
    from faceorganizer.hardware import probe_hardware
    from faceorganizer.logging_config import setup_logging
    from faceorganizer.ui.main_window import MainWindow

    # Set up file logging in ~/.faceorganizer/logs/
    from pathlib import Path
    log_dir = Path.home() / ".faceorganizer" / "logs"
    setup_logging(verbosity=0, log_dir=log_dir)

    app = QApplication(sys.argv)
    app.setApplicationName("FaceOrganizer")
    app.setApplicationVersion("0.2.0")
    app.setOrganizationName("FaceOrganizer")

    # High-DPI support (Qt 6 enables this by default but be explicit)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    profile = probe_hardware()
    settings = AppSettings.load()

    window = MainWindow(profile, settings)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
