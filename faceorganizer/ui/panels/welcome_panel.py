"""Welcome panel — shown when no folder is open."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class WelcomePanel(QWidget):
    """Full-window welcome state: prompt the user to open a folder."""

    open_folder_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WelcomePanel")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        title = QLabel("FaceOrganizer")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Open a folder of photos to get started.")
        subtitle.setObjectName("welcomeSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        open_btn = QPushButton("Open Folder…")
        open_btn.setObjectName("welcomeOpenButton")
        open_btn.setFixedWidth(180)
        open_btn.clicked.connect(self.open_folder_requested)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addWidget(open_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
