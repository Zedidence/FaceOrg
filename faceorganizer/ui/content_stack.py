"""ContentStack — QStackedWidget wrapper with named panel switching."""

from __future__ import annotations

from PySide6.QtWidgets import QStackedWidget, QWidget

PANEL_WELCOME = "welcome"
PANEL_PEOPLE = "people"
PANEL_PERSON_DETAIL = "person_detail"
PANEL_REVIEW = "review"
PANEL_TIMELINE = "timeline"
PANEL_DISMISSED = "dismissed"
PANEL_DUPLICATES = "duplicates"
PANEL_SETTINGS = "settings"


class ContentStack(QStackedWidget):
    """Named-panel stacked widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel_index: dict[str, int] = {}

    def add_panel(self, name: str, widget: QWidget) -> None:
        index = self.addWidget(widget)
        self._panel_index[name] = index

    def show_panel(self, name: str) -> None:
        index = self._panel_index.get(name)
        if index is not None:
            self.setCurrentIndex(index)

    def current_name(self) -> str | None:
        idx = self.currentIndex()
        for name, i in self._panel_index.items():
            if i == idx:
                return name
        return None
