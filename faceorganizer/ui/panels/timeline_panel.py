"""Timeline panel — faces grouped by EXIF date, paginated by year."""

from __future__ import annotations

import sqlite3
from collections import defaultdict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from faceorganizer.ui.widgets.face_grid import FaceGrid
from faceorganizer.ui.widgets.thumbnail_cache import ThumbnailCache


class TimelinePanel(QWidget):
    """Shows all faces grouped by the date the photo was taken.

    Years are paginated: only one year's worth of faces is fetched and rendered
    at a time, avoiding the 300 K-row fetchall that a full load would require.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TimelinePanel")
        self._conn: sqlite3.Connection | None = None
        self._cache: ThumbnailCache | None = None
        self._years: list[str] = []   # available years, ascending
        self._year_index: int = 0     # index into _years for the current view
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Year navigation bar ──────────────────────────────────────────────
        nav = QWidget()
        nav.setObjectName("timelineNav")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(12, 6, 12, 6)

        self._prev_btn = QPushButton("← Earlier")
        self._prev_btn.clicked.connect(self._go_prev)
        nav_layout.addWidget(self._prev_btn)

        self._year_label = QLabel("")
        self._year_label.setObjectName("timelineYearLabel")
        self._year_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._year_label, stretch=1)

        self._next_btn = QPushButton("Later →")
        self._next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self._next_btn)

        layout.addWidget(nav)

        # ── Scrollable day groups ────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setObjectName("timelineScroll")
        layout.addWidget(self._scroll, stretch=1)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 8, 12, 8)
        self._content_layout.setSpacing(12)
        self._scroll.setWidget(self._content)

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self, conn: sqlite3.Connection, cache: ThumbnailCache) -> None:
        self._conn = conn
        self._cache = cache
        self._years = self._fetch_years()
        # Start at the most recent year
        self._year_index = max(0, len(self._years) - 1)
        self._render_current_year()

    # ── Navigation ───────────────────────────────────────────────────────────

    def _go_prev(self) -> None:
        if self._year_index > 0:
            self._year_index -= 1
            self._render_current_year()

    def _go_next(self) -> None:
        if self._year_index < len(self._years) - 1:
            self._year_index += 1
            self._render_current_year()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fetch_years(self) -> list[str]:
        """Return sorted list of years that have non-dismissed faces with dates."""
        if self._conn is None:
            return []
        cur = self._conn.execute(
            """SELECT DISTINCT SUBSTR(p.exif_date, 1, 4)
               FROM faces f
               JOIN photos p ON p.id = f.photo_id
               WHERE f.dismissed = 0 AND p.exif_date IS NOT NULL
               ORDER BY 1 ASC"""
        )
        years = [r[0] for r in cur.fetchall()]
        # Include an "Unknown" bucket if any faces lack EXIF dates
        has_undated = self._conn.execute(
            """SELECT 1 FROM faces f JOIN photos p ON p.id = f.photo_id
               WHERE f.dismissed = 0 AND p.exif_date IS NULL LIMIT 1"""
        ).fetchone()
        if has_undated:
            years.append("Unknown")
        return years

    def _render_current_year(self) -> None:
        """Clear the scroll area and populate it with faces for the current year."""
        # Clear old widgets
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        if not self._years:
            self._year_label.setText("No photos with dates")
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            self._content_layout.addWidget(QLabel("No faces found. Run a scan first."))
            self._content_layout.addStretch()
            return

        year = self._years[self._year_index]
        self._year_label.setText(year)
        self._prev_btn.setEnabled(self._year_index > 0)
        self._next_btn.setEnabled(self._year_index < len(self._years) - 1)

        # Fetch only this year's faces — avoids full-library fetchall
        if year == "Unknown":
            cur = self._conn.execute(
                """SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                          p.path, p.exif_date, f.detection_confidence
                   FROM faces f
                   JOIN photos p ON p.id = f.photo_id
                   WHERE f.dismissed = 0 AND p.exif_date IS NULL
                   ORDER BY p.path ASC"""
            )
        else:
            cur = self._conn.execute(
                """SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                          p.path, p.exif_date, f.detection_confidence
                   FROM faces f
                   JOIN photos p ON p.id = f.photo_id
                   WHERE f.dismissed = 0 AND p.exif_date LIKE ?
                   ORDER BY p.exif_date ASC, p.path ASC""",
                (f"{year}%",),
            )
        rows = cur.fetchall()

        # Group by day (YYYY-MM-DD)
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            day = row[6][:10] if row[6] else "Unknown date"
            groups[day].append({
                "face_id": row[0],
                "bbox_x": row[1], "bbox_y": row[2],
                "bbox_w": row[3], "bbox_h": row[4],
                "photo_path": row[5],
                "confidence": row[7],
            })

        if not groups:
            self._content_layout.addWidget(QLabel("No faces for this year."))
            self._content_layout.addStretch()
            return

        for day, faces in groups.items():
            box = QGroupBox(f"{day}  ({len(faces):,} faces)")
            box_layout = QVBoxLayout(box)
            grid = FaceGrid(tile_size=80)
            grid.setFixedHeight(260)
            grid.load_faces(faces, self._cache)
            box_layout.addWidget(grid)
            self._content_layout.addWidget(box)

        self._content_layout.addStretch()
        # Scroll back to top when year changes
        self._scroll.verticalScrollBar().setValue(0)
