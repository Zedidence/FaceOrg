"""Main application window."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow,
    QMessageBox, QSplitter, QToolBar,
)

from faceorganizer.app_settings import AppSettings
from faceorganizer.config import get_db_path
from faceorganizer.database.core import get_clusters, get_scan_stats
from faceorganizer.database.schema import init_db
from faceorganizer.hardware import RuntimeProfile
from faceorganizer.ui.content_stack import (
    PANEL_DISMISSED, PANEL_PEOPLE, PANEL_PERSON_DETAIL,
    PANEL_REVIEW, PANEL_SETTINGS, PANEL_TIMELINE, PANEL_WELCOME,
    ContentStack,
)
from faceorganizer.ui.panels.dismissed_panel import DismissedPanel
from faceorganizer.ui.panels.people_panel import PeoplePanel
from faceorganizer.ui.panels.person_detail_panel import PersonDetailPanel
from faceorganizer.ui.panels.review_panel import ReviewPanel
from faceorganizer.ui.panels.settings_panel import SettingsPanel
from faceorganizer.ui.panels.timeline_panel import TimelinePanel
from faceorganizer.ui.panels.welcome_panel import WelcomePanel
from faceorganizer.ui.sidebar import SidebarPanel
from faceorganizer.ui.theme import apply_theme
from faceorganizer.ui.widgets.progress_bar import OperationProgressBar
from faceorganizer.ui.widgets.thumbnail_cache import ThumbnailCache
from faceorganizer.workers.cluster_worker import ClusterWorker
from faceorganizer.workers.export_worker import ExportWorker
from faceorganizer.workers.scan_worker import ScanWorker


class MainWindow(QMainWindow):
    """Root application window."""

    def __init__(self, profile: RuntimeProfile, settings: AppSettings) -> None:
        super().__init__()
        self._profile = profile
        self._settings = settings

        # Runtime state
        self._scan_root: Path | None = None
        self._db_conn: sqlite3.Connection | None = None
        self._cache: ThumbnailCache | None = None

        # Worker management (one active worker at a time)
        self._active_thread: QThread | None = None
        self._active_worker = None

        self._setup_window()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_central_widget()
        self._create_status_bar()

        apply_theme(QApplication.instance(), self._settings.theme == "dark")
        self._restore_state()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle("FaceOrganizer")
        self.resize(self._settings.window_width, self._settings.window_height)
        self.setMinimumSize(900, 600)

    def _create_menu_bar(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")
        file_menu.addAction("Open Folder…", self._open_folder, "Ctrl+O")
        self._recent_menu = file_menu.addMenu("Recent Folders")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close, "Ctrl+Q")

        # Operations menu
        ops_menu = mb.addMenu("&Operations")
        ops_menu.addAction("Scan for Faces", self._start_scan, "Ctrl+Shift+S")
        ops_menu.addAction("Cluster Faces", self._start_cluster, "Ctrl+Shift+C")
        ops_menu.addAction("Export…", self._start_export, "Ctrl+Shift+E")

        # View menu
        view_menu = mb.addMenu("&View")
        view_menu.addAction("Toggle Dark/Light Theme", self._toggle_theme)
        view_menu.addSeparator()
        view_menu.addAction("People", lambda: self._show_panel(PANEL_PEOPLE))
        view_menu.addAction("Review", lambda: self._show_panel(PANEL_REVIEW))
        view_menu.addAction("Timeline", lambda: self._show_panel(PANEL_TIMELINE))
        view_menu.addAction("Dismissed", lambda: self._show_panel(PANEL_DISMISSED))

        # Help menu
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("About", self._show_about)

    def _create_toolbar(self) -> None:
        tb = QToolBar("Main Toolbar")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._action_open = tb.addAction("Open Folder", self._open_folder)
        tb.addSeparator()
        self._action_scan = tb.addAction("Scan", self._start_scan)
        self._action_cluster = tb.addAction("Cluster", self._start_cluster)
        self._action_export = tb.addAction("Export", self._start_export)
        tb.addSeparator()
        self._action_settings = tb.addAction("Settings", lambda: self._show_panel(PANEL_SETTINGS))

        self._set_operations_enabled(False)

    def _create_central_widget(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")

        # Sidebar
        self._sidebar = SidebarPanel()
        self._sidebar.person_selected.connect(self._on_person_selected)
        self._sidebar.view_selected.connect(self._on_view_selected)
        splitter.addWidget(self._sidebar)

        # Content stack
        self._stack = ContentStack()

        self._welcome = WelcomePanel()
        self._welcome.open_folder_requested.connect(self._open_folder)
        self._stack.add_panel(PANEL_WELCOME, self._welcome)

        self._people = PeoplePanel()
        self._people.person_selected.connect(self._on_person_selected)
        self._people.clusters_changed.connect(self._on_clusters_changed)
        self._stack.add_panel(PANEL_PEOPLE, self._people)

        self._detail = PersonDetailPanel(self._settings)
        self._detail.back_requested.connect(lambda: self._show_panel(PANEL_PEOPLE))
        self._detail.clusters_changed.connect(self._on_clusters_changed)
        self._stack.add_panel(PANEL_PERSON_DETAIL, self._detail)

        self._review = ReviewPanel()
        self._review.clusters_changed.connect(self._on_clusters_changed)
        self._stack.add_panel(PANEL_REVIEW, self._review)

        self._timeline = TimelinePanel()
        self._stack.add_panel(PANEL_TIMELINE, self._timeline)

        self._dismissed = DismissedPanel()
        self._dismissed.faces_changed.connect(self._on_clusters_changed)
        self._stack.add_panel(PANEL_DISMISSED, self._dismissed)

        self._settings_panel = SettingsPanel(self._settings, self._profile)
        self._settings_panel.settings_saved.connect(self._on_settings_saved)
        self._stack.add_panel(PANEL_SETTINGS, self._settings_panel)

        splitter.addWidget(self._stack)
        splitter.setSizes([self._settings.sidebar_width, 1000])
        splitter.setStretchFactor(1, 1)

        self._splitter = splitter
        self.setCentralWidget(splitter)

        self._stack.show_panel(PANEL_WELCOME)

    def _create_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setObjectName("mainStatusBar")

        self._status_label = QLabel("Ready")
        sb.addWidget(self._status_label)

        self._progress = OperationProgressBar()
        self._progress.cancel_requested.connect(self._cancel_worker)
        sb.addPermanentWidget(self._progress)

    def _restore_state(self) -> None:
        """Reopen the last folder if available."""
        if self._settings.recent_folders:
            last = self._settings.recent_folders[0]
            if Path(last).exists():
                self._open_folder_path(Path(last))

    # ── Folder management ────────────────────────────────────────────────────

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Photo Folder")
        if folder:
            self._open_folder_path(Path(folder))

    def _open_folder_path(self, path: Path) -> None:
        if self._db_conn:
            self._db_conn.close()

        self._scan_root = path
        db_path = get_db_path(path)
        self._db_conn = init_db(db_path)
        self._cache = ThumbnailCache(path, self._profile.thumbnail_resolution)

        self._settings.add_recent_folder(str(path))
        self._settings.save()
        self._rebuild_recent_menu()

        self.setWindowTitle(f"FaceOrganizer — {path.name}")
        self._set_operations_enabled(True)

        # Populate people view and switch to it
        self._people.load(self._db_conn, self._cache)
        self._show_panel(PANEL_PEOPLE)
        self._refresh_sidebar()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        for folder in self._settings.recent_folders:
            p = Path(folder)
            self._recent_menu.addAction(p.name, lambda f=p: self._open_folder_path(f))
        if not self._settings.recent_folders:
            empty = self._recent_menu.addAction("(none)")
            empty.setEnabled(False)

    # ── Panel switching ──────────────────────────────────────────────────────

    def _show_panel(self, name: str) -> None:
        if self._db_conn is None and name not in (PANEL_WELCOME, PANEL_SETTINGS):
            self._stack.show_panel(PANEL_WELCOME)
            return

        if name == PANEL_REVIEW:
            self._review.load(self._db_conn, self._cache)
        elif name == PANEL_TIMELINE:
            self._timeline.load(self._db_conn, self._cache)
        elif name == PANEL_DISMISSED:
            self._dismissed.load(self._db_conn, self._cache)

        self._stack.show_panel(name)

    def _on_view_selected(self, view_id: str) -> None:
        panel_map = {
            "people": PANEL_PEOPLE,
            "review": PANEL_REVIEW,
            "timeline": PANEL_TIMELINE,
            "dismissed": PANEL_DISMISSED,
            "settings": PANEL_SETTINGS,
        }
        panel = panel_map.get(view_id)
        if panel:
            self._show_panel(panel)

    def _on_person_selected(self, cluster_id: int) -> None:
        if self._db_conn is None:
            return
        self._sidebar.select_person(cluster_id)
        self._detail.load(self._db_conn, self._cache, cluster_id)
        self._stack.show_panel(PANEL_PERSON_DETAIL)

    # ── Workers ──────────────────────────────────────────────────────────────

    def _start_scan(self) -> None:
        if self._scan_root is None or self._active_thread is not None:
            return
        workers = self._settings.effective_workers(self._profile.recommended_workers)
        self._launch_worker(
            ScanWorker(
                self._scan_root,
                workers,
                self._settings.detection_confidence,
                self._settings.min_face_size,
            ),
            "Scanning…",
        )

    def _start_cluster(self) -> None:
        if self._scan_root is None or self._active_thread is not None:
            return
        self._launch_worker(
            ClusterWorker(
                self._scan_root,
                incremental=self._settings.incremental_clustering,
                eps=self._settings.cluster_threshold,
            ),
            "Clustering…",
        )

    def _start_export(self) -> None:
        if self._scan_root is None or self._active_thread is not None:
            return
        from faceorganizer.ui.dialogs.export_dialog import ExportDialog
        dlg = ExportDialog(self)
        if not dlg.exec():
            return
        self._launch_worker(
            ExportWorker(self._scan_root, dlg.output_dir(), dlg.use_symlinks()),
            "Exporting…",
        )

    def _launch_worker(self, worker, label: str) -> None:
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_worker_thread_finished)

        worker.progress.connect(self._on_worker_progress)
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)

        self._active_thread = thread
        self._active_worker = worker
        self._set_operations_enabled(False)
        self._progress.start(label)
        thread.start()

    @Slot(int, int, str)
    def _on_worker_progress(self, processed: int, total: int, message: str) -> None:
        if processed == 0 and total > 0:
            # Image discovery just finished — models may still be downloading/
            # initialising.  Use the status message from warmup_models() when
            # available; fall back to the generic label if not.
            label = message if message else f"Scanning — {total} images found, initialising models…"
            self._progress.update_progress(0, total, label)
        else:
            self._progress.update_progress(processed, total, message)

    @Slot(dict)
    def _on_worker_finished(self, result: dict) -> None:
        self._progress.stop()

        # Reopen the DB connection on the main thread now that writing is done.
        if self._scan_root and self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass
            from faceorganizer.database.schema import init_db
            self._db_conn = init_db(get_db_path(self._scan_root))
            if self._cache:
                self._cache.invalidate()

        # Show brief summary immediately so the status bar updates before the
        # (potentially slow) thumbnail-generation pass inside _people.load().
        if "faces_found" in result:
            msg = f"Scan complete — {result['faces_found']} faces in {result['processed']} photos"
        elif "num_clusters" in result:
            msg = f"Clustering complete — {result['num_clusters']} clusters"
        elif "total_exported" in result:
            msg = f"Export complete — {result['total_exported']} photos exported"
        else:
            msg = "Done"
        self._status_label.setText(msg)

        # Defer the panel reload to the next event-loop iteration so the UI
        # can repaint the status bar before the thumbnail-generation work
        # starts.  Generating new thumbnails opens each source photo with PIL
        # on the main thread; deferring lets the window stay responsive-looking
        # while the work begins.
        if self._scan_root and self._db_conn:
            QTimer.singleShot(0, self._reload_people_panel)

    def _reload_people_panel(self) -> None:
        """Reload the people panel and sidebar (called deferred from _on_worker_finished)."""
        self._people.load(self._db_conn, self._cache)
        self._refresh_sidebar()

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self._progress.stop()
        self._set_operations_enabled(True)
        QMessageBox.critical(self, "Operation Failed", message)

    def _on_worker_thread_finished(self) -> None:
        self._active_thread = None
        self._active_worker = None
        self._set_operations_enabled(self._scan_root is not None)

    def _cancel_worker(self) -> None:
        if self._active_worker is not None:
            self._active_worker.cancel()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _refresh_sidebar(self) -> None:
        if self._db_conn is None:
            return
        clusters = get_clusters(self._db_conn)
        stats = get_scan_stats(self._db_conn)
        self._sidebar.refresh_clusters(clusters, stats)

    def _on_clusters_changed(self) -> None:
        self._refresh_sidebar()
        if self._stack.current_name() == PANEL_PEOPLE:
            self._people.refresh()

    def _set_operations_enabled(self, enabled: bool) -> None:
        for action in (self._action_scan, self._action_cluster, self._action_export):
            action.setEnabled(enabled)

    def _toggle_theme(self) -> None:
        self._settings.theme = "light" if self._settings.theme == "dark" else "dark"
        self._settings.save()
        apply_theme(QApplication.instance(), self._settings.theme == "dark")

    def _on_settings_saved(self) -> None:
        self._status_label.setText("Settings saved")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About FaceOrganizer",
            "<b>FaceOrganizer 0.2.0</b><br><br>"
            "Detect, cluster, and organize photos by face.<br>"
            "All processing is local — no cloud required.<br><br>"
            f"Hardware: {self._profile.summary()}",
        )

    def closeEvent(self, event) -> None:
        # Save window geometry
        self._settings.window_width = self.width()
        self._settings.window_height = self.height()
        try:
            sizes = self._splitter.sizes()
            if sizes:
                self._settings.sidebar_width = sizes[0]
        except Exception:
            pass
        self._settings.save()

        if self._db_conn:
            self._db_conn.close()
        event.accept()
