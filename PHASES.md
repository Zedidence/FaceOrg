# FaceOrganizer — Implementation Phases

## Phase 1: Project Scaffolding & Face Detection (MVP Core) ✅
- [x] Initialize git repo, create pyproject.toml, .gitignore
- [x] Implement config.py with image extensions, thresholds, paths
- [x] Implement models.py with PhotoInfo, FaceInfo, PersonCluster dataclasses
- [x] Implement scanner/file_discovery.py — walk directory, filter by image extension
- [x] Implement scanner/face_detector.py — ONNX-based SCRFD detection + ArcFace embeddings
- [x] Implement database/schema.py and database/core.py — SQLite tables, CRUD
- [x] Implement CLI: `faceorganizer scan <folder>` and `faceorganizer stats <folder>`

## Phase 2: Clustering ✅
- [x] Implement clustering/embeddings.py — load all embeddings from DB, L2-normalize
- [x] Implement clustering/cluster.py — DBSCAN with cosine metric (eps=0.55 default, `--threshold` CLI arg)
- [x] Write cluster assignments back to DB, auto-name clusters (Person_001, etc.)
- [x] CLI: `faceorganizer cluster <folder>` — run clustering on stored embeddings
- [x] CLI: `faceorganizer show <folder>` — print summary (N people, M faces, top people by count)

## Phase 3: Export & Organization ✅
- [x] Implement organizer/export.py — create per-person folders, copy or symlink photos
- [x] Implement organizer/naming.py — default names "Person_001", support rename
- [x] CLI: `faceorganizer export <folder> <output>` — write organized folder structure
- [x] CLI: `faceorganizer rename <folder> <cluster_id> "Name"` — assign a name to a cluster

## Phase 4: Web UI for Review ✅
- [x] Flask app with face thumbnail serving (crop + cache as JPEGs)
- [x] Dashboard page: grid of all people with representative face and count
- [x] Person detail page: all photos containing that person, face highlighted
- [x] Review page: side-by-side comparison for merging/splitting clusters
- [x] API endpoints: POST merge, POST split, POST rename

## Phase 5: Polish & Performance ✅
- [x] Parallel face detection using concurrent.futures.ProcessPoolExecutor
- [x] Incremental scanning — skip already-scanned photos (path + mtime check)
- [x] Face thumbnail cache on disk for fast web UI loading
- [x] EXIF date extraction for chronological sorting within a person
- [x] Progress bars via tqdm
- [x] Web UI: scan, cluster, and export operations from the dashboard
- [x] Background task system with progress polling for long-running web operations

## Phase 6: Efficiency & Resource Control ✅
- [x] `configure_connection()` helper — centralises all PRAGMA settings (WAL, synchronous=NORMAL, 32 MB cache, temp_store=MEMORY, busy_timeout=5 s); applied consistently to every connection (CLI, web requests, background tasks)
- [x] Composite index `faces(cluster_id, detection_confidence DESC)` — covers the ROW_NUMBER window queries on dashboard, review, and person-detail pages without a secondary sort
- [x] Index `photos(exif_date)` — speeds up timeline page ORDER BY
- [x] Batched scan commits — `_store_result` no longer commits per image; callers commit every 50 images, cutting fsync calls ~50× and reducing SSD write-amplification
- [x] `PRAGMA optimize` after scan and all clustering operations — refreshes query-planner statistics so subsequent reads benefit from accurate index cost estimates
- [x] YuNet per-thread size cache — dict keyed by (width, height), max 8 entries, FIFO eviction; reuses already-initialised detectors for recurring image dimensions (portrait vs. landscape)
- [x] Background task eviction thread — daemon thread runs every 30 min; prevents the in-process task registry from growing unbounded during long web-UI sessions
- [x] `get_scan_stats()` consolidated — three separate `COUNT(*)` scans of the faces table merged into one `SUM(CASE …)` query
- [x] `api/move-faces` batch update — replaced per-face loop + N commits with a single `executemany` + one commit via `update_face_clusters_batch`
- [x] SSE poll interval 0.5 s → 1.0 s — halves background polling load during long scan/cluster operations
