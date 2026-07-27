# FaceOrganizer

Scan a folder of photos, detect faces, group them by person, and organize your collection — all locally on your machine.

Uses OpenCV's YuNet for face detection and ArcFace (w600k_r50) for face recognition. No cloud APIs, no C++ compiler needed.

## Desktop App Quickstart

Everything after launch is point-and-click — no browser, no terminal.

### Option A — Run the .exe (no Python required)

1. Build the executable once (requires Python installed on the build machine):

   ```bash
   pip install pyinstaller
   pyinstaller faceorganizer.spec
   ```

2. The output is `dist/FaceOrganizer/FaceOrganizer.exe`.  
   Copy the entire `dist/FaceOrganizer/` folder to any Windows machine and double-click `FaceOrganizer.exe`.

### Option B — Run from source (Python 3.10+)

```bash
git clone https://github.com/your-username/FaceOrganizer.git
cd FaceOrganizer
pip install -e .
python -m faceorganizer
```

**GPU acceleration (optional — makes scanning significantly faster):**

```bash
pip install -e ".[gpu]"    # any Windows GPU via DirectML
pip install -e ".[cuda]"   # NVIDIA only (CUDA)
```

### First launch

FaceOrganizer detects your hardware automatically (GPU, CPU cores, RAM) and configures itself. No setup wizard needed.

On the **very first scan**, two models download to `~/.faceorganizer/models/`:

| Model | Size | Purpose |
|-------|------|---------|
| YuNet | ~3 MB | Face detection |
| ArcFace w600k_r50 | ~250 MB | Face recognition |

The download happens silently in the background — the status bar shows **"N images found, initialising models…"** while it completes. Subsequent scans skip this step entirely.

---

### Workflow

#### 1. Open a folder

Click **Open Folder** in the toolbar (or `File → Open Folder`, `Ctrl+O`).  
Select the folder containing your photos. Subfolders are included automatically.

#### 2. Scan for faces

Click **Scan** in the toolbar.

The status bar shows progress as each photo is processed. The first scan of a large library takes time — only new or changed files are re-processed on subsequent runs.

> **Note:** If the progress bar says "initialising models…" and appears to stall, the ArcFace model is downloading or compiling GPU shaders for the first time. This is normal and happens only once.

#### 3. Cluster faces into people

Click **Cluster** in the toolbar.

FaceOrganizer groups all detected faces by person using ArcFace embeddings. Groups appear in the **People** panel on the left, auto-named `Person_001`, `Person_002`, etc.

#### 4. Review and rename

Click any person card in the **People** panel to open their detail view. From here you can:

- **Rename** — click the name field or use the Rename button in the header
- **Merge** — right-click a card → *Merge into…* to combine two people into one
- **Dismiss** — right-click a face tile → *Dismiss* to exclude false detections (pets, posters, reflections)
- **Split** — right-click a face tile → *Split off* to move a face to a new person

Use the search bar at the top of the People panel to filter by name.

#### 5. Export organized folders

Click **Export** in the toolbar, choose an output folder, and click **Export**.

Photos are copied into per-person subfolders named after each person you renamed:

```
Output folder/
  Alice/
    IMG_001.jpg
    IMG_042.jpg
  Bob/
    IMG_003.jpg
  Person_003/
    IMG_007.jpg   ← still auto-named, rename it first if you like
```

Enable **Create symbolic links** to link rather than copy (saves disk space).

#### 6. Adding photos later

Drop new files into the same folder, click **Scan** again, then **Cluster** (the app uses incremental clustering by default — new faces are matched to your existing named people without losing any names or merges).

---

### Tips

| Symptom | Fix |
|---------|-----|
| Scan is slow / fan spins up | Open **Settings** (toolbar) and lower **Worker threads** |
| Too many false face detections | In Settings, raise **Detection confidence** (try 92–95%) or **Min face size** (try 50–60 px) |
| Wrong faces merged into one person | Lower **Cluster threshold** in Settings (default 0.30 — lower = stricter separation) |
| A person is split across multiple cards | Merge the cards: right-click → *Merge into…* |
| App seems slow after a large scan | The first people panel load generates thumbnails — it speeds up on subsequent opens |

---

## Full CLI Reference

### 1. Install

```bash
git clone https://github.com/your-username/FaceOrganizer.git
cd FaceOrganizer
pip install -e .
```

Optional GPU acceleration: `pip install -e ".[gpu]"` (DirectML, any Windows GPU) or `pip install -e ".[cuda]"` (NVIDIA).

On first run, models download automatically to `~/.faceorganizer/models/` (~253 MB total).

### 2. Scan photos for faces

```bash
python -m faceorganizer scan "C:\Users\you\Pictures\Vacation"
```

This recursively finds all images (JPG, PNG, HEIC, WebP, etc.), detects faces in parallel, extracts 512-dimensional ArcFace recognition embeddings, and stores everything in a local SQLite database at `<folder>/.faceorganizer/faces.db`.

Re-running scan on the same folder is fast — only new or modified files are processed (mtime-based incremental scanning).

```bash
python -m faceorganizer scan "C:\Photos" --no-recursive   # top-level only
python -m faceorganizer scan "C:\Photos" --no-parallel    # single-threaded mode
python -m faceorganizer scan "C:\Photos" --workers 2      # limit to 2 parallel threads
```

### 3. Check scan results

```bash
python -m faceorganizer stats "C:\Users\you\Pictures\Vacation"
```

Output:

```
Database: C:\Users\you\Pictures\Vacation\.faceorganizer\faces.db
  Photos:            247
  Faces:             312
  Clusters:          0
  Unclustered faces: 312
```

### 4. Cluster faces by person

```bash
python -m faceorganizer cluster "C:\Users\you\Pictures\Vacation"
```

This groups detected faces into people using DBSCAN clustering on cosine distance. Clusters are auto-named `Person_001`, `Person_002`, etc.

Adjust the distance threshold (lower = stricter matching, default 0.55):

```bash
python -m faceorganizer cluster "C:\Users\you\Pictures\Vacation" --threshold 0.4
```

### 5. Incremental clustering (after adding new photos)

If you've already clustered and renamed/merged people, then added new photos and rescanned, use `--incremental` to cluster only the new faces **without losing your existing names and merges**:

```bash
python -m faceorganizer scan "C:\Users\you\Pictures\Vacation"      # picks up new photos
python -m faceorganizer cluster "C:\Users\you\Pictures\Vacation" --incremental
```

This will:
1. Match new faces to existing clusters based on centroid similarity
2. Run DBSCAN on any remaining unmatched faces to discover new people
3. Leave all your renamed and merged clusters untouched

Without `--incremental`, clustering wipes all existing cluster assignments and starts fresh.

### 6. Re-cluster a single person

If a cluster contains mixed people, split it into sub-clusters:

```bash
python -m faceorganizer recluster "C:\Users\you\Pictures\Vacation" 3
python -m faceorganizer recluster "C:\Users\you\Pictures\Vacation" 3 --threshold 0.4
```

### 6b. Merge and split clusters

Merge one cluster into another (the merged cluster is deleted):

```bash
python -m faceorganizer merge "C:\Users\you\Pictures\Vacation" 1 5
```

Split a single face out of its cluster into a brand-new one:

```bash
python -m faceorganizer split "C:\Users\you\Pictures\Vacation" 42 "New Person"
```

### 7. View cluster summary

```bash
python -m faceorganizer show "C:\Users\you\Pictures\Vacation"
```

Output:

```
Database: C:\Users\you\Pictures\Vacation\.faceorganizer\faces.db
  Photos: 247  |  Faces: 312  |  People: 18  |  Unclustered: 24

Top people by face count:
  Person_001            42 face(s)
  Person_002            31 face(s)
  Person_003            28 face(s)
  ...
```

### 8. Dismiss false-positive detections

If the detector picks up non-human subjects (pets, statues, posters, etc.), dismiss them so they're excluded from clustering and export:

```bash
python -m faceorganizer dismiss "C:\Users\you\Pictures\Vacation" 42
```

Changed your mind? Restore it:

```bash
python -m faceorganizer dismiss "C:\Users\you\Pictures\Vacation" 42 --restore
```

To dismiss an entire cluster (every face in it, e.g. a cluster of pet photos), which also deletes the cluster:

```bash
python -m faceorganizer dismiss-cluster "C:\Users\you\Pictures\Vacation" 7
```

In the web UI, use the **Dismiss** button on any face thumbnail (`POST /api/dismiss`) and **Restore** (`POST /api/restore`).

### 9. Rename clusters

```bash
python -m faceorganizer rename "C:\Users\you\Pictures\Vacation" 1 "Alice"
python -m faceorganizer rename "C:\Users\you\Pictures\Vacation" 2 "Bob"
```

### 10. Export organized folders

```bash
python -m faceorganizer export "C:\Users\you\Pictures\Vacation" "C:\Sorted"
```

This creates a folder per person and copies their photos into it:

```
C:\Sorted\
  Alice/
    IMG_001.jpg
    IMG_042.jpg
  Bob/
    IMG_003.jpg
  Person_003/
    IMG_007.jpg
```

Use `--symlink` to create symbolic links instead of copying files:

```bash
python -m faceorganizer export "C:\Users\you\Pictures\Vacation" "C:\Sorted" --symlink
```

### 11. Review in the web UI

```bash
python -m faceorganizer serve "C:\Users\you\Pictures\Vacation"
```

Opens a local web app at `http://127.0.0.1:5000` **and automatically launches your default browser**. Everything below is available from the browser — no further CLI interaction needed.

- **Dashboard** — guided 3-step workflow (Scan → Cluster → Review & Export) with real-time progress. The **Workers** field controls how many parallel threads are used for scanning (lower = less CPU/RAM). The **Export** path has a **Browse…** button to navigate your filesystem without typing a path. Search, filter, and sort people by name or face count. Select multiple people with checkboxes for batch merge or dismiss.
- **Person detail** — **drag and drop** faces to reassign them to other people via the slide-out sidebar. Click any person name to **rename inline**. View full source photos in a lightbox, merge, split, or dismiss.
- **Review** — side-by-side cluster comparison with merge controls. Drag faces between panels to reassign without merging entire clusters.
- **Timeline** — browse faces by EXIF photo date, filter by person name.
- **Dismissed** — view dismissed false-positive detections and restore any by mistake.
- **Settings** — configure detection confidence threshold, minimum face size (px), and default worker count. Settings persist in `.faceorganizer/settings.json` and apply to all future scans.
- **Logs** — live view of the application log with colour-coded levels (error/warning/debug), configurable line count, 5-second auto-refresh, and a Clear button.

#### Keyboard shortcuts

Press `?` on any page to see available shortcuts. Key bindings include:

| Key | Action |
|-----|--------|
| `G` | Go to dashboard |
| `R` | Go to review |
| `T` | Go to timeline |
| `/` | Focus search (dashboard) |
| `S` | Start scan (dashboard) |
| `C` | Run clustering (dashboard) |
| `N` | Rename person (person detail) |
| `A` | Select all faces (person detail) |
| `?` | Toggle shortcut help |

#### Theme

Click the **Light/Dark** toggle in the navigation bar to switch themes. Your preference is saved in the browser.

Use `--port`, `--debug`, or `--no-browser` to customise startup:

```bash
python -m faceorganizer serve "C:\Photos" --port 8080 --debug
python -m faceorganizer serve "C:\Photos" --no-browser   # don't auto-open browser
```

## Verbosity

Add `-v` for progress details or `-vv` for full debug output:

```bash
python -m faceorganizer -v scan "C:\Photos"
python -m faceorganizer -vv scan "C:\Photos"
```

## How it works

1. **Scan** — finds images, runs YuNet face detection in parallel across worker threads, extracts 512-dimensional ArcFace embeddings. GPU is used automatically (DirectML on any Windows GPU, CUDA on NVIDIA). Only new or modified files are re-processed on subsequent runs. Database writes are committed in batches of 50 images to minimise SSD write pressure.
2. **Cluster** — groups embeddings using DBSCAN (cosine distance, no need to specify the number of people). Incremental mode matches new faces to existing named clusters before running DBSCAN on any remainder — your renamed and merged clusters are preserved.
3. **Review** — native desktop UI (PySide6 / Qt 6) with a DigiKam-inspired People grid. Rename, merge, split, and dismiss faces without leaving the app. Includes a Timeline view (by EXIF date), a side-by-side Review panel for cluster comparison, and a Dismissed panel for recovering false detections.
4. **Export** — copies (or symlinks) photos into per-person folders named after the people you renamed.

Everything runs locally — no internet connection after models are downloaded. The database, thumbnails, and settings live inside `<photo folder>/.faceorganizer/`.

## Performance notes

- **SSD-friendly writes** — scan commits are batched (every 50 images) rather than per-image, cutting fsync calls ~50× and reducing write-amplification on consumer SSDs.
- **SQLite tuning** — every connection is configured with `synchronous=NORMAL` (safe with WAL, ~3× faster than the default `FULL`), a 32 MB page cache, and in-memory temp tables. A `busy_timeout` of 5 s prevents spurious "database locked" errors when the web UI and a background task write concurrently.
- **Composite index** — `faces(cluster_id, detection_confidence DESC)` covers the window-function queries on the dashboard, review, and person-detail pages without a separate sort step.
- **EXIF date index** — `photos(exif_date)` speeds up timeline page queries.
- **YuNet per-thread cache** — up to 8 distinct image sizes are cached per worker thread (FIFO eviction), so portrait and landscape photos from the same camera reuse an already-initialised detector instead of recreating it.
- **CPU headroom** — parallel scan caps at `min(4, cpu_count − 1)` workers by default, limiting peak CPU and RAM usage. Override with `--workers N` on the CLI or the Workers field in the web UI. The Settings page lets you set a persistent default.
- **Background task cleanup** — a daemon thread evicts completed/errored tasks every 30 minutes so the in-process task registry doesn't grow unbounded during a long web-UI session.

## Requirements

- Python 3.10+
- No C++ compiler needed
- Models auto-download on first run (no authentication required)
- GPU optional: install `.[gpu]` for DirectML (any Windows GPU) or `.[cuda]` for NVIDIA CUDA

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Project structure

```
faceorganizer/
  ui/                   # PySide6 desktop GUI
    main_window.py      #   Root QMainWindow
    sidebar.py          #   Left navigation panel
    panels/             #   People, person detail, review, timeline, dismissed, settings
    widgets/            #   Reusable widgets (thumbnail cache, face grid, photo viewer, …)
    dialogs/            #   Rename, merge, export dialogs
    resources/          #   QSS themes (dark / light) and icons
    theme.py            #   apply_theme() — loads QSS, handles PyInstaller paths
  workers/              # QThread background workers (scan, cluster, export)
  scanner/              # Image discovery and face detection (YuNet + ArcFace)
  clustering/           # DBSCAN face grouping
  database/             # SQLite storage
  organizer/            # Export to per-person folders
  hardware.py           # Auto-detect GPU / CPU cores / RAM → RuntimeProfile
  app_settings.py       # Persistent user settings (~/.faceorganizer/app_settings.json)
  config.py             # Paths and constants
  models.py             # PhotoInfo, FaceInfo, PersonCluster dataclasses
  logging_config.py
faceorganizer.spec      # PyInstaller build spec → dist/FaceOrganizer/FaceOrganizer.exe
```
