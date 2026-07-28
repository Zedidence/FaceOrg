# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Everything before the "Initial commit" entry below predates this project's
git history (see `PHASES.md` for that pre-history: the original 6-phase
build covering detection, clustering, export, the web UI, and performance
work).

## [Unreleased]

### Added
- Perceptual near-duplicate photo detection, across all three interfaces:
  - Every scan now computes a perceptual hash per photo (schema v5:
    `photos.phash`, `photos.duplicate_group_id`, a new `duplicate_groups`
    table); `faceorganizer/duplicates.py::run_duplicate_detection()` groups
    photos by Hamming distance the same way faces are grouped by cosine
    distance (DBSCAN + ball-tree).
  - `actions.delete_photo()` sends a photo to the OS Recycle Bin (via
    `send2trash`) and removes its DB record.
  - CLI: `find-duplicates`, `duplicates`, `delete-photo`.
  - Web: a `/duplicates` review page, `/api/detect-duplicates`,
    `/api/delete-photo`.
  - Desktop: a `DuplicatesPanel` (Operations → Find Duplicates).
  - Report-only by design — nothing is deleted automatically; the user
    reviews each group and picks what to remove.
- `.github/workflows/ci.yml`: ruff + pytest (with a coverage gate) on every
  push/PR.
- `faceorganizer/actions.py`: a single shared layer for rename/merge/split/
  dismiss/recluster/export, used by the CLI, web app, and desktop UI instead
  of each interface calling the database/clustering modules directly.
- CLI subcommands `merge`, `split`, and `dismiss-cluster`, for parity with
  the web and desktop UIs.
- Test suite grew from 14 tests (covering 2 modules) to 93 tests covering
  the actions layer, CLI, Flask API, export, and both settings stores.
- Logging in `database/core.py`, `web/app.py`, and `web/tasks.py` — these
  were previously silent even on failure.
- `.pre-commit-config.yaml`, `CONTRIBUTING.md`, this changelog.

### Changed
- Ruff's rule set expanded from the bare pyflakes/pycodestyle defaults to
  also include import sorting, pyupgrade, and bugbear.
- Standardized the cluster/recluster distance threshold on `0.55`
  (`config.DEFAULT_CLUSTER_THRESHOLD`) as a single source of truth; it had
  silently diverged to `0.30` in the desktop app's settings default.

### Fixed
- Desktop Settings panel's detection-confidence/min-face-size fields were
  silently ignored during scanning.
- Rename behavior had diverged across all three interfaces (sanitization vs.
  absorbing nearby unassigned faces vs. neither) — unified via `actions.py`.
- `dismiss-cluster` (web) didn't validate the cluster existed before
  "succeeding".
- Web export dropped its cancellation token, so a running export couldn't
  actually be cancelled despite the UI/task infrastructure supporting it.
- `python -m faceorganizer <command>` (and the `faceorganizer` console
  script) couldn't reach the CLI at all — it always launched the GUI
  regardless of arguments.
- `flask` and `tqdm` were used but not declared as dependencies.
- Version string mismatch between `__init__.py` (0.1.0) and `pyproject.toml`
  (0.2.0).

### Security
- Documented (did not change) the localhost-only threat model: the web
  server always binds to `127.0.0.1`, and `/api/browse` exposes an
  unrestricted local filesystem listing that would need constraining if this
  app were ever exposed beyond localhost.

## Initial commit
- Baseline import of the existing codebase: face detection (YuNet) and
  recognition (ArcFace) pipeline, DBSCAN/agglomerative clustering, SQLite
  storage, and three interfaces (PySide6 desktop app, Flask web app, CLI)
  sharing one backend. No git history existed before this point.
