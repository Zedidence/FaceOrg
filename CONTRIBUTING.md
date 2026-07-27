# Contributing

FaceOrganizer is currently a single-maintainer project, but this doc exists so
future work (by you or anyone else) stays consistent with how the codebase is
already organized.

## Setup

```bash
pip install -e ".[dev]"
pre-commit install   # optional, runs ruff on every commit
```

## Before committing

```bash
ruff check .
pytest
```

CI (`.github/workflows/ci.yml`) runs both of these plus a coverage gate
(`--cov-fail-under=65`) on every push/PR to `main`. If you add new dev
dependencies, keep `pyproject.toml`'s `[project.optional-dependencies].dev`
and the CI workflow in sync — CI installs via `pip install -e ".[dev]"`.

## Where things live

- `faceorganizer/database/` — SQLite schema + CRUD. `schema.py` owns
  migrations (`SCHEMA_VERSION` in that file); bump it and add a migration
  function in `_migrate()` for any schema change.
- `faceorganizer/clustering/` — DBSCAN/agglomerative clustering logic.
- `faceorganizer/actions.py` — **the** shared layer for anything a user can
  trigger on a cluster/face (rename, merge, split, dismiss, recluster,
  export). The CLI, web app, and desktop UI all call into this rather than
  the database/clustering modules directly, specifically so the three
  interfaces can't silently diverge on validation or side effects the way
  they had before this module existed (see the git history around the
  "Extract shared actions layer" commit for the concrete bugs that caused).
  **If you're adding a new user-triggerable operation, add it here first**,
  then wire all three interfaces to call it.
- `faceorganizer/cli/`, `faceorganizer/web/`, `faceorganizer/ui/` — the three
  interfaces. Each should be a thin layer over `actions.py` plus
  interface-specific concerns (argparse, Flask routes/templates, Qt widgets).
- `faceorganizer/organizer/` — export-to-folders and name sanitization.

## Testing

- `tests/conftest.py` has shared fixtures/helpers (`conn`, `make_embedding`,
  `add_photo`, `add_faces`) — reuse these rather than re-deriving them.
- Coverage is gated in CI, but the gate (see
  `[tool.coverage.run]` in `pyproject.toml`) deliberately omits `ui/`,
  `workers/`, `scanner/`, `hardware.py`, and `__main__.py`. These need
  `pytest-qt`/a display, a Qt event loop, or real downloaded ONNX models
  respectively — none of which exist in this project's test setup today.
  If you add that infrastructure, remove the relevant omit entry so the gate
  actually covers it.
- Tests that touch `faceorganizer/app_settings.py` must monkeypatch
  `_SETTINGS_DIR`/`_SETTINGS_FILE` (see `tests/test_app_settings.py`) — this
  module writes to a real path under the user's home directory by default,
  and a test that doesn't redirect it will clobber the real settings file on
  whatever machine runs the test.

## Commit style

Commit messages explain *why*, not just *what* — the diff already shows what
changed. Look at recent commits (`git log`) for the expected level of detail;
in short, a one-line summary plus a body that gives the reasoning/context a
future reader won't get from the code alone.
