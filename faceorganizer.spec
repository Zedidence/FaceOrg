# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for FaceOrganizer.

Build:
    pyinstaller faceorganizer.spec

Output: dist/FaceOrganizer/FaceOrganizer.exe (single-directory bundle)

Notes:
  - ONNX models (~250 MB) are NOT bundled. They auto-download to
    ~/.faceorganizer/models/ on first run, as in development.
  - All onnxruntime provider DLLs are included explicitly because
    PyInstaller's auto-analysis misses them on Windows.
"""

import os
import sys
from pathlib import Path

block_cipher = None

# ── Collect onnxruntime DLLs ────────────────────────────────────────────────
import onnxruntime
_ort_dir = Path(onnxruntime.__file__).parent
_ort_dlls = [
    (str(f), ".")
    for f in _ort_dir.glob("*.dll")
]

# ── Analysis ────────────────────────────────────────────────────────────────
a = Analysis(
    ["faceorganizer/__main__.py"],
    pathex=["."],
    binaries=_ort_dlls,
    datas=[
        ("faceorganizer/ui/resources/style_dark.qss",  "faceorganizer/ui/resources"),
        ("faceorganizer/ui/resources/style_light.qss", "faceorganizer/ui/resources"),
        ("faceorganizer/ui/resources/icons",           "faceorganizer/ui/resources/icons"),
    ],
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi._pybind_state",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors.typedefs",
        "sklearn.neighbors._partition_nodes",
        "sklearn.tree",
        "sklearn.tree._utils",
        "pillow_heif",
        "cv2",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtNetwork",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "flask", "jinja2", "werkzeug", "click",
        "tqdm",
        "tkinter", "matplotlib",
        "IPython", "jupyter",
        "test", "tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceOrganizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # No terminal window on Windows
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="faceorganizer/ui/resources/icons/app.ico",  # Uncomment when icon is added
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FaceOrganizer",
)
