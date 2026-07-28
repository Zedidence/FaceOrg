"""Face detection and embedding extraction.

Uses YuNet for face detection and ArcFace (w600k_r50) for recognition
embeddings (512-dim).  The ArcFace ONNX model is loaded directly via
onnxruntime — no ``insightface`` package required.

Models are automatically downloaded on first run.
"""

from __future__ import annotations

import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path

import cv2
import imagehash
import numpy as np
import onnxruntime as ort
import PIL.Image

from faceorganizer.config import MIN_DETECTION_CONFIDENCE, MIN_FACE_SIZE
from faceorganizer.logging_config import get_logger
from faceorganizer.models import FaceInfo, PhotoInfo

log = get_logger("scanner.detector")

# Raise PIL's decompression bomb limit to handle large wallpapers/photos
PIL.Image.MAX_IMAGE_PIXELS = 250_000_000  # ~250MP

# Max dimension fed to the detector — larger images are downscaled proportionally.
_MAX_DETECT_DIM = 4096

# Register HEIC/HEIF support if pillow-heif is installed
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    log.debug("HEIC/HEIF support registered via pillow-heif")
except ImportError:
    log.debug("pillow-heif not installed — HEIC files will be skipped")

def _get_model_dir() -> Path:
    """Return the model directory, checking the PyInstaller bundle first."""
    import sys
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "models"
        if bundled.exists():
            return bundled
    return Path.home() / ".faceorganizer" / "models"


_MODEL_DIR = _get_model_dir()

# Public model URLs (no auth required)
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_ARCFACE_URL = (
    "https://huggingface.co/lithiumice/insightface/resolve/main/"
    "models/buffalo_l/w600k_r50.onnx"
)

# ── Global / thread-local singletons ──────────────────────────────────────
# YuNet detector is NOT thread-safe, so each thread gets its own instance.
_tls = threading.local()
# ArcFace session + lock: DirectML GPU execution is not thread-safe,
# so we serialize inference calls while still overlapping I/O in other threads.
_arcface_session: ort.InferenceSession | None = None
_arcface_lock = threading.Lock()
_arcface_init_lock = threading.Lock()
# Single lock shared by all model downloads — prevents multiple threads from
# trying to write the same file simultaneously on the first run.
_download_lock = threading.Lock()

# ArcFace standard alignment template (destination landmarks for 112×112 crop).
# Five points: left-eye, right-eye, nose-tip, left-mouth, right-mouth.
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _download_model(url: str, dest: Path) -> None:
    """Download a model file if it doesn't exist.

    Protected by _download_lock so that multiple worker threads started at
    the same time cannot race to download the same file simultaneously.
    """
    if dest.exists():
        log.debug("Model already cached: %s", dest.name)
        return
    with _download_lock:
        # Re-check inside the lock — another thread may have finished the
        # download while we were waiting.
        if dest.exists():
            log.debug("Model already cached (race avoided): %s", dest.name)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading model: %s ...", dest.name)
        urllib.request.urlretrieve(url, str(dest))
        log.info("Model saved to %s", dest)


_DETECTOR_CACHE_LIMIT = 8  # max distinct (width, height) sizes kept per thread


def _get_detector(width: int, height: int) -> cv2.FaceDetectorYN:
    """Load a per-thread YuNet face detection model, sized for the given image.

    Maintains a small per-thread dict cache keyed by (width, height) so that
    photos with varying but recurring dimensions (e.g. portrait vs. landscape
    from the same camera) reuse an already-initialised detector rather than
    recreating it.  The cache is bounded to _DETECTOR_CACHE_LIMIT entries;
    the oldest entry is evicted first (insertion order via Python dict).
    """
    model_path = _MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    _download_model(_YUNET_URL, model_path)

    cache = getattr(_tls, "detector_cache", None)
    if cache is None:
        _tls.detector_cache = {}
        cache = _tls.detector_cache

    key = (width, height)
    if key in cache:
        return cache[key]

    if not cache:
        log.info("Loading YuNet detection model (thread %s)", threading.current_thread().name)
    elif len(cache) >= _DETECTOR_CACHE_LIMIT:
        # Evict the oldest entry (insertion order)
        cache.pop(next(iter(cache)))

    detector = cv2.FaceDetectorYN.create(
        str(model_path), "", (width, height),
        score_threshold=MIN_DETECTION_CONFIDENCE,
        nms_threshold=0.2,
        top_k=100,
    )
    cache[key] = detector
    return detector


def _get_arcface_session() -> ort.InferenceSession:
    """Load the ArcFace w600k_r50 recognition model via onnxruntime."""
    global _arcface_session
    if _arcface_session is not None:
        return _arcface_session

    with _arcface_init_lock:
        # Double-check after acquiring lock
        if _arcface_session is not None:
            return _arcface_session

        model_path = _MODEL_DIR / "w600k_r50.onnx"
        _download_model(_ARCFACE_URL, model_path)
        log.info("Loading ArcFace recognition model (w600k_r50)")

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        # Prefer GPU providers: CUDA (NVIDIA), DirectML (any Windows GPU)
        if "DmlExecutionProvider" in available:
            providers.insert(0, "DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        # Read model bytes into memory to avoid Windows file-locking conflicts
        # when multiple worker processes load the model simultaneously.
        model_bytes = model_path.read_bytes()
        _arcface_session = ort.InferenceSession(model_bytes, providers=providers)
        active = _arcface_session.get_providers()
        log.info("ArcFace providers: %s", active)
        return _arcface_session


def _downscale_if_needed(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale an image if either dimension exceeds _MAX_DETECT_DIM.

    Returns (possibly_resized_image, scale_factor).
    """
    h, w = img.shape[:2]
    if w <= _MAX_DETECT_DIM and h <= _MAX_DETECT_DIM:
        return img, 1.0

    scale = min(_MAX_DETECT_DIM / w, _MAX_DETECT_DIM / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    log.debug("Downscaling %dx%d -> %dx%d for detection (scale=%.3f)", w, h, new_w, new_h, scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _align_face(img_bgr: np.ndarray, landmarks_5: np.ndarray) -> np.ndarray:
    """Align a face to the ArcFace canonical 112×112 template.

    Uses a similarity transform estimated from the five facial landmarks
    (left-eye, right-eye, nose, left-mouth, right-mouth) provided by YuNet.

    Args:
        img_bgr: Full-resolution BGR image.
        landmarks_5: (5, 2) array of landmark coordinates in image space.

    Returns:
        112×112 BGR aligned face crop.
    """
    # Estimate similarity transform: src landmarks → canonical ArcFace template
    M = cv2.estimateAffinePartial2D(landmarks_5, _ARCFACE_DST, method=cv2.LMEDS)[0]
    if M is None:
        # Fallback: simple crop + resize if transform estimation fails
        M = cv2.estimateAffinePartial2D(landmarks_5, _ARCFACE_DST, method=cv2.RANSAC)[0]
    if M is None:
        # Last resort: identity-ish crop around the face center
        cx, cy = landmarks_5.mean(axis=0)
        half = 56
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(img_bgr.shape[1], int(cx + half))
        y2 = min(img_bgr.shape[0], int(cy + half))
        crop = img_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, (112, 112))

    aligned = cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0)
    return aligned


def _extract_arcface_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    """Run ArcFace on a 112×112 aligned face crop.

    Returns an L2-normalized 512-dim float32 embedding.
    """
    session = _get_arcface_session()

    # ArcFace expects: (1, 3, 112, 112), float32, RGB, scaled to [-1, 1]
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32)
    img = (img / 127.5) - 1.0  # scale to [-1, 1]
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # add batch dim

    input_name = session.get_inputs()[0].name
    # DirectML GPU execution is not thread-safe — serialize inference calls.
    # Other threads continue doing I/O and preprocessing while one runs inference.
    with _arcface_lock:
        output = session.run(None, {input_name: img})[0]
    embedding = output.flatten().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def warmup_models(on_status: Callable[[str], None] | None = None) -> None:
    """Download (if needed) and initialise both models.

    Calls *on_status* with human-readable status strings so callers can
    surface progress during the one-time initialisation that would otherwise
    produce a silent "not responding" window.
    """
    def _report(msg: str) -> None:
        log.info(msg)
        if on_status:
            on_status(msg)

    yunet_path = _MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    if not yunet_path.exists():
        _report("Downloading YuNet detection model…")
    _download_model(_YUNET_URL, yunet_path)

    arcface_path = _MODEL_DIR / "w600k_r50.onnx"
    if not arcface_path.exists():
        _report("Downloading ArcFace recognition model…")
    elif _arcface_session is None:
        _report("Loading ArcFace recognition model…")
    _get_arcface_session()


def compute_phash(image_path: Path) -> str | None:
    """Compute a perceptual hash for an image file at rest.

    Used by the backfill path (faceorganizer.scanner.scan_runner.backfill_phashes)
    for photos scanned before duplicate detection existed. detect_faces() below
    computes its own hash inline from the PIL image it already has open, rather
    than calling this, to avoid opening the file twice during a real scan.
    """
    try:
        with PIL.Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception:
        log.warning("Failed to compute perceptual hash for %s", image_path.name)
        return None


def detect_faces(image_path: Path) -> tuple[PhotoInfo, list[FaceInfo]]:
    """Detect all faces in an image and extract their embeddings.

    Returns a (PhotoInfo, list[FaceInfo]) tuple.
    """
    # Load image with PIL for metadata (pillow-heif adds HEIC support)
    pil_img = PIL.Image.open(image_path)
    width, height = pil_img.size
    img_format = pil_img.format or image_path.suffix.lstrip(".").upper()

    # Extract EXIF date if available
    exif_date = None
    try:
        exif = pil_img.getexif()
        if exif:
            date_str = exif.get(36867) or exif.get(306)
            if date_str:
                from datetime import datetime

                for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        exif_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
    except Exception:
        pass

    photo = PhotoInfo(
        path=image_path.resolve(),
        file_size=image_path.stat().st_size,
        width=width,
        height=height,
        format=img_format,
        exif_date=exif_date,
    )

    # Perceptual hash for duplicate detection — computed here while pil_img is
    # still open, so it costs no extra file I/O.
    try:
        photo.phash = str(imagehash.phash(pil_img))
    except Exception:
        log.warning("Failed to compute perceptual hash for %s", image_path.name)

    # Convert to BGR numpy array for OpenCV
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    pil_img.close()

    # Downscale large images to prevent OOM in detector
    img_det, scale = _downscale_if_needed(img_bgr)
    det_h, det_w = img_det.shape[:2]

    # Detect faces with YuNet
    detector = _get_detector(det_w, det_h)
    _, raw_faces = detector.detect(img_det)

    if raw_faces is None or len(raw_faces) == 0:
        log.debug("No faces detected in %s", image_path.name)
        photo.num_faces = 0
        return photo, []

    log.debug("%d face(s) detected in %s", len(raw_faces), image_path.name)

    faces: list[FaceInfo] = []

    for face_raw in raw_faces:
        # Scale bbox and landmarks back to original image coordinates
        face_orig = face_raw.copy()
        if scale != 1.0:
            # x, y, w, h (indices 0-3) and landmark coords (indices 4-13)
            face_orig[:14] = face_orig[:14] / scale

        x = int(face_orig[0])
        y = int(face_orig[1])
        w = int(face_orig[2])
        h = int(face_orig[3])
        score = float(face_raw[-1])

        # Skip tiny detections — almost always false positives
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            log.debug("Skipping tiny detection %dx%d in %s", w, h, image_path.name)
            continue

        # Extract five landmarks from YuNet output:
        # indices 4,5 = right-eye; 6,7 = left-eye; 8,9 = nose;
        # 10,11 = right-mouth; 12,13 = left-mouth
        # ArcFace template order: left-eye, right-eye, nose, left-mouth, right-mouth
        landmarks = np.array(
            [
                [face_orig[6], face_orig[7]],   # left eye
                [face_orig[4], face_orig[5]],   # right eye
                [face_orig[8], face_orig[9]],   # nose tip
                [face_orig[12], face_orig[13]], # left mouth
                [face_orig[10], face_orig[11]], # right mouth
            ],
            dtype=np.float32,
        )

        # Align face to 112×112 ArcFace template and extract embedding
        aligned = _align_face(img_bgr, landmarks)
        embedding = _extract_arcface_embedding(aligned)

        faces.append(
            FaceInfo(
                photo_path=image_path.resolve(),
                bbox_x=max(0, x),
                bbox_y=max(0, y),
                bbox_w=w,
                bbox_h=h,
                embedding=embedding,
                detection_confidence=score,
            )
        )

    photo.num_faces = len(faces)
    return photo, faces
