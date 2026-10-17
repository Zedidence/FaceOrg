"""Tests for landmark handling in faceorganizer/scanner/face_detector.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from faceorganizer.scanner import face_detector as fd

# OpenCV's sample portrait; present on machines with the opencv samples package.
_SAMPLE_FACE = Path("/usr/share/opencv4/samples/data/lena.jpg")


def test_yunet_landmarks_keep_template_order():
    # YuNet row: bbox, then right-eye, left-eye, nose, right-mouth, left-mouth
    # (subject's sides), then score.  The subject's right eye sits on the
    # image's left, which is where ArcFace template point 0 is.
    row = np.array(
        [0, 0, 100, 100,
         30, 40,   # subject's right eye (image left)
         70, 40,   # subject's left eye (image right)
         50, 60,   # nose
         35, 80,   # subject's right mouth corner (image left)
         65, 80,   # subject's left mouth corner (image right)
         0.99],
        dtype=np.float32,
    )
    lm = fd._yunet_landmarks(row)
    assert lm.shape == (5, 2)
    np.testing.assert_array_equal(lm[:, 0], [30, 70, 50, 35, 65])
    # Same left-to-right layout as the template
    assert (lm[0, 0] < lm[1, 0]) == (fd._ARCFACE_DST[0, 0] < fd._ARCFACE_DST[1, 0])
    assert (lm[3, 0] < lm[4, 0]) == (fd._ARCFACE_DST[3, 0] < fd._ARCFACE_DST[4, 0])


@pytest.mark.skipif(
    not _SAMPLE_FACE.exists()
    or not (fd._MODEL_DIR / "face_detection_yunet_2023mar.onnx").exists()
    or not (fd._MODEL_DIR / "w600k_r50.onnx").exists(),
    reason="needs the sample face image and downloaded models",
)
def test_same_face_embeddings_match(tmp_path):
    """A resized copy of the same face must embed close to the original."""
    copy = tmp_path / "copy.jpg"
    with PIL.Image.open(_SAMPLE_FACE) as im:
        im.resize((im.width * 3 // 2, im.height * 3 // 2)).save(copy, quality=85)

    _, faces_a = fd.detect_faces(_SAMPLE_FACE)
    _, faces_b = fd.detect_faces(copy)
    assert faces_a and faces_b
    sim = float(np.dot(faces_a[0].embedding, faces_b[0].embedding))
    assert sim > 0.8
