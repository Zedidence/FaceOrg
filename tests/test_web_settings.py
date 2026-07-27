"""Tests for faceorganizer/web/settings.py (the web app's per-folder settings store)."""

from __future__ import annotations

from faceorganizer.config import MIN_DETECTION_CONFIDENCE, MIN_FACE_SIZE
from faceorganizer.web.settings import Settings


class TestLoad:
    def test_defaults_when_no_file_exists(self, tmp_path):
        settings = Settings.load(tmp_path)
        assert settings.detection_confidence == MIN_DETECTION_CONFIDENCE
        assert settings.min_face_size == MIN_FACE_SIZE
        assert settings.default_workers is None

    def test_defaults_on_corrupted_file(self, tmp_path):
        (tmp_path / "settings.json").write_text("{not valid json", encoding="utf-8")
        settings = Settings.load(tmp_path)
        assert settings == Settings()

    def test_ignores_unknown_keys(self, tmp_path):
        (tmp_path / "settings.json").write_text(
            '{"detection_confidence": 0.7, "some_future_field": 123}', encoding="utf-8"
        )
        settings = Settings.load(tmp_path)
        assert settings.detection_confidence == 0.7
        assert not hasattr(settings, "some_future_field")


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_values(self, tmp_path):
        original = Settings(detection_confidence=0.75, min_face_size=50, default_workers=3)
        original.save(tmp_path)

        loaded = Settings.load(tmp_path)
        assert loaded == original

    def test_save_creates_data_dir(self, tmp_path):
        data_dir = tmp_path / "nested" / ".faceorganizer"
        assert not data_dir.exists()
        Settings().save(data_dir)
        assert (data_dir / "settings.json").exists()
