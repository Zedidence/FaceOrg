"""Tests for faceorganizer/app_settings.py (the desktop app's global settings store).

_SETTINGS_DIR/_SETTINGS_FILE are monkeypatched to a temp path in every test so
these never touch the real ~/.faceorganizer/app_settings.json on the machine
running the tests.
"""

from __future__ import annotations

from faceorganizer import app_settings as mod
from faceorganizer.app_settings import AppSettings
from faceorganizer.config import DEFAULT_CLUSTER_THRESHOLD, MIN_DETECTION_CONFIDENCE, MIN_FACE_SIZE


def _use_tmp_settings_file(monkeypatch, tmp_path):
    settings_dir = tmp_path / ".faceorganizer"
    monkeypatch.setattr(mod, "_SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", settings_dir / "app_settings.json")


class TestLoad:
    def test_defaults_when_no_file_exists(self, monkeypatch, tmp_path):
        _use_tmp_settings_file(monkeypatch, tmp_path)
        settings = AppSettings.load()
        assert settings.detection_confidence == MIN_DETECTION_CONFIDENCE
        assert settings.min_face_size == MIN_FACE_SIZE
        assert settings.cluster_threshold == DEFAULT_CLUSTER_THRESHOLD
        assert settings.recent_folders == []

    def test_defaults_on_corrupted_file(self, monkeypatch, tmp_path):
        _use_tmp_settings_file(monkeypatch, tmp_path)
        mod._SETTINGS_DIR.mkdir(parents=True)
        mod._SETTINGS_FILE.write_text("{not valid json", encoding="utf-8")

        settings = AppSettings.load()
        assert settings == AppSettings()

    def test_ignores_unknown_keys(self, monkeypatch, tmp_path):
        _use_tmp_settings_file(monkeypatch, tmp_path)
        mod._SETTINGS_DIR.mkdir(parents=True)
        mod._SETTINGS_FILE.write_text(
            '{"theme": "light", "some_future_field": 123}', encoding="utf-8"
        )

        settings = AppSettings.load()
        assert settings.theme == "light"
        assert not hasattr(settings, "some_future_field")


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_values(self, monkeypatch, tmp_path):
        _use_tmp_settings_file(monkeypatch, tmp_path)
        original = AppSettings(
            detection_confidence=0.8, min_face_size=60, worker_count=4,
            cluster_threshold=0.4, theme="light", recent_folders=["/a", "/b"],
        )
        original.save()

        loaded = AppSettings.load()
        assert loaded == original

    def test_save_creates_settings_dir(self, monkeypatch, tmp_path):
        _use_tmp_settings_file(monkeypatch, tmp_path)
        assert not mod._SETTINGS_DIR.exists()
        AppSettings().save()
        assert mod._SETTINGS_FILE.exists()


class TestRecentFolders:
    def test_add_recent_folder_dedups_and_moves_to_front(self):
        settings = AppSettings(recent_folders=["/a", "/b", "/c"])
        settings.add_recent_folder("/b")
        assert settings.recent_folders == ["/b", "/a", "/c"]

    def test_add_recent_folder_caps_at_ten(self):
        settings = AppSettings(recent_folders=[f"/f{i}" for i in range(10)])
        settings.add_recent_folder("/new")
        assert len(settings.recent_folders) == 10
        assert settings.recent_folders[0] == "/new"
        assert "/f9" not in settings.recent_folders


class TestEffectiveWorkers:
    def test_uses_recommended_when_unset(self):
        settings = AppSettings(worker_count=None)
        assert settings.effective_workers(6) == 6

    def test_uses_override_when_set(self):
        settings = AppSettings(worker_count=2)
        assert settings.effective_workers(6) == 2
