"""Environment settings are validated, and bad input degrades to a default."""

from __future__ import annotations

import pytest

from config import (
    VALID_EXPERIENCE_LEVELS,
    VALID_SOURCE_CATEGORIES,
    Settings,
    load_settings,
)


class TestExperienceLevelSetting:
    """DEFAULT_EXPERIENCE_LEVEL follows the DEFAULT_SOURCE_CATEGORY pattern."""

    def test_the_workshop_default_is_the_student_level(self):
        assert Settings().default_experience_level == "internship"

    def test_every_real_level_plus_unknown_is_accepted(self):
        assert VALID_EXPERIENCE_LEVELS == {
            "internship",
            "entry",
            "mid",
            "senior",
            "staff_principal",
            "manager",
            "unknown",
        }

    @pytest.mark.parametrize("level", sorted(VALID_EXPERIENCE_LEVELS))
    def test_a_valid_level_is_read_from_the_environment(self, monkeypatch, level):
        monkeypatch.setenv("DEFAULT_EXPERIENCE_LEVEL", level)
        assert load_settings().default_experience_level == level

    @pytest.mark.parametrize(
        "raw",
        ["SENIOR", "  senior  ", "Staff_Principal"],
    )
    def test_case_and_padding_are_normalized(self, monkeypatch, raw):
        monkeypatch.setenv("DEFAULT_EXPERIENCE_LEVEL", raw)
        assert load_settings().default_experience_level == raw.strip().lower()

    @pytest.mark.parametrize(
        "garbage",
        ["", "   ", "principal engineer", "senior;drop", "42", "intern"],
    )
    def test_garbage_falls_back_to_the_default(self, monkeypatch, garbage):
        monkeypatch.setenv("DEFAULT_EXPERIENCE_LEVEL", garbage)
        assert load_settings().default_experience_level == "internship"

    def test_an_unset_variable_falls_back_to_the_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DEFAULT_EXPERIENCE_LEVEL", raising=False)
        settings = load_settings(env_file=tmp_path / "absent.env")
        assert settings.default_experience_level == "internship"

    def test_an_invalid_level_does_not_disturb_other_settings(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_EXPERIENCE_LEVEL", "nonsense")
        monkeypatch.setenv("DEFAULT_SOURCE_CATEGORY", "linkedin")
        settings = load_settings()
        assert settings.default_experience_level == "internship"
        assert settings.default_source_category == "linkedin"
        assert settings.default_source_category in VALID_SOURCE_CATEGORIES
