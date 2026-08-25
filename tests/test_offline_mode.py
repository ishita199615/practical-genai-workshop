"""Tests for the single-switch standalone mode.

``OFFLINE=true`` has to be trustworthy: if it makes even one network call, the
guarantee it offers a presenter on a bad conference connection is worthless.
"""

from __future__ import annotations

import pytest

from config import load_settings


def _settings(monkeypatch, tmp_path, **env):
    """Load settings from a clean environment plus the given overrides."""
    for key in (
        "OFFLINE",
        "DEMO_MODE",
        "GEMINI_API_KEY",
        "FIRECRAWL_API_KEY",
        "LLM_MODELS",
        "RESUME_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return load_settings(env_file=tmp_path / "absent.env")


def test_offline_defaults_to_off(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    assert settings.offline is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "True"])
def test_offline_accepts_the_spellings_people_type(monkeypatch, tmp_path, value):
    assert _settings(monkeypatch, tmp_path, OFFLINE=value).offline is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
def test_offline_stays_off_for_falsey_values(monkeypatch, tmp_path, value):
    assert _settings(monkeypatch, tmp_path, OFFLINE=value).offline is False


def test_offline_drops_keys_even_when_they_are_present(monkeypatch, tmp_path):
    """The whole point: keys stay in .env but are not used."""
    settings = _settings(
        monkeypatch,
        tmp_path,
        OFFLINE="true",
        GEMINI_API_KEY="real-key",
        FIRECRAWL_API_KEY="fc-real-key",
    )

    assert settings.has_gemini is False
    assert settings.has_firecrawl is False
    assert settings.gemini_api_key == ""
    assert settings.firecrawl_api_key == ""


def test_offline_forces_cached_mode_over_an_explicit_live_setting(
    monkeypatch, tmp_path
):
    settings = _settings(
        monkeypatch, tmp_path, OFFLINE="true", DEMO_MODE="live",
        FIRECRAWL_API_KEY="fc-real-key",
    )
    assert settings.demo_mode == "cached"


def test_offline_says_so_once_and_suppresses_missing_key_noise(
    monkeypatch, tmp_path
):
    """Offline is a choice, so 'your key is missing' would be misleading."""
    settings = _settings(monkeypatch, tmp_path, OFFLINE="true")
    warnings = settings.startup_warnings

    assert any("OFFLINE is on" in w for w in warnings)
    assert not any("FIRECRAWL_API_KEY is not set" in w for w in warnings)
    assert not any("GEMINI_API_KEY is not set" in w for w in warnings)


def test_without_offline_missing_keys_still_warn(monkeypatch, tmp_path):
    """The existing warnings must survive for people not using the switch."""
    settings = _settings(monkeypatch, tmp_path, DEMO_MODE="auto")
    assert any("FIRECRAWL_API_KEY is not set" in w for w in settings.startup_warnings)
    assert any("GEMINI_API_KEY is not set" in w for w in settings.startup_warnings)


def test_offline_suppresses_the_personal_resume_privacy_warning(
    monkeypatch, tmp_path
):
    """Nothing leaves the machine, so the 'sent to Google' warning is wrong."""
    import json
    from pathlib import Path

    sample = Path(__file__).resolve().parent.parent / "data" / "sample_resume.json"
    custom = tmp_path / "mine.json"
    custom.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")

    settings = _settings(
        monkeypatch,
        tmp_path,
        OFFLINE="true",
        GEMINI_API_KEY="real-key",
        RESUME_FILE=str(custom),
    )

    assert settings.using_custom_resume is True
    assert not any("sent to Google" in w for w in settings.startup_warnings)


def test_offline_builds_deps_with_no_usable_clients(monkeypatch, tmp_path):
    """An offline run must not be able to reach either service."""
    from agent.graph import build_deps

    settings = _settings(
        monkeypatch,
        tmp_path,
        OFFLINE="true",
        GEMINI_API_KEY="real-key",
        FIRECRAWL_API_KEY="fc-real-key",
    )
    deps = build_deps(settings)

    assert deps.search_adapter.available is False
    assert getattr(deps.llm, "available", False) is False
