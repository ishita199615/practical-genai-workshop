"""Runtime configuration loaded from the environment.

Secrets are read here and nowhere else. API keys are never logged, never put
into Streamlit session state, and never written into exported files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

#: The synthetic profile the workshop demo ships with. Anything else means the
#: operator pointed the app at a resume of their own, which changes what the
#: interface is allowed to claim about the data it is handling.
DEFAULT_RESUME_FILE = "data/sample_resume.json"

VALID_DEMO_MODES = {"live", "cached", "auto"}
VALID_SOURCE_CATEGORIES = {
    "all",
    "linkedin",
    "indeed",
    "google_jobs",
    "company_careers",
}
VALID_FRESHNESS_WINDOWS = {
    "last_hour",
    "last_24_hours",
    "last_3_days",
    "last_7_days",
}
# "unknown" is a real choice here: it means the user is not filtering by
# seniority at all, which is the behaviour the demo had before levels existed.
VALID_EXPERIENCE_LEVELS = {
    "internship",
    "entry",
    "mid",
    "senior",
    "staff_principal",
    "manager",
    "unknown",
}


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on bad input."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Accepts the spellings people actually type in a .env file.
    """
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _choice_env(name: str, default: str, allowed: set[str]) -> str:
    """Read an environment variable constrained to a fixed set of values."""
    raw = (os.getenv(name) or "").strip().lower()
    return raw if raw in allowed else default


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    firecrawl_api_key: str = ""
    firecrawl_base_url: str = "https://api.firecrawl.dev"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_models: str = ""
    demo_mode: str = "auto"
    default_source_category: str = "company_careers"
    default_freshness_window: str = "last_24_hours"
    default_experience_level: str = "internship"
    max_job_results: int = 8
    max_job_description_chars: int = 20000
    ats_recommendation_threshold: int = 80
    search_timeout_seconds: int = 10
    cache_file: str = "data/cached_jobs.json"
    output_dir: str = "output"
    resume_file: str = DEFAULT_RESUME_FILE
    offline: bool = False
    startup_warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_firecrawl(self) -> bool:
        """True when a Firecrawl API key is configured."""
        return bool(self.firecrawl_api_key)

    @property
    def has_gemini(self) -> bool:
        """True when a Gemini API key is configured."""
        return bool(self.gemini_api_key)

    @property
    def model_chain(self) -> list[str]:
        """The ordered model-routing chain.

        Falls back to a single-model chain built from ``GEMINI_MODEL`` so the
        project still works with only the settings the spec defines.
        """
        from services.router_client import parse_model_chain  # noqa: PLC0415

        chain = parse_model_chain(self.llm_models)
        if chain:
            return chain
        return [f"gemini/{self.gemini_model}"] if self.gemini_model else []

    @property
    def cache_path(self) -> Path:
        """Absolute path to the cached demonstration data."""
        return _resolve(self.cache_file)

    @property
    def output_path(self) -> Path:
        """Absolute path to the export directory."""
        return _resolve(self.output_dir)

    @property
    def resume_path(self) -> Path:
        """Absolute path to the resume the app should load."""
        return _resolve(self.resume_file)

    @property
    def using_custom_resume(self) -> bool:
        """True when the operator supplied a resume of their own.

        The interface must not describe a real person's data as fictional, so
        every user-facing label and the exported package key off this.
        """
        return self.resume_path != _resolve(DEFAULT_RESUME_FILE)

    @property
    def resume_descriptor(self) -> str:
        """Short, accurate description of whose resume is loaded."""
        return (
            "your own resume"
            if self.using_custom_resume
            else "fictional demonstration profile"
        )


def _resolve(value: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load and validate settings from ``.env`` and the process environment.

    Missing keys degrade the demo rather than crashing it: the app reports the
    limitation and falls back to clearly labelled cached data.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    warnings: list[str] = []
    firecrawl_key = (os.getenv("FIRECRAWL_API_KEY") or "").strip()
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    demo_mode = _choice_env("DEMO_MODE", "auto", VALID_DEMO_MODES)
    offline = _bool_env("OFFLINE", False)

    if offline:
        # One switch for a guaranteed-deterministic run. The keys stay in .env
        # untouched so the operator can flip back without re-pasting them; they
        # are simply dropped here, which makes every downstream component treat
        # the network as unavailable without needing to know about this flag.
        firecrawl_key = ""
        gemini_key = ""
        demo_mode = "cached"
        warnings.append(
            "OFFLINE is on. No network call is made: saved postings are used "
            "and every step runs its deterministic path. Both scores are "
            "unchanged, because neither ever depended on the model."
        )

    if not offline and not firecrawl_key and demo_mode != "cached":
        warnings.append(
            "FIRECRAWL_API_KEY is not set. Live retrieval is unavailable; "
            "the demo will use clearly labelled cached results."
        )
    if not offline and not gemini_key:
        warnings.append(
            "GEMINI_API_KEY is not set. Extraction, explanation, drafting, and "
            "claim review will use deterministic offline fallbacks."
        )

    resume_file = (os.getenv("RESUME_FILE") or DEFAULT_RESUME_FILE).strip()
    if not (_resolve(resume_file)).exists():
        warnings.append(
            f"RESUME_FILE points at '{resume_file}', which does not exist. "
            f"Falling back to the sample profile at {DEFAULT_RESUME_FILE}."
        )
        resume_file = DEFAULT_RESUME_FILE
    elif resume_file != DEFAULT_RESUME_FILE and gemini_key and not offline:
        warnings.append(
            "A personal resume is loaded and a Gemini key is configured, so "
            "resume text will be sent to Google's API. Set DEMO_MODE=cached "
            "and remove GEMINI_API_KEY to keep everything on this machine."
        )

    return Settings(
        firecrawl_api_key=firecrawl_key,
        firecrawl_base_url=(
            os.getenv("FIRECRAWL_BASE_URL") or "https://api.firecrawl.dev"
        ).strip(),
        gemini_api_key=gemini_key,
        gemini_model=(os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip(),
        llm_models=(os.getenv("LLM_MODELS") or "").strip(),
        demo_mode=demo_mode,
        default_source_category=_choice_env(
            "DEFAULT_SOURCE_CATEGORY", "company_careers", VALID_SOURCE_CATEGORIES
        ),
        default_freshness_window=_choice_env(
            "DEFAULT_FRESHNESS_WINDOW", "last_24_hours", VALID_FRESHNESS_WINDOWS
        ),
        default_experience_level=_choice_env(
            "DEFAULT_EXPERIENCE_LEVEL", "internship", VALID_EXPERIENCE_LEVELS
        ),
        max_job_results=max(1, min(_int_env("MAX_JOB_RESULTS", 8), 8)),
        max_job_description_chars=_int_env("MAX_JOB_DESCRIPTION_CHARS", 20000),
        ats_recommendation_threshold=_int_env("ATS_RECOMMENDATION_THRESHOLD", 80),
        search_timeout_seconds=_int_env("SEARCH_TIMEOUT_SECONDS", 10),
        cache_file=(os.getenv("CACHE_FILE") or "data/cached_jobs.json").strip(),
        output_dir=(os.getenv("OUTPUT_DIR") or "output").strip(),
        resume_file=resume_file,
        offline=offline,
        startup_warnings=tuple(warnings),
    )
