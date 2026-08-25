"""Prompt templates stored as Markdown and rendered with literal replacement."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a prompt template by file stem, for example ``extract_job``."""
    path = PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **values: str) -> str:
    """Render a prompt template.

    Uses literal ``{TOKEN}`` replacement rather than ``str.format`` so that
    braces inside scraped job text can never break rendering.
    """
    template = load_prompt(name)
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template
