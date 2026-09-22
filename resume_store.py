"""The resume the app is currently working with, and where it came from.

The agent scores against one resume. It can come from the file the settings
point at, or from the editor on the Resume page. Both pages need the same
answer to two questions — which resume is active, and whether it belongs to a
real person — so both ask here.

Whose resume it is drives what the interface is allowed to say. A profile the
user typed must never be described as fictional, so :func:`is_custom` is what
the banners key off rather than the file-based setting alone.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from config import Settings
from models.resume import ResumeProfile

#: Session keys. The resume itself and the account of where it came from are
#: stored together so they can never disagree.
_RESUME_KEY = "active_resume"
_SOURCE_KEY = "active_resume_source"

#: Source values, in order of how much they are the user's own.
SAMPLE = "sample"
EDITED = "edited"
UPLOADED = "uploaded"


def load_from_file(settings: Settings) -> ResumeProfile:
    """Read and validate the resume the settings point at."""
    payload = json.loads(settings.resume_path.read_text(encoding="utf-8"))
    return ResumeProfile.model_validate(payload)


def active_resume(settings: Settings) -> ResumeProfile:
    """The resume this session is working with, loading the file on first use."""
    if _RESUME_KEY not in st.session_state:
        st.session_state[_RESUME_KEY] = load_from_file(settings)
        st.session_state[_SOURCE_KEY] = SAMPLE
    return st.session_state[_RESUME_KEY]


def source(settings: Settings) -> str:
    """Where the active resume came from."""
    active_resume(settings)
    return st.session_state.get(_SOURCE_KEY, SAMPLE)


def set_resume(resume: ResumeProfile, origin: str) -> None:
    """Make ``resume`` the one the agent will use."""
    st.session_state[_RESUME_KEY] = resume
    st.session_state[_SOURCE_KEY] = origin


def reset(settings: Settings) -> ResumeProfile:
    """Discard edits and go back to the resume on disk."""
    resume = load_from_file(settings)
    set_resume(resume, SAMPLE)
    return resume


def is_custom(settings: Settings) -> bool:
    """True when the active resume is not the shipped fictional profile.

    A resume typed or uploaded here is the user's own even though the file the
    settings name has not changed, which is why this is not simply
    ``settings.using_custom_resume``.
    """
    return settings.using_custom_resume or source(settings) != SAMPLE


def descriptor(settings: Settings) -> str:
    """Short, accurate description of whose resume is loaded."""
    return "your own resume" if is_custom(settings) else "fictional demonstration profile"


def to_payload(resume: ResumeProfile) -> dict[str, Any]:
    """The resume as the JSON the app reads back."""
    return resume.model_dump(mode="json", exclude_none=False)


def renumber(payload: dict[str, Any]) -> dict[str, Any]:
    """Give every entry and bullet a unique, positional id.

    The claim validator maps a revised bullet back to its source through these
    ids, so a bullet added in the editor needs one. Numbering by position keeps
    them unique without asking the user to invent identifiers.
    """
    for index, entry in enumerate(payload.get("experience") or [], start=1):
        entry["id"] = f"experience_{index}"
        for bullet_index, bullet in enumerate(entry.get("bullets") or [], start=1):
            bullet["id"] = f"experience_{index}_bullet_{bullet_index}"
    for index, project in enumerate(payload.get("projects") or [], start=1):
        project["id"] = f"project_{index}"
        for bullet_index, bullet in enumerate(project.get("bullets") or [], start=1):
            bullet["id"] = f"project_{index}_bullet_{bullet_index}"
    for index, entry in enumerate(payload.get("education") or [], start=1):
        entry["id"] = f"education_{index}"
    return payload
