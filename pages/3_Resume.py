"""Cougar Career Agent — the resume the agent scores against.

Until now the profile came only from a JSON file on disk, which meant editing
a file and restarting to try your own. This page is that file, as a form.

What you save here is what the agent uses: the run carries the resume with it,
so nothing is written to disk and nothing you type leaves the machine unless a
model key is configured. Bullet ids are regenerated on save because the claim
validator traces every rewritten line back to the bullet it came from.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import streamlit as st
from pydantic import ValidationError

import resume_store
from config import load_settings
from models.resume import ResumeProfile

# --------------------------------------------------------------------------
# Session bootstrap
# --------------------------------------------------------------------------

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
settings = st.session_state.settings

#: Prefixes for every widget this page owns, so the editor can be reloaded
#: wholesale when the underlying resume is replaced.
EDITOR_PREFIXES = ("res_", "exp_", "prj_", "edu_")


def new_uid() -> str:
    """A stable handle for one repeatable block.

    Widgets are keyed by this rather than by position, so removing the first
    role does not shuffle every later role's text into the wrong box.
    """
    return uuid.uuid4().hex[:8]


def load_into_editor(resume: ResumeProfile) -> None:
    """Replace the editor's contents with one resume.

    Only safe before this run draws its widgets — Streamlit refuses a write to
    a widget's state after the widget exists — so a button queues a ``load``
    in ``pending_actions`` rather than calling this directly.
    """
    for key in [k for k in st.session_state if k.startswith(EDITOR_PREFIXES)]:
        del st.session_state[key]

    payload = resume_store.to_payload(resume)
    st.session_state.res_candidate_id = payload.get("candidate_id", "")
    st.session_state.res_document_features = payload.get("document_features", {})
    st.session_state.res_name = payload.get("name", "")
    st.session_state.res_email = payload.get("email", "")
    st.session_state.res_phone = payload.get("phone") or ""
    st.session_state.res_location = payload.get("location", "")
    st.session_state.res_summary = payload.get("professional_summary", "")
    st.session_state.res_roles = list(payload.get("target_roles") or [])
    st.session_state.res_skills = list(payload.get("skills") or [])

    st.session_state.edu_uids = []
    for entry in payload.get("education") or []:
        uid = new_uid()
        st.session_state.edu_uids.append(uid)
        st.session_state[f"edu_degree_{uid}"] = entry.get("degree", "")
        st.session_state[f"edu_school_{uid}"] = entry.get("institution", "")
        st.session_state[f"edu_grad_{uid}"] = entry.get("graduation", "")

    st.session_state.exp_uids = []
    for entry in payload.get("experience") or []:
        uid = new_uid()
        st.session_state.exp_uids.append(uid)
        st.session_state[f"exp_title_{uid}"] = entry.get("title", "")
        st.session_state[f"exp_org_{uid}"] = entry.get("organization", "")
        st.session_state[f"exp_dates_{uid}"] = entry.get("dates") or ""
        st.session_state[f"exp_bullets_{uid}"] = "\n".join(
            bullet.get("text", "") for bullet in entry.get("bullets") or []
        )

    st.session_state.prj_uids = []
    for project in payload.get("projects") or []:
        uid = new_uid()
        st.session_state.prj_uids.append(uid)
        st.session_state[f"prj_name_{uid}"] = project.get("name", "")
        st.session_state[f"prj_bullets_{uid}"] = "\n".join(
            bullet.get("text", "") for bullet in project.get("bullets") or []
        )


#: Structural edits collected while the page renders and applied once it has
#: finished. Rerunning the moment a button is clicked would leave every widget
#: below that button undrawn, and Streamlit discards the state of a keyed
#: widget it did not draw — removing the first role wiped the second one's text.
pending_actions: list[tuple[str, Any]] = []


def lines(text: str) -> list[str]:
    """Split a textarea into one entry per non-empty line."""
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def collect() -> dict[str, Any]:
    """Read the whole editor back into a resume payload."""
    payload: dict[str, Any] = {
        "candidate_id": st.session_state.get("res_candidate_id") or "ui_candidate_001",
        "name": st.session_state.get("res_name", "").strip(),
        "email": st.session_state.get("res_email", "").strip(),
        "phone": st.session_state.get("res_phone", "").strip() or None,
        "location": st.session_state.get("res_location", "").strip(),
        "professional_summary": st.session_state.get("res_summary", "").strip(),
        "target_roles": list(st.session_state.get("res_roles") or []),
        "skills": list(st.session_state.get("res_skills") or []),
        "document_features": dict(st.session_state.get("res_document_features") or {}),
        "education": [
            {
                "degree": st.session_state.get(f"edu_degree_{uid}", "").strip(),
                "institution": st.session_state.get(f"edu_school_{uid}", "").strip(),
                "graduation": st.session_state.get(f"edu_grad_{uid}", "").strip(),
            }
            for uid in st.session_state.get("edu_uids", [])
        ],
        "experience": [
            {
                "title": st.session_state.get(f"exp_title_{uid}", "").strip(),
                "organization": st.session_state.get(f"exp_org_{uid}", "").strip(),
                "dates": st.session_state.get(f"exp_dates_{uid}", "").strip() or None,
                "bullets": [
                    {"text": text}
                    for text in lines(st.session_state.get(f"exp_bullets_{uid}", ""))
                ],
            }
            for uid in st.session_state.get("exp_uids", [])
        ],
        "projects": [
            {
                "name": st.session_state.get(f"prj_name_{uid}", "").strip(),
                "bullets": [
                    {"text": text}
                    for text in lines(st.session_state.get(f"prj_bullets_{uid}", ""))
                ],
            }
            for uid in st.session_state.get("prj_uids", [])
        ],
    }
    return resume_store.renumber(payload)


def blank_reasons(payload: dict[str, Any]) -> list[str]:
    """Problems the schema would accept but a reader would not.

    A required string passes validation while empty, and a role with no
    bullets gives the tailoring step nothing it is allowed to quote.
    """
    problems: list[str] = []
    for field, label in (
        ("name", "Name"),
        ("email", "Email"),
        ("location", "Location"),
        ("professional_summary", "Professional summary"),
    ):
        if not payload.get(field):
            problems.append(f"{label} is empty.")
    if not payload.get("skills"):
        problems.append("Add at least one skill — matching scores against them.")
    for entry in payload["experience"]:
        who = entry["title"] or entry["organization"] or "A role"
        if not entry["title"] or not entry["organization"]:
            problems.append(f"{who} needs both a title and an organization.")
        elif not entry["bullets"]:
            problems.append(f"{who} has no bullets, so nothing can be tailored from it.")
    for project in payload["projects"]:
        if not project["name"]:
            problems.append("A project has no name.")
    return problems


# A queued load runs before any widget exists, which is the only moment the
# editor's state can be rewritten wholesale.
if "pending_editor_load" in st.session_state:
    load_into_editor(st.session_state.pop("pending_editor_load"))
elif "exp_uids" not in st.session_state:
    load_into_editor(resume_store.active_resume(settings))

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title(":material/description: Resume")
st.caption(
    "The profile every posting is scored against, and the only source a "
    "tailored draft is allowed to quote."
)

if resume_store.is_custom(settings):
    st.warning(
        "This is your own resume. It is not submitted anywhere, but with a "
        "model key configured its text is sent to that provider for matching "
        "and drafting.",
        icon=":material/warning:",
    )
else:
    st.info(
        f"Loaded from `{settings.resume_file}` — the fictional demonstration "
        "profile. Edit anything below and save to run the agent against your own.",
        icon=":material/info:",
    )

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"), icon=":material/check_circle:")

# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------

st.subheader("Basics")
with st.container(border=True):
    row = st.columns(2)
    row[0].text_input("Full name", key="res_name")
    row[1].text_input("Email", key="res_email")
    row = st.columns(2)
    row[0].text_input("Phone", key="res_phone", placeholder="Optional")
    row[1].text_input(
        "Location",
        key="res_location",
        help="Where you are based. Where to search is set separately on the demo page.",
    )
    st.text_area(
        "Professional summary",
        key="res_summary",
        height=90,
        help="Two or three lines. This is quoted when a draft is tailored.",
    )
    st.multiselect(
        "Target roles",
        options=list(st.session_state.res_roles),
        key="res_roles",
        accept_new_options=True,
        placeholder="Type a role and press enter",
    )
    st.multiselect(
        "Skills",
        options=list(st.session_state.res_skills),
        key="res_skills",
        accept_new_options=True,
        placeholder="Type a skill and press enter",
        help="Matching counts these against each posting, so spell them as employers do.",
    )

# --------------------------------------------------------------------------
# Education
# --------------------------------------------------------------------------

st.subheader("Education")
for uid in list(st.session_state.edu_uids):
    with st.container(border=True):
        row = st.columns([3, 3, 2, 1], vertical_alignment="bottom")
        row[0].text_input("Degree", key=f"edu_degree_{uid}")
        row[1].text_input("Institution", key=f"edu_school_{uid}")
        row[2].text_input("Graduation", key=f"edu_grad_{uid}")
        if row[3].button("Remove", icon=":material/delete:", key=f"edu_del_{uid}"):
            pending_actions.append(("remove", ("edu_uids", uid)))

if st.button("Add education", icon=":material/add:", key="edu_add"):
    pending_actions.append(("add", "edu_uids"))

# --------------------------------------------------------------------------
# Experience
# --------------------------------------------------------------------------

st.subheader("Experience")
st.caption("One bullet per line. These are the only claims a tailored draft may build on.")

for position, uid in enumerate(list(st.session_state.exp_uids), start=1):
    title = st.session_state.get(f"exp_title_{uid}") or "New role"
    org = st.session_state.get(f"exp_org_{uid}")
    heading = f"{position}. {title}" + (f" — {org}" if org else "")
    with st.expander(heading, expanded=True):
        row = st.columns([3, 3, 2])
        row[0].text_input("Title", key=f"exp_title_{uid}")
        row[1].text_input("Organization", key=f"exp_org_{uid}")
        row[2].text_input("Dates", key=f"exp_dates_{uid}", placeholder="Optional")
        st.text_area("Bullets", key=f"exp_bullets_{uid}", height=120)
        if st.button("Remove this role", icon=":material/delete:", key=f"exp_del_{uid}"):
            pending_actions.append(("remove", ("exp_uids", uid)))

if st.button("Add a role", icon=":material/add:", key="exp_add"):
    pending_actions.append(("add", "exp_uids"))

# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------

st.subheader("Projects")
for position, uid in enumerate(list(st.session_state.prj_uids), start=1):
    name = st.session_state.get(f"prj_name_{uid}") or "New project"
    with st.expander(f"{position}. {name}", expanded=True):
        st.text_input("Project name", key=f"prj_name_{uid}")
        st.text_area("Bullets", key=f"prj_bullets_{uid}", height=100)
        if st.button("Remove this project", icon=":material/delete:", key=f"prj_del_{uid}"):
            pending_actions.append(("remove", ("prj_uids", uid)))

if st.button("Add a project", icon=":material/add:", key="prj_add"):
    pending_actions.append(("add", "prj_uids"))

# --------------------------------------------------------------------------
# Save, reset, and transfer
# --------------------------------------------------------------------------

st.divider()

draft = collect()
problems = blank_reasons(draft)

actions = st.columns([2, 2, 3])

if actions[0].button(
    "Save resume", icon=":material/save:", type="primary", disabled=bool(problems)
):
    try:
        resume = ResumeProfile.model_validate(draft)
    except ValidationError as exc:
        st.error(f"This resume could not be validated: {exc.error_count()} problem(s).")
        st.code(str(exc))
    else:
        resume_store.set_resume(resume, resume_store.EDITED)
        st.session_state.pop("result", None)
        # Redrawn from the top, so the banner above stops describing a real
        # person's resume as the fictional sample.
        st.session_state.flash = (
            f"Saved. The next run scores against {resume.name}'s profile."
        )
        pending_actions.append(("rerun", None))

if actions[1].button("Reset to the file", icon=":material/restart_alt:"):
    pending_actions.append(("load", resume_store.reset(settings)))

actions[2].download_button(
    "Download as JSON",
    data=json.dumps(draft, indent=2),
    file_name="my_resume.json",
    mime="application/json",
    icon=":material/download:",
    help="Save it beside the sample and point RESUME_FILE at it to load it on startup.",
)

if problems:
    st.warning(
        "Finish these before saving:\n\n"
        + "\n".join(f"- {problem}" for problem in problems),
        icon=":material/edit_note:",
    )

with st.expander("Load a resume from JSON"):
    st.caption(
        "Accepts the same shape as the download above, or "
        f"`{settings.resume_file}`. Ids are regenerated on save."
    )
    upload = st.file_uploader("Resume JSON", type="json", label_visibility="collapsed")
    if upload is not None and st.button("Load into the editor", icon=":material/upload:"):
        try:
            resume = ResumeProfile.model_validate(
                json.loads(upload.getvalue().decode("utf-8"))
            )
        except (ValueError, ValidationError) as exc:
            st.error(f"That file could not be read as a resume: {exc}")
        else:
            resume_store.set_resume(resume, resume_store.UPLOADED)
            pending_actions.append(("load", resume))

# --------------------------------------------------------------------------
# Deferred edits
#
# Every widget above has now been drawn, so its state is safe. Only one action
# is applied per run: each arrives from a single click.
# --------------------------------------------------------------------------

if pending_actions:
    action, argument = pending_actions[0]
    if action == "add":
        st.session_state[argument].append(new_uid())
    elif action == "remove":
        bucket, uid = argument
        st.session_state[bucket].remove(uid)
    elif action == "load":
        st.session_state.pending_editor_load = argument
    # Anything the agent produced was scored against the resume as it was.
    if action == "load":
        st.session_state.pop("result", None)
    st.rerun()
