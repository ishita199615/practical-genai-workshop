"""Cougar Career Agent — Streamlit interface.

One page, one controlled graph, two human pauses. Rendering lives here; every
decision, score, and guardrail lives in the tools and the graph, so the screen
only shows what the agent actually did.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import streamlit as st
from langgraph.types import Command

from agent.graph import build_deps, build_graph
from config import load_settings
from models.ats import ATS_LONG_DISCLAIMER, BAND_LABELS, AtsAssessment, AtsRecommendation
from models.job import (
    EXPERIENCE_LEVEL_LABELS,
    FRESHNESS_LABELS,
    SOURCE_CATEGORY_LABELS,
    JobPosting,
)
from models.match import MatchResult
from tools.experience_level import levels_conflict

# --------------------------------------------------------------------------
# Option tables
# --------------------------------------------------------------------------

SOURCE_OPTIONS: list[tuple[str, str]] = [
    ("company_careers", "Direct Company Careers"),
    ("linkedin", "LinkedIn — public job pages only"),
    ("indeed", "Indeed — public job pages only"),
    ("google_jobs", "Google Jobs / Web"),
    ("all", "All Public Sources"),
]

FRESHNESS_OPTIONS: list[tuple[str, str]] = [
    ("last_hour", "Last 1 hour"),
    ("last_24_hours", "Last 24 hours"),
    ("last_3_days", "Last 3 days"),
    ("last_7_days", "Last 7 days"),
]

# Seniority order, most junior first, with an explicit opt-out at the end.
# "Any level" is the request-side reading of "unknown": the user is not
# filtering by seniority, which is not the same statement as a posting that
# never stated one ("Level not stated").
EXPERIENCE_OPTIONS: list[tuple[str, str]] = [
    ("internship", EXPERIENCE_LEVEL_LABELS["internship"]),
    ("entry", EXPERIENCE_LEVEL_LABELS["entry"]),
    ("mid", EXPERIENCE_LEVEL_LABELS["mid"]),
    ("senior", EXPERIENCE_LEVEL_LABELS["senior"]),
    ("staff_principal", EXPERIENCE_LEVEL_LABELS["staff_principal"]),
    ("manager", EXPERIENCE_LEVEL_LABELS["manager"]),
    ("unknown", "Any level"),
]

WORK_MODES = ["Any", "Remote", "Hybrid", "On-site"]

FRESHNESS_EVIDENCE_BADGE: dict[str, str] = {
    "exact_timestamp": "🕒 Exact timestamp",
    "date_only": "📅 Date only",
    "search_filter_only": "🔎 Search-filtered / unverified",
    "unavailable": "❔ Posting time unavailable",
}

BAND_ICON: dict[str, str] = {
    "strong": "🟢",
    "needs_targeted_changes": "🟠",
    "low": "🔴",
}

PRIORITY_ICON: dict[str, str] = {"high": "🔺", "medium": "🔸", "low": "▪️"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def is_safe_public_job_url(url: str | None) -> bool:
    """True when a URL may be rendered as a clickable link."""
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not parsed.username


def format_timestamp(value: datetime | None) -> str:
    """Format a timestamp for display, or say it is unavailable."""
    if value is None:
        return "not available"
    return value.strftime("%Y-%m-%d %H:%M UTC")


def experience_level_badge(job: JobPosting) -> str:
    """Return the badge for the level detected on a posting.

    A posting that never states a level reads "Level not stated". The level the
    user asked for is never shown here: asking for senior roles is not evidence
    that a result is senior, exactly as a freshness filter is not evidence of a
    posting time.
    """
    icon = "❔" if job.experience_level == "unknown" else "🎯"
    return f"{icon} {job.experience_level_label()}"


def experience_level_evidence(job: JobPosting) -> str:
    """Explain why a posting carries its detected level, without guessing."""
    if job.experience_level == "unknown":
        return (
            "this posting does not state a level, so none was assigned — the "
            "requested level filtered the search, it is not evidence about "
            "this posting"
        )
    return job.experience_level_evidence or "stated on the posting"


def label_for(options: list[tuple[str, str]], value: str) -> str:
    """Return the display label for an internal option value."""
    return next((label for key, label in options if key == value), value)


def value_index(options: list[tuple[str, str]], value: str) -> int:
    """Return the index of an internal option value, defaulting to zero."""
    keys = [key for key, _ in options]
    return keys.index(value) if value in keys else 0


def interrupt_payloads(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the pause payloads carried on a graph result."""
    return [item.value for item in result.get("__interrupt__", ())]


def pause_kind(result: dict[str, Any] | None) -> str | None:
    """Return the kind of pause the graph is currently waiting on."""
    if not result:
        return None
    payloads = interrupt_payloads(result)
    return payloads[0].get("kind") if payloads else None


def graph_config() -> dict[str, Any]:
    """LangGraph config for this Streamlit session's thread."""
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def get_graph():
    """Build the graph once per session and reuse it across reruns."""
    if "graph" not in st.session_state:
        deps = build_deps(st.session_state.settings)
        st.session_state.deps = deps
        st.session_state.graph = build_graph(deps)
    return st.session_state.graph


def reset_run() -> None:
    """Start a fresh graph thread, discarding any paused run."""
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:12]}"
    st.session_state.result = None
    st.session_state.stage = "idle"
    st.session_state.pop("graph", None)


def run_agent(params: dict[str, Any]) -> None:
    """Start a new run and stop at the job-selection pause."""
    reset_run()
    st.session_state.run_params = params
    graph = get_graph()
    with st.status("Running the career agent…", expanded=True) as status:
        status.write("Retrieving current public job pages…")
        result = graph.invoke(params, graph_config())
        status.update(label="Agent run complete", state="complete", expanded=False)
    st.session_state.result = result
    st.session_state.stage = (
        "selecting" if pause_kind(result) == "job_selection" else "finished"
    )


def resume_agent(value: Any, spinner: str) -> None:
    """Resume the paused graph with a human decision."""
    graph = get_graph()
    with st.spinner(spinner):
        result = graph.invoke(Command(resume=value), graph_config())
    st.session_state.result = result
    kind = pause_kind(result)
    if kind == "approval":
        st.session_state.stage = "reviewing"
    elif kind == "job_selection":
        st.session_state.stage = "selecting"
    else:
        st.session_state.stage = "finished"


# --------------------------------------------------------------------------
# Session bootstrap
# --------------------------------------------------------------------------

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
if "thread_id" not in st.session_state:
    reset_run()

settings = st.session_state.settings

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Cougar Career Agent")
st.caption("Fresh jobs. Explainable matching. ATS-ready, truthful tailoring.")

st.warning(
    (
        "You are running this against your own resume. It is not submitted "
        "anywhere, but with a Gemini key configured its text is sent to "
        "Google's API. ATS readiness is estimated using a transparent "
        "workshop rubric, not an employer's proprietary ATS."
    )
    if settings.using_custom_resume
    else (
        "Demo uses a fictional resume and public job data. It does not submit "
        "applications. ATS readiness is estimated using a transparent workshop "
        "rubric, not an employer's proprietary ATS."
    ),
    icon="⚠️",
)

for warning in settings.startup_warnings:
    st.info(warning, icon="ℹ️")

# --------------------------------------------------------------------------
# Section A — Search preferences
# --------------------------------------------------------------------------

st.subheader("A · Search preferences")

with st.form("search_preferences"):
    row_one = st.columns([2, 2, 1])
    role = row_one[0].text_input("Target role", value="Data Analyst Intern")
    location = row_one[1].text_input("Location", value="Houston, TX")
    work_mode = row_one[2].selectbox("Work mode", WORK_MODES, index=0)

    row_two = st.columns([2, 2, 2])
    query_category = row_two[0].selectbox(
        "Job source / query category",
        options=[key for key, _ in SOURCE_OPTIONS],
        index=value_index(SOURCE_OPTIONS, settings.default_source_category),
        format_func=lambda key: label_for(SOURCE_OPTIONS, key),
        help="This filters the search. The actual source is detected from each result's URL.",
    )
    freshness_window = row_two[1].selectbox(
        "Freshness",
        options=[key for key, _ in FRESHNESS_OPTIONS],
        index=value_index(FRESHNESS_OPTIONS, settings.default_freshness_window),
        format_func=lambda key: label_for(FRESHNESS_OPTIONS, key),
        help="A search filter narrows results; it is not proof of the posting time.",
    )
    experience_level = row_two[2].selectbox(
        "Experience level",
        options=[key for key, _ in EXPERIENCE_OPTIONS],
        index=value_index(EXPERIENCE_OPTIONS, settings.default_experience_level),
        format_func=lambda key: label_for(EXPERIENCE_OPTIONS, key),
        help=(
            "This shapes the search. Each result's level is read from the "
            "posting itself, so a posting that does not state one stays "
            "'Level not stated'."
        ),
    )

    row_three = st.columns([2, 2, 2])
    submitted = row_three[2].form_submit_button(
        "Run Career Agent", type="primary", use_container_width=True
    )

if submitted:
    run_agent(
        {
            "role": role,
            "location": location,
            "work_mode": work_mode,
            "query_category": query_category,
            "freshness_window": freshness_window,
            "experience_level": experience_level,
        }
    )
    st.rerun()

result: dict[str, Any] | None = st.session_state.get("result")

if result is None:
    st.info("Set your preferences and click **Run Career Agent** to begin.", icon="👆")
    st.stop()

params = st.session_state.run_params
st.markdown(
    f"**Searching:** {label_for(SOURCE_OPTIONS, params['query_category'])} · "
    f"{label_for(FRESHNESS_OPTIONS, params['freshness_window'])} · "
    f"{label_for(EXPERIENCE_OPTIONS, params.get('experience_level', 'unknown'))} · "
    f"{params['role']} · {params['location']}"
)

data_mode = result.get("data_mode", "live")
retrieved_at = result.get("retrieval_timestamp")
if data_mode == "live":
    st.success(
        f"🟢 LIVE PUBLIC RESULTS · Retrieved at: {format_timestamp(retrieved_at)}",
        icon="🟢",
    )
else:
    # "unavailable" would be false in standalone mode, where cached data is a
    # deliberate choice rather than a failure.
    reason = (
        "Standalone mode is on, so no network call was made."
        if settings.offline
        else "Live retrieval was unavailable."
    )
    st.warning(
        "🟡 CACHED DEMONSTRATION RESULTS · Originally retrieved at: "
        f"{format_timestamp(retrieved_at)}. {reason} "
        "These are synthetic demonstration records; their links are illustrative "
        "examples, not live employer postings.",
        icon="🟡",
    )

for message in result.get("errors", []):
    st.error(message, icon="🚫")

# Zero results for Last 1 hour: offer an explicit expansion, never a silent one.
if not result.get("filtered_jobs") and params["freshness_window"] == "last_hour":
    st.markdown("**No verifiable postings were found in the last hour.**")
    expand = st.columns(2)
    if expand[0].button("Expand to Last 24 hours", type="primary"):
        run_agent({**params, "freshness_window": "last_24_hours"})
        st.rerun()
    if expand[1].button("Retry with Direct Company Careers"):
        run_agent({**params, "query_category": "company_careers"})
        st.rerun()

# A level filter that emptied the screen gets the same explicit one-click
# recovery as an empty freshness window, offering exactly the levels that were
# actually retrieved. Clicking is the user's decision: a chosen level is never
# widened by the agent itself.
requested_level = params.get("experience_level", "unknown")
if not result.get("filtered_jobs") and requested_level != "unknown":
    off_level_levels = sorted(
        {
            job.experience_level
            for job in result.get("normalized_jobs", [])
            if levels_conflict(requested_level, job.experience_level)
        }
    )
    if off_level_levels:
        st.markdown(
            f"**Nothing matched {label_for(EXPERIENCE_OPTIONS, requested_level)}.** "
            "These levels were retrieved instead:"
        )
        recovery = st.columns(len(off_level_levels) + 1)
        for index, level_key in enumerate(off_level_levels):
            if recovery[index].button(
                f"Search {label_for(EXPERIENCE_OPTIONS, level_key)}",
                key=f"level_retry_{level_key}",
                use_container_width=True,
            ):
                run_agent({**params, "experience_level": level_key})
                st.rerun()
        if recovery[-1].button(
            "Search Any level",
            key="level_retry_any",
            type="primary",
            use_container_width=True,
        ):
            run_agent({**params, "experience_level": "unknown"})
            st.rerun()

# --------------------------------------------------------------------------
# Section B — Agent activity
# --------------------------------------------------------------------------

st.subheader("B · Agent activity")

with st.expander("What the agent did", expanded=True):
    for item in result.get("progress_events", []):
        icon = {"ok": "✓", "warn": "△", "error": "✗"}.get(item.get("status", "ok"), "•")
        detail = f" — {item['detail']}" if item.get("detail") else ""
        st.markdown(f"{icon} {item['label']}{detail}")
    for warning in result.get("warnings", []):
        st.caption(f"△ {warning}")

# --------------------------------------------------------------------------
# Section C — Top three job cards
# --------------------------------------------------------------------------


def render_source_evidence(job: JobPosting) -> None:
    """Render query category, detected source, freshness, and retrieval time."""
    cols = st.columns(4)
    cols[0].markdown(
        f"**Query category**  \n{SOURCE_CATEGORY_LABELS.get(job.query_category, job.query_category)}"
    )
    cols[1].markdown(
        f"**Actual source**  \n{SOURCE_CATEGORY_LABELS.get(job.source_category, job.source_category)}"
        f" ({job.source_label})"
    )
    cols[2].markdown(
        f"**Requested freshness**  \n{FRESHNESS_LABELS.get(job.freshness_window, job.freshness_window)}"
    )
    cols[3].markdown(f"**Retrieved at**  \n{format_timestamp(job.retrieved_at)}")

    posting_time = "not available"
    if job.posted_at is not None:
        posting_time = format_timestamp(job.posted_at)
    elif job.posting_date is not None:
        posting_time = job.posting_date.isoformat()

    age = (
        f" · about {job.posting_age_hours:.0f}h old"
        if job.posting_age_hours is not None
        else ""
    )
    st.caption(
        f"{FRESHNESS_EVIDENCE_BADGE[job.freshness_evidence]} · Source posting time: "
        f"{posting_time}{age} · {job.freshness_label()} · "
        f"{'🟢 Live' if job.data_mode == 'live' else '🟡 Cached'}"
    )


def render_level_badge(job: JobPosting) -> None:
    """Render the compact level line carried by every job card."""
    st.caption(
        f"{experience_level_badge(job)} · Requested level: "
        f"{label_for(EXPERIENCE_OPTIONS, job.requested_experience_level)}"
    )


def render_experience_level(job: JobPosting) -> None:
    """Render the requested level beside the level detected on the posting.

    Presented as a pair for the same reason as query category beside detected
    source: what was asked for shaped the search, and only what the page says
    describes the posting.
    """
    cols = st.columns(2)
    cols[0].markdown(
        "**Requested level**  \n"
        + label_for(EXPERIENCE_OPTIONS, job.requested_experience_level)
    )
    cols[1].markdown(f"**Detected level**  \n{experience_level_badge(job)}")
    st.caption(f"Level evidence: {experience_level_evidence(job)}")


def render_open_link(job: JobPosting, key_prefix: str) -> None:
    """Render the validated Open job posting link, or say why there is none."""
    if not is_safe_public_job_url(job.source_url):
        st.error(
            "This posting has no validated public link, so it cannot be opened.",
            icon="🚫",
        )
        return
    st.link_button(
        "Open job posting ↗",
        job.source_url,
        key=f"{key_prefix}_open_{job.job_id}",
        help=f"Open the original {job.source_label} posting in a new tab",
        use_container_width=True,
    )
    st.caption(job.source_url)
    if job.apply_url and is_safe_public_job_url(job.apply_url):
        st.link_button(
            "Apply on source ↗",
            job.apply_url,
            key=f"{key_prefix}_apply_{job.job_id}",
            use_container_width=True,
        )


def render_job_description(job: JobPosting, key_prefix: str) -> None:
    """Render the cleaned full description and everything derived from it."""
    st.markdown("**Full job description as retrieved**")
    st.markdown(
        f"<div style='max-height:340px;overflow-y:auto;padding:0.5rem;"
        f"border:1px solid rgba(128,128,128,0.3);border-radius:0.4rem;'>"
        f"{job.description.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if job.responsibilities:
        st.markdown("**Responsibilities**")
        for item in job.responsibilities:
            st.markdown(f"- {item}")
    detail = st.columns(2)
    detail[0].markdown(
        "**Required skills**  \n" + (", ".join(job.required_skills) or "not stated")
    )
    detail[1].markdown(
        "**Preferred skills**  \n" + (", ".join(job.preferred_skills) or "not stated")
    )
    requirements = st.columns(2)
    requirements[0].markdown(
        "**Education requirement**  \n" + (job.education_requirement or "not stated")
    )
    requirements[1].markdown(
        "**Minimum experience**  \n"
        + (
            f"{job.minimum_experience_years:g} year(s)"
            if job.minimum_experience_years is not None
            else "not stated"
        )
    )
    st.divider()
    render_experience_level(job)
    render_source_evidence(job)
    if job.original_source_url and job.original_source_url != job.source_url:
        st.caption(f"Original result URL: {job.original_source_url}")
    render_open_link(job, f"{key_prefix}_desc")


st.subheader("C · Top three matches")

jobs_by_id: dict[str, JobPosting] = {
    job.job_id: job for job in result.get("filtered_jobs", [])
}
ranked: list[MatchResult] = result.get("ranked_matches", [])

if not ranked:
    st.info("No ranked jobs are available for this search.", icon="ℹ️")
else:
    for rank, match in enumerate(ranked, start=1):
        job = jobs_by_id.get(match.job_id)
        if job is None:
            continue
        with st.container(border=True):
            header = st.columns([5, 2])
            header[0].markdown(
                f"### #{rank} · {job.title}\n**{job.company}** — "
                f"{job.location or 'location not stated'} · {job.work_mode}"
            )
            header[1].metric("Demo Job Match Score", f"{match.total_score}/100")

            st.markdown(job.description_excerpt)

            # Surfaced on the card so the gaps are visible before selecting,
            # rather than only after the ATS assessment runs.
            skill_bits: list[str] = []
            if match.matched_skills:
                skill_bits.append("✓ " + ", ".join(match.matched_skills))
            if match.missing_skills:
                skill_bits.append("△ Missing: " + ", ".join(match.missing_skills))
            if skill_bits:
                st.caption(" · ".join(skill_bits))

            render_level_badge(job)
            render_source_evidence(job)

            with st.expander("View job description"):
                render_job_description(job, f"card{rank}")

            actions = st.columns([3, 2])
            with actions[0]:
                render_open_link(job, f"card{rank}")
            with actions[1]:
                if st.session_state.stage == "selecting":
                    if st.button(
                        "Select this job",
                        key=f"select_{job.job_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        resume_agent(
                            job.job_id,
                            "Scoring ATS readiness, drafting, and validating…",
                        )
                        st.rerun()
                elif result.get("selected_job_id") == job.job_id:
                    st.success("Selected", icon="✅")

if st.session_state.stage == "selecting":
    st.info("Select one job to continue.", icon="👉")
    st.stop()

selected_job: JobPosting | None = result.get("selected_job")
if selected_job is None:
    st.stop()

selected_match = next(
    (match for match in ranked if match.job_id == selected_job.job_id), None
)

# --------------------------------------------------------------------------
# Section D — Job details and match explanation
# --------------------------------------------------------------------------

st.subheader("D · Selected job and match explanation")

with st.container(border=True):
    st.markdown(f"### {selected_job.title} — {selected_job.company}")
    detail_cols = st.columns([3, 2])

    with detail_cols[0]:
        render_level_badge(selected_job)
        with st.expander("Job description used for scoring", expanded=False):
            render_job_description(selected_job, "selected")
        render_open_link(selected_job, "selected_main")

    with detail_cols[1]:
        if selected_match is not None:
            st.metric("Demo Job Match Score", f"{selected_match.total_score}/100")
            st.caption("Calculated in Python. Gemini explains it; it cannot change it.")
            st.markdown(
                f"- Required-skill coverage (45%): **{selected_match.skill_score}/100**\n"
                f"- Text similarity (20%): **{selected_match.similarity_score}/100**\n"
                f"- Role alignment (15%): **{selected_match.role_score}/100**\n"
                f"- Experience alignment (10%): **{selected_match.experience_score}/100**\n"
                f"- Location and work mode (10%): **{selected_match.preference_score}/100**"
            )
            if selected_match.matched_skills:
                st.markdown(
                    "**Strong matches**  \n"
                    + "  \n".join(f"✓ {skill}" for skill in selected_match.matched_skills)
                )
            if selected_match.missing_skills:
                st.markdown(
                    "**Missing or unclear**  \n"
                    + "  \n".join(f"△ {skill}" for skill in selected_match.missing_skills)
                )
            for concern in selected_match.concerns:
                st.markdown(f"! {concern}")
            if selected_match.explanation:
                st.info(selected_match.explanation, icon="💬")

# --------------------------------------------------------------------------
# Section E — ATS readiness
# --------------------------------------------------------------------------

st.subheader("E · Demo ATS Readiness and what to change first")

assessment: AtsAssessment | None = result.get("ats_assessment")
projected: AtsAssessment | None = result.get("projected_ats_assessment")


def render_components(item: AtsAssessment) -> None:
    """Render the six-component ATS breakdown."""
    rows = [
        ("Required keyword and skill coverage (40%)", item.keyword_score),
        ("Required qualification alignment (20%)", item.qualification_score),
        ("Evidence and specificity (15%)", item.evidence_score),
        ("Standard section completeness (10%)", item.section_score),
        ("ATS-safe structure and parseability (10%)", item.structure_score),
        ("Contact and application essentials (5%)", item.contact_score),
    ]
    for name, score in rows:
        st.markdown(f"{name}: **{score}/100**")
        st.progress(score / 100)


def render_recommendation(rec: AtsRecommendation) -> None:
    """Render one recommendation with its safety label and evidence."""
    icon = PRIORITY_ICON.get(rec.priority, "•")
    with st.container(border=True):
        st.markdown(
            f"{icon} **{rec.priority.upper()} · {rec.target_section}** "
            f"`{rec.recommendation_id}`"
        )
        if rec.current_text:
            st.caption(f"Current: {rec.current_text}")
        st.markdown(rec.recommended_change)
        st.caption(f"Why it matters: {rec.reason}")
        meta = st.columns(3)
        meta[0].markdown(
            f"**Evidence**  \n{', '.join(rec.evidence_resume_ids) or 'none'}"
        )
        meta[1].markdown(
            f"**Safe to apply**  \n{'Yes' if rec.safe_to_apply else 'No'}"
        )
        meta[2].markdown(f"**Expected effect**  \n{rec.projected_effect.title()}")


if assessment is None:
    st.info("ATS readiness was not scored for this run.", icon="ℹ️")
else:
    with st.container(border=True):
        score_cols = st.columns([2, 3])
        score_cols[0].metric(
            "Demo ATS Readiness Score",
            f"{assessment.total_score}/100",
            help="Deterministic Python rubric",
        )
        score_cols[0].markdown(
            f"### {BAND_ICON[assessment.band]} {BAND_LABELS[assessment.band]}"
        )
        with score_cols[1]:
            render_components(assessment)
        st.caption(ATS_LONG_DISCLAIMER)

    st.markdown("#### What to change first")
    recommendations: list[AtsRecommendation] = result.get("ats_recommendations", [])
    safe = [rec for rec in recommendations if rec.safe_to_apply]
    gaps = [rec for rec in recommendations if not rec.safe_to_apply]

    if not recommendations:
        st.success("No changes are recommended for this posting.", icon="✅")

    change_cols = st.columns(2)
    with change_cols[0]:
        st.markdown("**Safe, evidence-backed changes**")
        if not safe:
            st.caption("No safe wording changes are available for this posting.")
        for rec in safe:
            render_recommendation(rec)
    with change_cols[1]:
        st.markdown("**Unsupported job gaps — not added to the resume**")
        if not gaps:
            st.caption("No unsupported gaps were detected.")
        for rec in gaps:
            render_recommendation(rec)

    if assessment.unsupported_job_gaps:
        st.error(
            "Job requirement(s) with no evidence in the master resume: "
            + ", ".join(assessment.unsupported_job_gaps)
            + ". Decision: these were **not** added to the resume. "
            "Recommendation: treat them as learning gaps.",
            icon="🛡️",
        )

    if projected is not None:
        st.markdown("#### Original versus projected")
        compare = st.columns(2)
        compare[0].metric(
            "Original estimated score",
            f"{assessment.total_score}/100",
            help=BAND_LABELS[assessment.band],
        )
        compare[1].metric(
            "Projected under the same rubric",
            f"{projected.total_score}/100",
            delta=projected.total_score - assessment.total_score,
            help=BAND_LABELS[projected.band],
        )
        st.caption(
            "Projected under this demo rubric after the proposed safe changes. "
            "It is not a promise that a real ATS score will improve."
        )
        if projected.total_score < assessment.total_score:
            st.warning(
                "The projected score is lower than the original score. These "
                "changes are not an improvement under this rubric.",
                icon="⚠️",
            )
        with st.expander("Projected component breakdown"):
            render_components(projected)

# --------------------------------------------------------------------------
# Section F — Resume patch
# --------------------------------------------------------------------------

st.subheader("F · Resume patch")

application = result.get("tailored_application")
resume = result.get("resume")

if application is None or resume is None:
    st.info("No tailored application was produced.", icon="ℹ️")
else:
    with st.container(border=True):
        st.markdown("**Professional summary**")
        summary_cols = st.columns(2)
        summary_cols[0].markdown("_Original_")
        summary_cols[0].info(resume.professional_summary)
        summary_cols[1].markdown("_Revised_")
        summary_cols[1].success(application.revised_summary)

        st.markdown("**Experience bullets**")
        for bullet in application.revised_bullets:
            bullet_cols = st.columns(2)
            bullet_cols[0].markdown(f"_Original_ · `{bullet.source_bullet_id}`")
            bullet_cols[0].info(bullet.original_text)
            bullet_cols[1].markdown("_Revised_")
            bullet_cols[1].success(bullet.revised_text)

        st.markdown("**Skills (reordered only — nothing added)**")
        order_cols = st.columns(2)
        order_cols[0].markdown("_Original order_  \n" + ", ".join(resume.skills))
        order_cols[1].markdown(
            "_Revised order_  \n" + ", ".join(application.reordered_skills)
        )

        meta_cols = st.columns(3)
        meta_cols[0].markdown(
            "**Applied ATS recommendations**  \n"
            + (", ".join(application.applied_ats_recommendation_ids) or "none")
        )
        meta_cols[1].markdown(
            "**Missing requirements**  \n"
            + (", ".join(application.missing_requirements) or "none")
        )
        meta_cols[2].markdown(
            "**Unsupported gaps not applied**  \n"
            + (", ".join(application.unsupported_ats_gaps_not_applied) or "none")
        )

        st.markdown("**Cover letter**")
        st.markdown(
            f"<div style='padding:0.75rem;border:1px solid rgba(128,128,128,0.3);"
            f"border-radius:0.4rem;white-space:pre-wrap;'>"
            f"{application.cover_letter}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{len(application.cover_letter.split())} words")

# --------------------------------------------------------------------------
# Section G — Validation
# --------------------------------------------------------------------------

st.subheader("G · Truthfulness validation")

report = result.get("validation_report")

if report is None:
    st.info("Validation has not run yet.", icon="ℹ️")
else:
    with st.container(border=True):
        if report.passed:
            st.success("All truthfulness checks passed.", icon="✅")
        else:
            st.warning(
                "Some checks need a human decision. Details are below.", icon="⚠️"
            )
        for check in report.deterministic_checks:
            icon = "✓" if check.get("passed") else "✗"
            st.markdown(f"{icon} **{check['name']}** — {check['detail']}")

        if report.unsupported_claims:
            st.markdown("**Unsupported claims**")
            for review in report.unsupported_claims:
                st.error(f"{review.claim}\n\n_{review.reason}_", icon="🚫")
        if report.unclear_claims:
            st.markdown("**Unclear claims**")
            for review in report.unclear_claims:
                st.warning(f"{review.claim}\n\n_{review.reason}_", icon="❓")

# --------------------------------------------------------------------------
# Section H — Human approval
# --------------------------------------------------------------------------

st.subheader("H · Human approval")

stage = st.session_state.stage
decision = result.get("approval_decision")

if stage == "reviewing":
    st.info(
        "**Application package ready.**  \nNo application has been submitted.  \n"
        "Waiting for human approval.",
        icon="⏸️",
    )
    feedback = st.text_input(
        "Feedback for Request Changes (optional)", key="approval_feedback_input"
    )
    approval_cols = st.columns(3)
    if approval_cols[0].button("Approve", type="primary", use_container_width=True):
        resume_agent({"decision": "approve"}, "Exporting the application package…")
        st.rerun()
    if approval_cols[1].button("Request Changes", use_container_width=True):
        resume_agent(
            {"decision": "request_changes", "feedback": feedback},
            "Revising and re-validating…",
        )
        st.rerun()
    if approval_cols[2].button("Reject", use_container_width=True):
        resume_agent({"decision": "reject"}, "Stopping the run…")
        st.rerun()

elif decision == "approve":
    st.success("Approved. Application package exported.", icon="✅")
    st.markdown("**No application was submitted.**")
    files = result.get("output_files", [])
    download_cols = st.columns(max(len(files), 1))
    for index, path_text in enumerate(files):
        path = Path(path_text)
        if not path.exists():
            continue
        download_cols[index].download_button(
            f"Download {path.suffix.lstrip('.').upper()}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/markdown" if path.suffix == ".md" else "application/json",
            use_container_width=True,
            key=f"download_{path.name}",
        )
    st.caption("Exported to: " + ", ".join(files))

elif decision == "reject":
    st.error(
        "Rejected. Nothing was exported and no application was submitted. "
        "The review screen above is preserved for discussion.",
        icon="🛑",
    )

# --------------------------------------------------------------------------
# Architecture reveal
# --------------------------------------------------------------------------

with st.expander("How this agent is built"):
    st.markdown(
        """
| Component | Responsibility |
| --- | --- |
| **Firecrawl** | Retrieves current public job pages |
| **Python** | Calculates the Job Match Score and the ATS Readiness Score |
| **Gemini** | Extracts, explains, recommends, and drafts safe changes |
| **Validator** | Checks every revised claim against the master resume |
| **Human** | Approves, requests changes, or rejects |

The model is not the agent. The agent is the model plus tools, state,
validation, decisions, and permissions.
"""
    )
    st.caption(
        f"Mode: {settings.demo_mode} · Model: {settings.gemini_model} · "
        f"Thread: {st.session_state.thread_id}"
    )

if st.button("Start a new run"):
    reset_run()
    st.rerun()
