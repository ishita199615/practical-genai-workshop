"""Cougar Career Agent — the same search, answered from employers' own boards.

The Full Demo answers a search by retrieving public job pages and reading them.
This page answers the *same* search a second way: by reading the JSON boards
Workday, Greenhouse, Lever, and Ashby already publish. One agent, one query,
two routes to it.

The difference is what each route can prove. A scraped page states its location
and posting date in prose, if at all, so both have to be read out and verified.
A board states them as fields. That is why this route can group by city and
country with confidence, and why it needs no language model — which also means
it costs nothing and returns the same answer twice.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import load_settings
from models.job import FRESHNESS_LABELS
from models.query import SearchQuery
from tools.ats_boards import (
    AtsPosting,
    _filter,
    fetch_all,
    group_by_country,
    load_employers,
)
from tools.job_filter import requested_location_terms

REGISTRY_PATH = Path("data/employers.json")
CACHE_TTL_SECONDS = 600

st.title("🎓 Cougar Career Agent")
st.subheader("🌍 Browse by location")
st.caption(
    "The same search as the Full Demo, answered from employers' own job-board "
    "APIs — no web search, no scraping, no API key, and no AI model. Location "
    "and posting date are as the employer published them."
)

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
settings = st.session_state.settings

if "query" not in st.session_state:
    st.session_state.query = SearchQuery(
        query_category=settings.default_source_category,
        freshness_window=settings.default_freshness_window,
        experience_level=settings.default_experience_level,
    )
query: SearchQuery = st.session_state.query

employers = load_employers(REGISTRY_PATH)
if not employers:
    st.error(
        f"No employers loaded. Add rows to `{REGISTRY_PATH}` — each needs a "
        "`name`, an `ats` (workday, greenhouse, lever, or ashby), and a `slug`.",
        icon="🚫",
    )
    st.stop()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_postings(search_text: str) -> list[AtsPosting]:
    """Read every board once and cache it, so filtering does not refetch."""
    return fetch_all(load_employers(REGISTRY_PATH), search_text=search_text)


# --------------------------------------------------------------------------
# The agent's current search, editable here and shared with the Full Demo
# --------------------------------------------------------------------------

with st.form("board_search"):
    row = st.columns([3, 3, 2])
    role = row[0].text_input(
        "Target role",
        value=query.role,
        help="Matched against the job title. Each word acts as an alternative.",
    )
    location = row[1].text_input(
        "Location",
        value=query.location,
        help="The city and its commuting towns. Leave blank for every city.",
    )
    freshness_window = row[2].selectbox(
        "Freshness",
        options=list(FRESHNESS_LABELS),
        index=list(FRESHNESS_LABELS).index(query.freshness_window)
        if query.freshness_window in FRESHNESS_LABELS
        else 1,
        format_func=lambda key: FRESHNESS_LABELS[key],
    )
    controls = st.columns([2, 2, 2])
    anywhere = controls[0].checkbox(
        "Search every location",
        value=False,
        help="Ignore the location above and show every city the boards return.",
    )
    submitted = controls[2].form_submit_button(
        "Run on employer boards", type="primary", use_container_width=True
    )

if submitted:
    # One agent, one search: what is set here is what the Full Demo opens with.
    st.session_state.query = SearchQuery(
        role=role,
        location=location,
        work_mode=query.work_mode,
        query_category=query.query_category,
        freshness_window=freshness_window,
        experience_level=query.experience_level,
    )
    st.session_state.board_run = True
    st.rerun()

st.caption(
    f"Reading {len(employers)} employer board(s): "
    + ", ".join(sorted({employer.name for employer in employers}))
)

if not st.session_state.get("board_run"):
    st.info("Set the search and select **Run on employer boards**.", icon="👆")
    st.stop()

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

st.markdown(
    f"**Searching:** employer job boards · {FRESHNESS_LABELS[query.freshness_window]} · "
    f"{query.role} · {query.location if not anywhere else 'every location'}"
)

with st.spinner("Reading employer job boards…"):
    pool = load_postings(query.role)

# The same commuting-town list the Full Demo filters on, so "Houston" means the
# same place on both screens.
terms = [] if anywhere else requested_location_terms(query.location)
matches = _filter(pool, terms, query.role_pattern(), query.max_age_days)
tree = group_by_country(matches)

summary = st.columns(4)
summary[0].metric("Postings", len(matches))
summary[1].metric("Countries", sum(1 for c in tree if "not stated" not in c))
summary[2].metric("Cities", sum(len(cities) for cities in tree.values()))
summary[3].metric("Boards read", len(employers))

if not matches:
    where = "every location" if anywhere else query.location
    st.info(
        f"**No recent job postings** for {query.role} in {where} in the "
        f"{FRESHNESS_LABELS[query.freshness_window].lower()}, across "
        f"{len(employers)} employer board(s). Widen the freshness, tick "
        "**Search every location**, or add employers to the registry.",
        icon="ℹ️",
    )
    st.stop()

st.divider()

for country, cities in tree.items():
    total = sum(len(jobs) for jobs in cities.values())
    unstated = "not stated" in country.lower()
    with st.expander(
        f"{'❔' if unstated else '📍'}  {country} — {total} posting(s)",
        expanded=not unstated,
    ):
        if unstated:
            st.caption(
                "These boards put a work mode, or nothing, in the location "
                "field. No country is claimed for them."
            )
        for city, jobs in cities.items():
            st.markdown(f"**{city}** · {len(jobs)}")
            for job in jobs:
                left, right = st.columns([5, 2])
                if job.url:
                    left.markdown(f"[{job.title}]({job.url}) — {job.company}")
                else:
                    left.markdown(f"{job.title} — {job.company}")
                age = job.age_days
                if age is None:
                    right.caption(f"date not stated · {job.ats}")
                elif age < 1:
                    right.caption(f"today · {job.ats}")
                else:
                    right.caption(f"{age:.0f}d ago · {job.ats}")
            st.markdown("")
