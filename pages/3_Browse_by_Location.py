"""Browse live postings straight from employers' own applicant-tracking systems.

This page does not search the web. It reads the JSON boards Workday, Greenhouse,
Lever, and Ashby already publish, so every posting arrives with the location and
publish date the employer stated, rather than ones inferred from page text.

Nothing here calls a language model, which is why it costs nothing to run and
returns the same answer twice.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from tools.ats_boards import (
    AtsPosting,
    _filter,
    fetch_all,
    group_by_country,
    load_employers,
)

REGISTRY_PATH = Path("data/employers.json")
CACHE_TTL_SECONDS = 600

AGE_OPTIONS: list[tuple[int | None, str]] = [
    (1, "Last 24 hours"),
    (3, "Last 3 days"),
    (7, "Last 7 days"),
    (30, "Last 30 days"),
    (None, "Any age"),
]

st.title("🌍 Browse by location")
st.caption(
    "Read directly from employers' own job-board APIs — no web search, no "
    "scraping, no API key, and no AI model. Location and posting date are as "
    "the employer published them."
)

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


with st.form("browse"):
    row = st.columns([3, 2, 2])
    role_text = row[0].text_input(
        "Role keywords",
        value="analyst",
        help="Matched against the job title. Several words act as alternatives.",
    )
    max_age = row[1].selectbox(
        "Posted within",
        options=[days for days, _ in AGE_OPTIONS],
        index=3,
        format_func=lambda days: dict(AGE_OPTIONS)[days],
    )
    city_text = row[2].text_input(
        "City contains", value="", help="Optional. Leave blank for every city."
    )
    submitted = st.form_submit_button(
        "Fetch postings", type="primary", use_container_width=False
    )

st.caption(f"Reading {len(employers)} employer board(s): " + ", ".join(
    sorted({employer.name for employer in employers})
))

if not submitted and "browse_done" not in st.session_state:
    st.info("Choose your filters and select **Fetch postings**.", icon="👆")
    st.stop()

if submitted:
    st.session_state["browse_done"] = True

terms = [word for word in re.split(r"[,\s]+", role_text.strip()) if word]
pattern = (
    re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
    if terms
    else re.compile(".")
)

with st.spinner("Reading employer job boards…"):
    pool = load_postings(role_text.strip())

city_terms = [part.strip() for part in city_text.split(",") if part.strip()]
matches = _filter(pool, city_terms, pattern, max_age)
tree = group_by_country(matches)

summary = st.columns(4)
summary[0].metric("Postings", len(matches))
summary[1].metric("Countries", sum(1 for c in tree if "not stated" not in c))
summary[2].metric("Cities", sum(len(cities) for cities in tree.values()))
summary[3].metric("Boards read", len(employers))

if not matches:
    st.info(
        f"**No recent job postings** matching “{role_text}”"
        + (f" in “{city_text}”" if city_text else "")
        + f" from these {len(employers)} employer board(s). "
        "Widen the age, change the keywords, or add employers to the registry.",
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
