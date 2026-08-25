"""Threading the requested experience level through the graph.

The level shapes what is *searched*, what is *filtered out*, and which titles
the ranking rewards. It never becomes a claim about a posting: what a result
says about its own seniority is detected separately, and a run that never sets
a level behaves exactly as it did before the control existed.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.graph import build_graph
from agent.nodes import (
    AgentDeps,
    build_search_query,
    filter_and_deduplicate_jobs,
    normalize_jobs_node,
    requested_experience_level,
    score_jobs,
    search_current_jobs,
    searched_roles,
)
from agent.state import CareerAgentState
from config import Settings
from models.job import EXPERIENCE_LEVEL_LABELS
from services.llm_interface import NullLLMClient
from tests.conftest import FIXED_NOW, make_job, make_raw
from tools.firecrawl_search import FirecrawlSearchAdapter
from tools.job_scorer import rank_jobs

REAL_LEVELS = ("internship", "entry", "mid", "senior", "staff_principal", "manager")

BASE_STATE: dict[str, Any] = {
    "role": "Data Analyst Intern",
    "location": "Houston, TX",
    "work_mode": "Any",
    "query_category": "company_careers",
    "freshness_window": "last_24_hours",
}


def state_with(**overrides: Any) -> dict[str, Any]:
    """Build a run state from the workshop defaults."""
    payload = dict(BASE_STATE)
    payload.update(overrides)
    return payload


class CapturingClient:
    """A Firecrawl stand-in that records the query it was asked to run."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def search(self, query: str, **kwargs: Any):
        self.calls.append((query, kwargs))
        return {"web": []}


@pytest.fixture
def deps() -> AgentDeps:
    """Cached-mode dependencies with no model and no network."""
    return AgentDeps(
        settings=Settings(demo_mode="cached"),
        search_adapter=FirecrawlSearchAdapter(),
        llm=NullLLMClient(),
        now=lambda: FIXED_NOW,
    )


def live_deps(client: CapturingClient) -> AgentDeps:
    """Dependencies whose live search is answered by a capturing fake."""
    return AgentDeps(
        settings=Settings(demo_mode="auto", firecrawl_api_key="test-key"),
        search_adapter=FirecrawlSearchAdapter(
            api_key="test-key", client_factory=lambda key, url: client
        ),
        llm=NullLLMClient(),
        now=lambda: FIXED_NOW,
    )


class TestStateCarriesTheLevel:
    """The graph state has a place for the requested level, defaulting safely."""

    def test_the_state_declares_the_field(self):
        assert "experience_level" in CareerAgentState.__annotations__

    def test_a_state_without_a_level_reads_as_unknown(self):
        assert requested_experience_level({}) == "unknown"

    @pytest.mark.parametrize("level", REAL_LEVELS)
    def test_a_selected_level_is_read_back(self, level):
        assert requested_experience_level({"experience_level": level}) == level

    @pytest.mark.parametrize("value", [None, "", "wizard", "principal engineer", "3"])
    def test_an_unrecognised_value_falls_back_to_unknown(self, value):
        assert requested_experience_level({"experience_level": value}) == "unknown"

    @pytest.mark.parametrize("value", [["senior"], {"level": "senior"}, {"senior"}, 42])
    def test_an_unhashable_value_falls_back_instead_of_raising(self, value):
        # The value crosses a checkpointer boundary, so it may be anything.
        assert requested_experience_level({"experience_level": value}) == "unknown"
        assert searched_roles({"role": "Data Analyst", "experience_level": value}) == [
            "Data Analyst"
        ]


class TestSearchQueryNode:
    """The level reaches the query builder and the activity log."""

    def test_no_level_reproduces_todays_query(self, deps):
        without = build_search_query(state_with(), deps)
        explicit = build_search_query(state_with(experience_level="unknown"), deps)
        assert without["search_query"] == explicit["search_query"]
        assert '"data analyst intern"' in without["search_query"]

    def test_a_selected_level_reshapes_the_query(self, deps):
        update = build_search_query(state_with(experience_level="senior"), deps)
        assert '"senior data analyst"' in update["search_query"]
        # The level word the user typed is stripped, so the two cannot disagree.
        assert "intern" not in update["search_query"]

    def test_the_event_names_the_selected_level(self, deps):
        update = build_search_query(state_with(experience_level="senior"), deps)
        label = update["progress_events"][0]["label"]
        assert label == (
            "Search query created for Direct Company Careers · Last 24 hours · Senior"
        )

    def test_the_event_omits_the_level_when_none_is_selected(self, deps):
        update = build_search_query(state_with(), deps)
        assert update["progress_events"][0]["label"] == (
            "Search query created for Direct Company Careers · Last 24 hours"
        )

    @pytest.mark.parametrize("level", REAL_LEVELS)
    def test_every_level_is_named_with_its_human_label(self, level, deps):
        update = build_search_query(state_with(experience_level=level), deps)
        assert update["progress_events"][0]["label"].endswith(
            f"· {EXPERIENCE_LEVEL_LABELS[level]}"
        )

    def test_the_category_and_freshness_are_untouched_by_a_level(self, deps):
        without = build_search_query(state_with(), deps)
        with_level = build_search_query(state_with(experience_level="senior"), deps)
        assert with_level["freshness_tbs"] == without["freshness_tbs"]
        assert with_level["source_domains"] == without["source_domains"]
        assert with_level["freshness_cutoff_utc"] == without["freshness_cutoff_utc"]


class TestSearchNodeSendsTheLevel:
    """The request that actually goes out carries the selected seniority."""

    def test_the_level_shapes_the_live_query(self):
        client = CapturingClient()
        search_current_jobs(state_with(experience_level="senior"), live_deps(client))
        query, _ = client.calls[0]
        assert '"senior data analyst"' in query

    def test_without_a_level_the_live_query_is_unchanged(self):
        client = CapturingClient()
        search_current_jobs(state_with(), live_deps(client))
        query, _ = client.calls[0]
        assert '"data analyst intern"' in query
        assert "senior" not in query


class TestNormalizeNodePassesTheLevel:
    """The request is recorded on each posting, and never becomes the detection."""

    def test_the_requested_level_is_recorded(self, deps):
        state = state_with(
            raw_jobs=[make_raw()], data_mode="cached", experience_level="senior"
        )
        update = normalize_jobs_node(state, deps)
        posting = update["normalized_jobs"][0]
        assert posting.requested_experience_level == "senior"

    def test_the_request_does_not_become_the_detected_level(self, deps):
        state = state_with(
            raw_jobs=[make_raw()], data_mode="cached", experience_level="senior"
        )
        update = normalize_jobs_node(state, deps)
        posting = update["normalized_jobs"][0]
        # The page is an internship, and says so. Asking for Senior does not
        # change what the page states.
        assert posting.experience_level == "internship"

    def test_a_run_without_a_level_records_unknown(self, deps):
        state = state_with(raw_jobs=[make_raw()], data_mode="cached")
        update = normalize_jobs_node(state, deps)
        assert update["normalized_jobs"][0].requested_experience_level == "unknown"


def senior_job() -> Any:
    """A posting whose own page states a senior level."""
    return make_job(
        job_id="job_senior00001",
        title="Senior Data Analyst",
        source_url="https://job-boards.greenhouse.io/lakeside/jobs/1",
        experience_level="senior",
        experience_level_evidence='title contains "senior"',
    )


def intern_job() -> Any:
    """A posting whose own page states an internship level."""
    return make_job(
        job_id="job_intern00001",
        title="Data Analyst Intern",
        source_url="https://job-boards.greenhouse.io/lakeside/jobs/2",
        experience_level="internship",
        experience_level_evidence='title contains "intern"',
    )


def silent_job() -> Any:
    """A posting that never states a level at all."""
    return make_job(
        job_id="job_silent00001",
        title="Data Analyst",
        source_url="https://job-boards.greenhouse.io/lakeside/jobs/3",
        experience_level="unknown",
    )


class TestFilterNodeAppliesTheLevel:
    """Only a posting that states a *different* level is removed."""

    def test_a_conflicting_posting_is_removed(self, deps):
        state = state_with(
            normalized_jobs=[senior_job(), intern_job()], experience_level="senior"
        )
        update = filter_and_deduplicate_jobs(state, deps)
        assert [job.job_id for job in update["filtered_jobs"]] == ["job_senior00001"]

    def test_a_posting_that_states_no_level_is_kept(self, deps):
        state = state_with(
            normalized_jobs=[senior_job(), silent_job()], experience_level="senior"
        )
        update = filter_and_deduplicate_jobs(state, deps)
        assert {job.job_id for job in update["filtered_jobs"]} == {
            "job_senior00001",
            "job_silent00001",
        }

    def test_the_removal_reason_is_reported(self, deps):
        state = state_with(
            normalized_jobs=[senior_job(), intern_job()], experience_level="senior"
        )
        update = filter_and_deduplicate_jobs(state, deps)
        assert any("you searched Senior" in reason for reason in update["warnings"])

    def test_a_run_without_a_level_filters_nothing_by_seniority(self, deps):
        jobs = [senior_job(), intern_job(), silent_job()]
        update = filter_and_deduplicate_jobs(state_with(normalized_jobs=jobs), deps)
        assert len(update["filtered_jobs"]) == 3
        assert "errors" not in update


class TestNoPostingsSurviveTheLevelFilter:
    """An empty screen explains itself and never broadens the search."""

    def test_the_error_names_the_level_and_the_next_step(self, deps):
        state = state_with(
            normalized_jobs=[intern_job()], experience_level="senior"
        )
        update = filter_and_deduplicate_jobs(state, deps)
        assert update["filtered_jobs"] == []
        message = update["errors"][0]
        assert "Senior" in message
        assert "1 of the 1 posting(s)" in message
        # The message names the level that *was* found, so the next click is
        # obvious rather than a guess.
        assert "Internship" in message
        assert "Any level" in message
        assert "was not widened for you" in message

    def test_the_error_counts_every_off_level_posting(self, deps):
        mid_job = make_job(
            job_id="job_mid0000001",
            title="Mid-level Data Analyst",
            source_url="https://job-boards.greenhouse.io/lakeside/jobs/4",
            experience_level="mid",
        )
        state = state_with(
            normalized_jobs=[intern_job(), mid_job], experience_level="senior"
        )
        update = filter_and_deduplicate_jobs(state, deps)
        assert update["filtered_jobs"] == []
        message = update["errors"][0]
        assert "2 of the 2 posting(s)" in message
        # Every off-level level that was actually retrieved is offered by name.
        assert "Internship" in message
        assert "Mid-level" in message

    def test_a_surviving_posting_produces_no_error(self, deps):
        state = state_with(
            normalized_jobs=[intern_job(), senior_job()], experience_level="internship"
        )
        update = filter_and_deduplicate_jobs(state, deps)
        assert [job.job_id for job in update["filtered_jobs"]] == ["job_intern00001"]
        assert "errors" not in update

    def test_a_level_wipeout_is_named_ahead_of_the_freshness_guidance(self, deps):
        # Freshness was not the reason nothing survived, so saying so would send
        # the user to the wrong control.
        state = state_with(
            normalized_jobs=[intern_job()],
            freshness_window="last_hour",
            experience_level="senior",
        )
        update = filter_and_deduplicate_jobs(state, deps)
        assert "Senior" in update["errors"][0]
        assert "Last 1 hour" not in update["errors"][0]

    def test_the_generic_message_survives_when_no_level_is_selected(self, deps):
        state = state_with(normalized_jobs=[make_job(is_closed=True)])
        update = filter_and_deduplicate_jobs(state, deps)
        assert "Retry with Direct Company Careers" in update["errors"][0]

    def test_the_last_hour_message_survives_a_selected_level(self, deps):
        # Nothing was removed for its level here, so the freshness guidance is
        # still the honest thing to say.
        state = state_with(
            normalized_jobs=[make_job(is_closed=True)],
            freshness_window="last_hour",
            experience_level="senior",
        )
        update = filter_and_deduplicate_jobs(state, deps)
        assert "Last 1 hour" in update["errors"][0]


class TestSearchedRoles:
    """The titles handed to the scorer mirror the titles that were searched."""

    def test_no_role_produces_no_extra_titles(self):
        assert searched_roles({}) == []

    def test_without_a_level_the_typed_role_is_used(self):
        assert searched_roles({"role": "  Data Analyst  Intern "}) == [
            "Data Analyst Intern"
        ]

    def test_with_a_level_the_qualified_phrasing_is_used(self):
        assert searched_roles(
            {"role": "Data Analyst Intern", "experience_level": "senior"}
        ) == ["senior data analyst", "sr data analyst"]

    def test_the_typed_level_word_never_survives_a_selected_level(self):
        roles = searched_roles(
            {"role": "Data Analyst Intern", "experience_level": "manager"}
        )
        assert all("intern" not in role for role in roles)


class TestScoreJobsUsesTheSearchedRole:
    """Ranking rewards the title the user searched for, at the level they chose."""

    def test_the_searched_role_lifts_role_alignment(self, deps, resume):
        job = make_job(title="Senior Data Scientist")
        searched = score_jobs(
            state_with(role="Data Scientist", filtered_jobs=[job], resume=resume), deps
        )
        unsearched = score_jobs(
            state_with(role="", filtered_jobs=[job], resume=resume), deps
        )
        assert (
            searched["ranked_matches"][0].role_score
            > unsearched["ranked_matches"][0].role_score
        )

    def test_the_level_qualified_phrasing_aligns_with_a_senior_title(self, deps, resume):
        job = make_job(title="Senior Data Scientist")
        state = state_with(role="Data Scientist Intern", filtered_jobs=[job], resume=resume)
        without = score_jobs(state, deps)
        with_level = score_jobs({**state, "experience_level": "senior"}, deps)
        assert (
            with_level["ranked_matches"][0].role_score
            > without["ranked_matches"][0].role_score
        )

    def test_the_workshop_defaults_score_exactly_as_before(self, deps, resume):
        # "Data Analyst Intern" is already one of the resume's target roles, so
        # merging the searched role is a no-op for the demo script.
        jobs = [senior_job(), intern_job(), silent_job()]
        update = score_jobs(state_with(filtered_jobs=jobs, resume=resume), deps)
        expected = rank_jobs(
            jobs,
            resume,
            location="Houston, TX",
            work_mode="Any",
            now=FIXED_NOW,
            top_n=3,
        )
        assert [match.model_dump() for match in update["ranked_matches"]] == [
            match.model_dump() for match in expected
        ]

    def test_an_off_resume_search_is_rewarded_without_any_level(self, deps, resume):
        # The resume is fixed demo data; the search box is not. A role the user
        # typed counts even when no level is chosen.
        job = make_job(title="Data Scientist")
        update = score_jobs(
            state_with(role="Data Scientist", filtered_jobs=[job], resume=resume), deps
        )
        resume_only = rank_jobs(
            [job], resume, location="Houston, TX", work_mode="Any", now=FIXED_NOW
        )
        assert update["ranked_matches"][0].role_score > resume_only[0].role_score

    def test_asking_for_a_level_makes_no_claim_about_a_silent_posting(self, deps, resume):
        # The request may reshape the *titles* being compared, but it is not
        # evidence about the posting: nothing else moves, and no concern is
        # raised about a seniority the page never stated.
        job = silent_job()
        base = state_with(role="Data Analyst", filtered_jobs=[job], resume=resume)
        without = score_jobs(base, deps)["ranked_matches"][0]
        with_level = score_jobs({**base, "experience_level": "senior"}, deps)[
            "ranked_matches"
        ][0]
        assert with_level.skill_score == without.skill_score
        assert with_level.similarity_score == without.similarity_score
        assert with_level.experience_score == without.experience_score
        assert with_level.preference_score == without.preference_score
        assert with_level.concerns == without.concerns
        assert all("senior" not in concern.lower() for concern in with_level.concerns)


class TestGraphRunCarriesTheLevel:
    """End to end, against the labelled cache, with no model and no network."""

    def run(self, level: str | None, thread: str) -> dict:
        """Invoke the compiled graph once with an optional level."""
        deps = AgentDeps(
            settings=Settings(demo_mode="cached"),
            search_adapter=FirecrawlSearchAdapter(),
            llm=NullLLMClient(),
            now=lambda: FIXED_NOW,
        )
        run_input = state_with()
        if level is not None:
            run_input["experience_level"] = level
        return build_graph(deps).invoke(
            run_input, {"configurable": {"thread_id": thread}}
        )

    def test_the_level_survives_a_checkpointed_run(self):
        result = self.run("internship", "level-internship")
        assert result["experience_level"] == "internship"

    def test_a_matching_level_still_reaches_the_selection_pause(self):
        result = self.run("internship", "level-internship-pause")
        assert result["__interrupt__"]
        assert result["filtered_jobs"]

    def test_a_level_with_no_postings_ends_with_an_actionable_error(self):
        result = self.run("senior", "level-senior")
        # The cache holds internship and entry-level postings only.
        assert result["filtered_jobs"] == []
        assert "__interrupt__" not in result
        message = result["errors"][0]
        assert "Senior" in message
        assert "Any level" in message
        # A cached run says so, so the screen never reads as a broken feature.
        assert "cached demonstration set" in message
        assert "was not widened for you" in message

    def test_a_run_without_a_level_is_unaffected(self):
        result = self.run(None, "level-absent")
        assert result["filtered_jobs"]
        assert result["__interrupt__"]
