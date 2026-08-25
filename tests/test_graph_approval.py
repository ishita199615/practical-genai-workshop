"""End-to-end graph behaviour: pauses, routing, approval, and export."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.types import Command

from agent.graph import build_graph
from agent.nodes import AgentDeps
from agent.routing import route_after_approval, route_after_ats, route_after_validation
from config import Settings
from models.validation import ValidationReport
from services.llm_interface import NullLLMClient
from tests.conftest import FIXED_NOW, make_job
from tools.ats_scorer import assess_ats
from tools.firecrawl_search import FirecrawlSearchAdapter

RUN_INPUT: dict[str, Any] = {
    "role": "Data Analyst Intern",
    "location": "Houston, TX",
    "work_mode": "Any",
    "query_category": "company_careers",
    "freshness_window": "last_24_hours",
}


@pytest.fixture
def deps(tmp_path) -> AgentDeps:
    """Cached-mode dependencies writing exports into a temporary directory."""
    settings = Settings(demo_mode="cached", output_dir=str(tmp_path))
    return AgentDeps(
        settings=settings,
        search_adapter=FirecrawlSearchAdapter(),
        llm=NullLLMClient(),
        now=lambda: FIXED_NOW,
    )


@pytest.fixture
def graph(deps):
    """A compiled graph with in-memory checkpointing."""
    return build_graph(deps)


def config(thread: str) -> dict:
    """Build a LangGraph config for one thread."""
    return {"configurable": {"thread_id": thread}}


def interrupts(result: dict) -> list[dict]:
    """Return the interrupt payloads from a graph result."""
    return [item.value for item in result.get("__interrupt__", ())]


def run_to_approval(graph, thread: str) -> dict:
    """Drive the graph through job selection and up to the approval pause."""
    first = graph.invoke(RUN_INPUT, config(thread))
    job_id = interrupts(first)[0]["options"][0]["job_id"]
    return graph.invoke(Command(resume=job_id), config(thread))


class TestRoutingFunctions:
    """Routing is a pure function of state."""

    def test_a_low_ats_score_routes_to_recommendations(self, resume, now):
        assessment = assess_ats(
            make_job(required_skills=["Power BI", "Looker", "SAS", "ETL"]), resume, now=now
        )
        state = {"ats_assessment": assessment}
        assert route_after_ats(state, threshold=80) == "recommend_ats_changes"

    def test_a_strong_ats_score_routes_straight_to_drafting(self, resume, now):
        assessment = assess_ats(make_job(), resume, now=now)
        state = {"ats_assessment": assessment}
        assert route_after_ats(state, threshold=80) == "draft_application"

    def test_unsupported_claims_trigger_one_revision(self):
        report = ValidationReport(valid_source_ids=True, passed=False)
        assert route_after_validation({"validation_report": report, "revision_count": 0}) == (
            "revise_application"
        )

    def test_the_revision_limit_is_one(self):
        report = ValidationReport(valid_source_ids=True, passed=False)
        assert route_after_validation({"validation_report": report, "revision_count": 1}) == (
            "human_approval"
        )

    def test_a_passing_report_goes_to_approval(self):
        report = ValidationReport(valid_source_ids=True, passed=True)
        assert route_after_validation({"validation_report": report}) == "human_approval"

    @pytest.mark.parametrize(
        ("decision", "expected"),
        [
            ("approve", "export_package"),
            ("request_changes", "revise_application"),
            ("reject", "__end__"),
            (None, "__end__"),
        ],
    )
    def test_approval_routes(self, decision, expected):
        assert route_after_approval({"approval_decision": decision}) == expected


class TestJobSelectionPause:
    """The graph pauses for a human to choose a job."""

    def test_the_graph_pauses_with_three_options(self, graph):
        result = graph.invoke(RUN_INPUT, config("t-select"))
        payloads = interrupts(result)
        assert payloads[0]["kind"] == "job_selection"
        assert len(payloads[0]["options"]) == 3

    def test_the_activity_log_is_observable(self, graph):
        result = graph.invoke(RUN_INPUT, config("t-log"))
        labels = [item["label"] for item in result["progress_events"]]
        assert any("resume loaded" in label.lower() for label in labels)
        assert any("scored" in label.lower() for label in labels)

    def test_selection_produces_ats_scoring_and_a_draft(self, graph):
        result = run_to_approval(graph, "t-flow")
        assert result["selected_job"] is not None
        assert result["ats_assessment"] is not None
        assert result["projected_ats_assessment"] is not None
        assert result["tailored_application"] is not None
        assert result["validation_report"] is not None


class TestApprovalPause:
    """No side effect happens before a human decides."""

    def test_the_graph_pauses_before_export(self, graph, deps):
        result = run_to_approval(graph, "t-pause")
        payload = interrupts(result)[0]
        assert payload["kind"] == "approval"
        assert payload["allowed_decisions"] == ["approve", "request_changes", "reject"]
        assert "No application has been submitted." in payload["message"]
        assert not list(deps.settings.output_path.glob("application_package_*"))

    def test_the_pause_payload_is_json_serializable(self, graph):
        import json

        payload = interrupts(run_to_approval(graph, "t-json"))[0]
        assert json.dumps(payload)

    def test_approve_exports_both_files(self, graph, deps):
        run_to_approval(graph, "t-approve")
        final = graph.invoke(
            Command(resume={"decision": "approve"}), config("t-approve")
        )
        assert len(final["output_files"]) == 2
        assert final["output_files"][0].endswith(".md")
        assert final["output_files"][1].endswith(".json")
        assert final["approved_at"] is not None

    def test_the_exported_markdown_states_no_submission(self, graph, deps):
        run_to_approval(graph, "t-export")
        final = graph.invoke(Command(resume={"decision": "approve"}), config("t-export"))
        from pathlib import Path

        markdown = Path(final["output_files"][0]).read_text(encoding="utf-8")
        assert "No application was submitted." in markdown
        assert "not an official employer ATS score" in markdown
        assert "Demo Job Match Score" in markdown

    def test_reject_exports_nothing(self, graph, deps):
        run_to_approval(graph, "t-reject")
        final = graph.invoke(Command(resume={"decision": "reject"}), config("t-reject"))
        assert final.get("output_files", []) == []
        assert final["approval_decision"] == "reject"
        assert not list(deps.settings.output_path.glob("application_package_*"))

    def test_request_changes_revises_and_pauses_again(self, graph, deps):
        run_to_approval(graph, "t-changes")
        result = graph.invoke(
            Command(
                resume={
                    "decision": "request_changes",
                    "feedback": "Shorten the summary.",
                }
            ),
            config("t-changes"),
        )
        assert interrupts(result)[0]["kind"] == "approval"
        assert result["revision_count"] >= 1
        assert not list(deps.settings.output_path.glob("application_package_*"))

    def test_a_second_approval_after_changes_exports(self, graph):
        run_to_approval(graph, "t-second")
        graph.invoke(
            Command(resume={"decision": "request_changes", "feedback": "Tighten it."}),
            config("t-second"),
        )
        final = graph.invoke(Command(resume={"decision": "approve"}), config("t-second"))
        assert len(final["output_files"]) == 2


class TestGuardrailsEndToEnd:
    """The truthfulness guarantees hold across the whole run."""

    def test_no_unsupported_skill_reaches_the_draft(self, graph):
        result = run_to_approval(graph, "t-guard")
        application = result["tailored_application"]
        assessment = result["ats_assessment"]
        text = " ".join(
            [
                application.revised_summary,
                application.cover_letter,
                " ".join(bullet.revised_text for bullet in application.revised_bullets),
                " ".join(application.reordered_skills),
            ]
        ).lower()
        for gap in assessment.unsupported_job_gaps:
            assert gap not in text

    def test_skills_are_only_reordered(self, graph):
        result = run_to_approval(graph, "t-skills")
        assert set(result["tailored_application"].reordered_skills) == set(
            result["resume"].skills
        )

    def test_every_revised_bullet_has_a_real_source_id(self, graph):
        result = run_to_approval(graph, "t-ids")
        valid = set(result["resume"].bullet_index())
        bullets = result["tailored_application"].revised_bullets
        assert len(bullets) == 2
        for bullet in bullets:
            assert bullet.source_bullet_id in valid

    def test_validation_passes_for_the_rehearsal_run(self, graph):
        result = run_to_approval(graph, "t-valid")
        report = result["validation_report"]
        assert report.passed, report.failed_checks() + [
            review.model_dump() for review in report.unsupported_claims
        ]

    def test_the_run_matches_the_recorded_rehearsal_fixture(self, tmp_path):
        """The rehearsal fixture is a regression guard on the whole pipeline."""
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        fixture = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "expected_demo_output.json")
            .read_text(encoding="utf-8")
        )
        reference = datetime.fromisoformat(fixture["reference_time"])
        deps = AgentDeps(
            settings=Settings(demo_mode="cached", output_dir=str(tmp_path)),
            search_adapter=FirecrawlSearchAdapter(),
            llm=NullLLMClient(),
            now=lambda: reference,
        )
        graph = build_graph(deps)
        state = run_to_approval(graph, "t-fixture")

        assert state["data_mode"] == fixture["data_mode"]
        assert state["search_query"] == fixture["search"]["search_query"]
        assert len(state["filtered_jobs"]) == fixture["kept_count"]
        assert [match.total_score for match in state["ranked_matches"]] == [
            entry["total_score"] for entry in fixture["ranked"]
        ]
        assert state["ats_assessment"].total_score == fixture["ats_original"]["total_score"]
        assert (
            state["projected_ats_assessment"].total_score
            == fixture["ats_projected"]["total_score"]
        )

    def test_scores_are_reproducible_across_runs(self, graph):
        first = run_to_approval(graph, "t-repeat-1")
        second = run_to_approval(graph, "t-repeat-2")
        assert first["ats_assessment"].total_score == second["ats_assessment"].total_score
        assert [match.total_score for match in first["ranked_matches"]] == [
            match.total_score for match in second["ranked_matches"]
        ]
