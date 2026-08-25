"""Regenerate ``data/expected_demo_output.json`` from a cached rehearsal run.

Run this after changing the cached data, the scoring rubrics, or the drafting
fallback, so the rehearsal fixture keeps matching what the app actually does::

    python scripts/regenerate_expected_output.py

The run uses cached data, a fixed reference time, and no LLM provider, so the
output is reproducible on any machine.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.types import Command  # noqa: E402

from agent.graph import build_graph  # noqa: E402
from agent.nodes import AgentDeps  # noqa: E402
from config import Settings  # noqa: E402
from services.llm_interface import NullLLMClient  # noqa: E402
from tools.firecrawl_search import FirecrawlSearchAdapter  # noqa: E402

REFERENCE_TIME = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)

SEARCH = {
    "role": "Data Analyst Intern",
    "location": "Houston, TX",
    "work_mode": "Any",
    "query_category": "company_careers",
    "freshness_window": "last_24_hours",
}


def main() -> None:
    """Run the rehearsal pipeline and write the expected-output fixture."""
    deps = AgentDeps(
        settings=Settings(demo_mode="cached"),
        search_adapter=FirecrawlSearchAdapter(),
        llm=NullLLMClient(),
        now=lambda: REFERENCE_TIME,
    )
    graph = build_graph(deps)
    config = {"configurable": {"thread_id": "expected_demo"}}

    state = graph.invoke(dict(SEARCH), config)
    job_id = state["__interrupt__"][0].value["options"][0]["job_id"]
    state = graph.invoke(Command(resume=job_id), config)

    jobs = {job.job_id: job for job in state["filtered_jobs"]}
    payload = {
        "notice": (
            "Expected output of the cached rehearsal run with no LLM provider "
            "configured. Regenerate with scripts/regenerate_expected_output.py "
            "when the cache or a rubric changes."
        ),
        "reference_time": REFERENCE_TIME.isoformat(),
        "data_mode": state["data_mode"],
        "search": {
            **SEARCH,
            "search_query": state["search_query"],
            "freshness_tbs": state["freshness_tbs"],
        },
        "retrieved_count": len(state["raw_jobs"]),
        "normalized_count": len(state["normalized_jobs"]),
        "kept_count": len(state["filtered_jobs"]),
        "ranked": [
            {
                "rank": index,
                "job_id": match.job_id,
                "title": jobs[match.job_id].title,
                "company": jobs[match.job_id].company,
                "source_label": jobs[match.job_id].source_label,
                "source_url": jobs[match.job_id].source_url,
                "freshness_evidence": jobs[match.job_id].freshness_evidence,
                "total_score": match.total_score,
                "components": {
                    "skill": match.skill_score,
                    "similarity": match.similarity_score,
                    "role": match.role_score,
                    "experience": match.experience_score,
                    "preference": match.preference_score,
                },
                "matched_skills": match.matched_skills,
                "missing_skills": match.missing_skills,
            }
            for index, match in enumerate(state["ranked_matches"], start=1)
        ],
        "selected_job_id": state["selected_job_id"],
        "ats_original": state["ats_assessment"].model_dump(mode="json"),
        "ats_projected": state["projected_ats_assessment"].model_dump(mode="json"),
        "tailored_application": state["tailored_application"].model_dump(mode="json"),
        "validation": state["validation_report"].model_dump(mode="json"),
        "approval_pause": state["__interrupt__"][0].value["message"],
    }

    output = PROJECT_ROOT / "data" / "expected_demo_output.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {output}")
    print(
        f"Selected: {jobs[state['selected_job_id']].company} · "
        f"ATS {payload['ats_original']['total_score']} → "
        f"{payload['ats_projected']['total_score']}"
    )


if __name__ == "__main__":
    main()
