"""Assembly of the single controlled Cougar Career Agent graph.

One graph, explicit edges, two human pauses. There is no autonomous
multi-agent conversation here: every step is a named node with a known input
and a known output.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from agent import nodes, routing
from agent.nodes import AgentDeps
from agent.state import CareerAgentState
from config import Settings, load_settings
from services.gemini_client import build_llm_client
from tools.firecrawl_search import FirecrawlSearchAdapter

# The checkpointer round-trips our own Pydantic models, so they are declared
# explicitly rather than relying on a deprecated implicit allowance.
CHECKPOINT_MODULES: tuple[tuple[str, str], ...] = (
    ("models.resume", "ResumeProfile"),
    ("models.resume", "ResumeBullet"),
    ("models.resume", "ExperienceEntry"),
    ("models.job", "RawJobResult"),
    ("models.job", "JobPosting"),
    ("models.job", "ExtractedJobFields"),
    ("models.match", "MatchResult"),
    ("models.ats", "AtsAssessment"),
    ("models.ats", "AtsRecommendation"),
    ("models.application", "TailoredApplication"),
    ("models.application", "TailoredDraft"),
    ("models.application", "RevisedBullet"),
    ("models.validation", "ValidationReport"),
    ("models.validation", "ClaimReview"),
    ("models.validation", "ClaimReviewBatch"),
)


def build_checkpointer() -> InMemorySaver:
    """Build the in-memory checkpointer used for the workshop prototype."""
    return InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_MODULES)
    )


NODE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "load_sample_resume": nodes.load_sample_resume,
    "build_search_query": nodes.build_search_query,
    "search_current_jobs": nodes.search_current_jobs,
    "normalize_jobs": nodes.normalize_jobs_node,
    "filter_and_deduplicate_jobs": nodes.filter_and_deduplicate_jobs,
    "score_jobs": nodes.score_jobs,
    "explain_top_matches": nodes.explain_top_matches,
    "select_job": nodes.select_job,
    "score_ats_readiness": nodes.score_ats_readiness,
    "recommend_ats_changes": nodes.recommend_ats_changes,
    "draft_application": nodes.draft_application,
    "rescore_proposed_resume": nodes.rescore_proposed_resume,
    "validate_application": nodes.validate_application,
    "revise_application": nodes.revise_application,
    "human_approval": nodes.human_approval,
    "export_package": nodes.export_package,
}


def build_deps(settings: Settings | None = None, llm: Any | None = None) -> AgentDeps:
    """Build the default dependency set from configuration."""
    resolved = settings or load_settings()
    return AgentDeps(
        settings=resolved,
        search_adapter=FirecrawlSearchAdapter(
            api_key=resolved.firecrawl_api_key,
            base_url=resolved.firecrawl_base_url,
        ),
        llm=llm
        or build_llm_client(
            resolved.gemini_api_key, resolved.gemini_model, resolved.model_chain
        ),
    )


def build_graph(deps: AgentDeps, checkpointer: Any | None = None) -> Any:
    """Compile the career-agent graph with checkpointing enabled.

    ``InMemorySaver`` is intentional for a workshop prototype: state lives for
    the length of the Streamlit session and nothing is persisted to disk.
    """
    builder = StateGraph(CareerAgentState)
    for name, function in NODE_FUNCTIONS.items():
        builder.add_node(name, partial(function, deps=deps))

    builder.add_edge(START, "load_sample_resume")
    builder.add_edge("load_sample_resume", "build_search_query")
    builder.add_edge("build_search_query", "search_current_jobs")
    builder.add_edge("search_current_jobs", "normalize_jobs")
    builder.add_edge("normalize_jobs", "filter_and_deduplicate_jobs")

    builder.add_conditional_edges(
        "filter_and_deduplicate_jobs",
        routing.route_after_retrieval,
        {"score_jobs": "score_jobs", "__end__": END},
    )
    builder.add_edge("score_jobs", "explain_top_matches")
    builder.add_edge("explain_top_matches", "select_job")
    builder.add_edge("select_job", "score_ats_readiness")

    threshold = deps.settings.ats_recommendation_threshold
    builder.add_conditional_edges(
        "score_ats_readiness",
        partial(routing.route_after_ats, threshold=threshold),
        {
            "recommend_ats_changes": "recommend_ats_changes",
            "draft_application": "draft_application",
            "__end__": END,
        },
    )
    builder.add_edge("recommend_ats_changes", "draft_application")
    builder.add_edge("draft_application", "rescore_proposed_resume")
    builder.add_edge("rescore_proposed_resume", "validate_application")

    builder.add_conditional_edges(
        "validate_application",
        routing.route_after_validation,
        {
            "revise_application": "revise_application",
            "human_approval": "human_approval",
        },
    )
    builder.add_edge("revise_application", "validate_application")

    builder.add_conditional_edges(
        "human_approval",
        routing.route_after_approval,
        {
            "export_package": "export_package",
            "revise_application": "revise_application",
            "__end__": END,
        },
    )
    builder.add_edge("export_package", END)

    return builder.compile(checkpointer=checkpointer or build_checkpointer())
