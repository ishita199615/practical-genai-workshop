"""Conditional routing decisions for the graph.

Routing is deterministic and readable: each function returns the name of the
next node from the state alone.
"""

from __future__ import annotations

from agent.state import CareerAgentState

MAX_AUTO_REVISIONS = 1


def route_after_retrieval(state: CareerAgentState) -> str:
    """Stop early when nothing usable was retrieved."""
    return "score_jobs" if state.get("filtered_jobs") else "__end__"


def route_after_ats(state: CareerAgentState, threshold: int = 80) -> str:
    """Send a below-threshold score through the recommendation panel."""
    assessment = state.get("ats_assessment")
    if assessment is None:
        return "__end__"
    if assessment.total_score < threshold:
        return "recommend_ats_changes"
    return "draft_application"


def route_after_validation(state: CareerAgentState) -> str:
    """Revise once automatically when unsupported claims remain."""
    report = state.get("validation_report")
    if report is None:
        return "human_approval"
    needs_revision = bool(report.unsupported_claims) or not report.passed
    if needs_revision and state.get("revision_count", 0) < MAX_AUTO_REVISIONS:
        return "revise_application"
    return "human_approval"


def route_after_approval(state: CareerAgentState) -> str:
    """Export only after an explicit approval; never on reject."""
    decision = state.get("approval_decision")
    if decision == "approve":
        return "export_package"
    if decision == "request_changes":
        return "revise_application"
    return "__end__"
