"""Graph node implementations.

Each node is a small, testable function over :class:`CareerAgentState`.
External clients arrive through :class:`AgentDeps`, so the graph never imports
a vendor SDK and tests can inject fakes.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from langgraph.types import interrupt

from agent.state import CareerAgentState, event
from config import Settings
from models.application import RevisedBullet, TailoredApplication, TailoredDraft
from models.ats import BAND_LABELS, AtsAssessment, AtsRecommendation
from models.job import (
    EXPERIENCE_LEVEL_LABELS,
    FRESHNESS_LABELS,
    SOURCE_CATEGORY_LABELS,
    ExperienceLevel,
    JobPosting,
    RawJobResult,
)
from models.match import MatchResult
from models.resume import ResumeProfile
from prompts import render_prompt
from tools.ats_scorer import (
    assess_ats,
    rescore_proposed,
    safe_recommendations,
)
from tools.claim_validator import validate_application as run_validation
from tools.exporter import export_package as write_package
from tools.experience_level import level_query_terms, levels_conflict
from tools.firecrawl_search import (
    FirecrawlError,
    FirecrawlSearchAdapter,
    build_search_request,
    cached_raw_results,
    domain_filter_for,
    filter_to_category,
    freshness_cutoff,
    load_cache,
    raw_results_to_models,
    search_with_domain_retry,
)
from tools.job_filter import filter_and_deduplicate
from tools.job_normalizer import normalize_jobs
from tools.job_scorer import canonical_skill, rank_jobs

logger = logging.getLogger(__name__)

MAX_AUTO_REVISIONS = 1

# Page extractions run concurrently to fit the demo's time budget, but a wide
# burst trips free-tier request-per-minute limits. Four keeps both in view.
EXTRACTION_CONCURRENCY = 4


@dataclass
class AgentDeps:
    """Injected dependencies for every node."""

    settings: Settings
    search_adapter: FirecrawlSearchAdapter
    llm: Any
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


def requested_experience_level(state: CareerAgentState) -> ExperienceLevel:
    """Return the seniority the user asked for, defaulting to ``"unknown"``.

    A run that never sets a level behaves exactly as it did before the control
    existed: the query is not narrowed and no posting is filtered on seniority.

    The value crosses a checkpointer boundary, so it is not trusted to be a
    usable key: an unhashable value would otherwise raise here instead of
    falling back to no level filter at all.
    """
    level = state.get("experience_level") or "unknown"
    if not isinstance(level, str) or level not in EXPERIENCE_LEVEL_LABELS:
        return "unknown"
    return level


def searched_roles(state: CareerAgentState) -> list[str]:
    """Return the role titles the user actually searched for.

    Without a level this is the free-text role as typed. With a level selected
    it is the level-qualified phrasing instead: :func:`level_query_terms`
    strips any seniority word already in the text, so a Senior search never
    also rewards the "Intern" the user happened to type first.
    """
    role = " ".join((state.get("role") or "").split())
    level = requested_experience_level(state)
    if level == "unknown":
        return [role] if role else []
    return level_query_terms(role, level)


def _level_suffix(level: ExperienceLevel) -> str:
    """Return the activity-log suffix naming a selected level, if any."""
    if level == "unknown":
        return ""
    return f" · {EXPERIENCE_LEVEL_LABELS[level]}"


# --------------------------------------------------------------------------
# Retrieval stage
# --------------------------------------------------------------------------


def load_sample_resume(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Load and validate the master resume the settings point at."""
    path = deps.settings.resume_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        resume = ResumeProfile.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - user-facing failure
        logger.error("Sample resume could not be loaded: %s", type(exc).__name__)
        return {
            "errors": ["The resume file could not be loaded or validated."],
            "progress_events": [event("Resume loaded", "error")],
        }
    return {
        "resume": resume,
        "revision_count": 0,
        "progress_events": [
            event(
                "Resume loaded",
                detail=f"{resume.name} ({deps.settings.resume_descriptor})",
            )
        ],
    }


def build_search_query(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Combine the query category, freshness window, and level into one request."""
    now = deps.now()
    experience_level = requested_experience_level(state)
    request = build_search_request(
        role=state.get("role", ""),
        location=state.get("location", ""),
        work_mode=state.get("work_mode", "Any"),
        query_category=state.get("query_category", "company_careers"),
        freshness_window=state.get("freshness_window", "last_24_hours"),
        experience_level=experience_level,
        limit=deps.settings.max_job_results,
        timeout_seconds=deps.settings.search_timeout_seconds,
        now=now,
    )
    category_label = SOURCE_CATEGORY_LABELS.get(
        state.get("query_category", "all"), "All Public Sources"
    )
    freshness_label = FRESHNESS_LABELS.get(
        state.get("freshness_window", "last_24_hours"), "Last 24 hours"
    )
    return {
        "search_query": request.query,
        "freshness_tbs": request.tbs,
        "freshness_cutoff_utc": freshness_cutoff(
            state.get("freshness_window", "last_24_hours"), now
        ),
        "source_domains": domain_filter_for(state.get("query_category", "all")),
        "progress_events": [
            event(
                f"Search query created for {category_label} · {freshness_label}"
                f"{_level_suffix(experience_level)}",
                detail=request.query,
            )
        ],
    }


def search_current_jobs(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Retrieve current public job pages, falling back to the labelled cache.

    Cached results are never presented as live, and a user-selected category is
    never silently broadened.
    """
    settings = deps.settings
    now = deps.now()
    query_category = state.get("query_category", "company_careers")
    freshness_window = state.get("freshness_window", "last_24_hours")

    request = build_search_request(
        role=state.get("role", ""),
        location=state.get("location", ""),
        work_mode=state.get("work_mode", "Any"),
        query_category=query_category,
        freshness_window=freshness_window,
        experience_level=requested_experience_level(state),
        limit=settings.max_job_results,
        timeout_seconds=settings.search_timeout_seconds,
        now=now,
    )

    updates: dict[str, Any] = {}
    warnings: list[str] = []
    raw_dicts: list[dict[str, Any]] = []
    time_filter_applied = True
    data_mode = "cached"

    if settings.demo_mode in {"live", "auto"} and deps.search_adapter.available:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    search_with_domain_retry, deps.search_adapter, request
                )
                outcome = future.result(
                    timeout=settings.search_timeout_seconds * 2 + 15
                )
            raw_dicts = outcome.results
            time_filter_applied = outcome.time_filter_applied
            warnings.extend(outcome.notes)
            data_mode = "live"
        except concurrent.futures.TimeoutError:
            warnings.append(
                f"Live retrieval exceeded {settings.search_timeout_seconds} seconds."
            )
        except FirecrawlError as exc:
            warnings.append(_retrieval_message(str(exc)))
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            warnings.append(_retrieval_message(str(exc)))
    elif settings.demo_mode != "cached":
        warnings.append("Firecrawl is not configured, so live retrieval was skipped.")

    if raw_dicts:
        raw_jobs, url_warnings = raw_results_to_models(
            raw_dicts,
            query_category=query_category,
            freshness_window=freshness_window,
            retrieved_at=now,
            time_filter_applied=time_filter_applied,
        )
        warnings.extend(url_warnings)
        # The selected category is enforced on the detected source, so a
        # relaxed retry can never put an off-category result on screen.
        raw_jobs, off_category = filter_to_category(raw_jobs, query_category)
        if off_category:
            warnings.append(
                f"{off_category} result(s) outside the selected category were removed."
            )
    else:
        raw_jobs = []

    if not raw_jobs:
        if settings.demo_mode == "live" and data_mode == "live":
            return {
                "raw_jobs": [],
                "data_mode": "live",
                "retrieval_timestamp": now,
                "warnings": warnings,
                "errors": [
                    "No public job pages were returned for the selected category and "
                    "freshness window. Try Direct Company Careers or a wider window."
                ],
                "progress_events": [event("Firecrawl searched public job pages", "warn")],
            }
        raw_jobs, cache_warnings, cached_at = _load_cached_jobs(
            deps, query_category, freshness_window
        )
        warnings.extend(cache_warnings)
        data_mode = "cached"
        # Cached records keep the timestamp of the run that produced them, so
        # the banner never implies the data was fetched just now.
        if cached_at is not None:
            now = cached_at

    updates.update(
        {
            "raw_jobs": raw_jobs,
            "data_mode": data_mode,
            "retrieval_timestamp": now,
            "warnings": warnings,
            "progress_events": [
                event(
                    "Firecrawl searched public job pages"
                    if data_mode == "live"
                    else "Loaded cached demonstration results",
                    "ok" if data_mode == "live" else "warn",
                    detail=f"{len(raw_jobs)} page(s) retrieved and source categories classified",
                )
            ],
        }
    )
    if not raw_jobs:
        updates["errors"] = [
            "No public job pages were available from live retrieval or the cache."
        ]
    return updates


QUOTA_WARNING = (
    "The Gemini API reported a quota or rate limit, so some steps used the "
    "deterministic offline fallback instead of the model. Both scores are "
    "unaffected — they are always calculated in Python."
)


def _quota_warnings(deps: AgentDeps) -> list[str]:
    """Report a provider quota limit once, rather than degrading in silence."""
    if getattr(deps.llm, "quota_limited", False):
        return [QUOTA_WARNING]
    return []


def _retrieval_message(detail: str) -> str:
    """Turn a provider error into one concise, actionable sentence.

    Raw provider text is deliberately not shown: it is noisy on a projector and
    can echo request contents back into the interface.
    """
    lowered = detail.lower()
    if any(term in lowered for term in ("unauthorized", "401", "invalid token", "api key")):
        return (
            "Firecrawl rejected the API key. Check FIRECRAWL_API_KEY, or continue "
            "with cached demonstration results."
        )
    if any(term in lowered for term in ("rate limit", "429", "quota")):
        return (
            "Firecrawl reported a rate or quota limit. Wait a moment and retry, "
            "or continue with cached demonstration results."
        )
    if any(term in lowered for term in ("timeout", "timed out")):
        return "Live retrieval timed out before Firecrawl responded."
    if any(
        term in lowered
        for term in ("connection", "network", "dns", "resolve", "unreachable", "ssl")
    ):
        return "Live retrieval could not reach Firecrawl. Check the network connection."
    return "Live retrieval failed, so the demo fell back to cached results."


def _load_cached_jobs(
    deps: AgentDeps, query_category: str, freshness_window: str
) -> tuple[list[RawJobResult], list[str], datetime | None]:
    """Load clearly-labelled cached demonstration results.

    Returns the records, any warnings, and the timestamp of the run that
    originally produced the cache.
    """
    try:
        payload = load_cache(deps.settings.cache_path)
    except Exception as exc:  # noqa: BLE001 - user-facing failure
        logger.error("Cache load failed: %s", type(exc).__name__)
        return [], ["Cached demonstration data could not be loaded."], None

    entries = cached_raw_results(
        payload, query_category=query_category, freshness_window=freshness_window
    )
    retrieved_at = _parse_iso(payload.get("originally_retrieved_at")) or deps.now()
    raw_jobs: list[RawJobResult] = []
    for entry in entries:
        record = dict(entry)
        record["query_category"] = query_category
        record["freshness_window"] = freshness_window
        record["retrieved_at"] = retrieved_at
        try:
            raw_jobs.append(RawJobResult.model_validate(record))
        except Exception:  # noqa: BLE001 - cache is data
            logger.warning("Skipping an invalid cached record.")
    warnings = [
        "Live retrieval was unavailable. These are cached demonstration results, "
        f"originally retrieved at {retrieved_at.isoformat()}."
    ]
    return raw_jobs, warnings, retrieved_at


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO timestamp from cached data, tolerating a trailing Z."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_jobs_node(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Extract structured fields and preserve the cleaned full description."""
    raw_jobs = state.get("raw_jobs", [])
    if not raw_jobs:
        return {"normalized_jobs": [], "progress_events": [event("Pages normalized", "warn")]}

    postings, warnings = normalize_jobs(
        raw_jobs,
        deps.llm,
        max_description_chars=deps.settings.max_job_description_chars,
        data_mode=state.get("data_mode", "live"),
        limit=deps.settings.max_job_results,
        max_workers=EXTRACTION_CONCURRENCY,
        requested_experience_level=requested_experience_level(state),
    )
    return {
        "normalized_jobs": postings,
        "warnings": warnings + _quota_warnings(deps),
        "progress_events": [
            event(
                f"{len(postings)} page(s) normalized with full job descriptions",
                "ok" if postings else "warn",
            )
        ],
    }


def filter_and_deduplicate_jobs(
    state: CareerAgentState, deps: AgentDeps
) -> dict[str, Any]:
    """Remove closed, stale, snippet-only, generic, off-level, and duplicate postings."""
    normalized = state.get("normalized_jobs", [])
    experience_level = requested_experience_level(state)
    outcome = filter_and_deduplicate(
        normalized, requested_experience_level=experience_level
    )
    updates: dict[str, Any] = {
        "filtered_jobs": outcome.kept,
        "warnings": outcome.reasons(),
        "progress_events": [
            event(
                f"{outcome.removed_count} invalid, closed, or duplicate page(s) removed",
                "ok",
                detail=f"{len(outcome.kept)} posting(s) remain",
            )
        ],
    }
    if not outcome.kept:
        window = state.get("freshness_window", "last_24_hours")
        message = (
            "No usable public job descriptions remained after filtering. "
            "Retry with Direct Company Careers or All Public Sources."
        )
        if window == "last_hour":
            message = (
                "No verifiable postings remained for Last 1 hour. "
                "Use the expand action to search Last 24 hours with the same category."
            )
        # A level filter that emptied the screen is named explicitly, and the
        # message says which levels *were* found so the next click is obvious.
        # The search is never broadened silently.
        off_level = [
            job
            for job in normalized
            if levels_conflict(experience_level, job.experience_level)
        ]
        if off_level:
            label = EXPERIENCE_LEVEL_LABELS[experience_level]
            found = ", ".join(
                EXPERIENCE_LEVEL_LABELS[level]
                for level in sorted({job.experience_level for job in off_level})
            )
            source = (
                "cached demonstration set"
                if state.get("data_mode") == "cached"
                else "search"
            )
            message = (
                f"No {label} postings were found. {len(off_level)} of the "
                f"{len(normalized)} posting(s) the {source} returned state a "
                f"different level ({found}). Pick one of those levels, or choose "
                f"Any level to include every seniority — your choice of {label} "
                f"was not widened for you."
            )
        updates["errors"] = [message]
    return updates


def score_jobs(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Calculate the deterministic Demo Job Match Score and rank the top three."""
    jobs = state.get("filtered_jobs", [])
    resume = state.get("resume")
    if not jobs or resume is None:
        return {"ranked_matches": [], "progress_events": [event("Jobs scored", "warn")]}

    matches = rank_jobs(
        jobs,
        resume,
        location=state.get("location", ""),
        work_mode=state.get("work_mode", "Any"),
        now=deps.now(),
        top_n=3,
        extra_target_roles=searched_roles(state),
    )
    return {
        "ranked_matches": matches,
        "progress_events": [
            event(f"{len(jobs)} job(s) scored in Python"),
            event(f"Top {len(matches)} match(es) prepared"),
        ],
    }


def explain_top_matches(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Ask Gemini to explain the already-calculated scores.

    The explanation is prose only. It cannot change a number.
    """
    matches = state.get("ranked_matches", [])
    jobs = {job.job_id: job for job in state.get("filtered_jobs", [])}
    if not matches:
        return {}

    def explain(match: MatchResult) -> MatchResult:
        job = jobs.get(match.job_id)
        if job is None:
            return match
        try:
            explanation = _explain_match(match, job, deps)
        except Exception as exc:  # noqa: BLE001 - prose is optional, scores are not
            logger.warning("Match explanation failed: %s", type(exc).__name__)
            return match
        return match.model_copy(update={"explanation": explanation})

    # The three explanations are independent; running them together keeps the
    # whole workflow inside the demo's time budget.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(matches)) as pool:
        explained: list[MatchResult] = list(pool.map(explain, matches))

    return {
        "ranked_matches": explained,
        "progress_events": [event("Match explanations generated")],
    }


def _explain_match(match: MatchResult, job: JobPosting, deps: AgentDeps) -> str:
    """Return a short match explanation, with a deterministic fallback."""
    components = (
        f"skill coverage {match.skill_score}/100; text similarity "
        f"{match.similarity_score}/100; role alignment {match.role_score}/100; "
        f"experience {match.experience_score}/100; location and work mode "
        f"{match.preference_score}/100; total {match.total_score}/100"
    )
    if getattr(deps.llm, "available", False):
        prompt = render_prompt(
            "explain_match",
            JOB_TITLE=job.title,
            COMPANY=job.company,
            COMPONENTS=components,
            MATCHED_SKILLS=", ".join(match.matched_skills) or "none",
            MISSING_SKILLS=", ".join(match.missing_skills) or "none",
            CONCERNS="; ".join(match.concerns) or "none",
            PREFERENCE_SUMMARY=f"{job.location or 'location not stated'} · {job.work_mode}",
        )
        text = deps.llm.generate_text(prompt, temperature=0.2)
        if text:
            return text

    parts = [f"Scored {match.total_score}/100 against this posting."]
    if match.matched_skills:
        parts.append("Matches " + ", ".join(match.matched_skills) + ".")
    if match.missing_skills:
        parts.append("Missing or unclear: " + ", ".join(match.missing_skills) + ".")
    if match.concerns:
        parts.append(match.concerns[0])
    return " ".join(parts)


# --------------------------------------------------------------------------
# Selection and assessment
# --------------------------------------------------------------------------


def select_job(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Pause for the human to choose one of the ranked jobs."""
    matches = state.get("ranked_matches", [])
    jobs = {job.job_id: job for job in state.get("filtered_jobs", [])}
    selected_id = state.get("selected_job_id")

    if not selected_id:
        payload = {
            "kind": "job_selection",
            "prompt": "Select one job to tailor the application for.",
            "options": [
                {
                    "job_id": match.job_id,
                    "rank": index + 1,
                    "title": jobs[match.job_id].title if match.job_id in jobs else "",
                    "company": jobs[match.job_id].company if match.job_id in jobs else "",
                    "total_score": match.total_score,
                }
                for index, match in enumerate(matches)
            ],
        }
        selected_id = interrupt(payload)

    # The resume value crosses a process boundary, so it is not trusted to be a
    # usable key: an unhashable value would otherwise raise instead of failing
    # gracefully.
    job = jobs.get(selected_id) if isinstance(selected_id, str) else None
    if job is None:
        return {
            "errors": ["The selected job is no longer available in this run."],
            "progress_events": [event("Job selected", "error")],
        }
    return {
        "selected_job_id": selected_id,
        "selected_job": job,
        "progress_events": [event(f"Job selected: {job.title} at {job.company}")],
    }


def score_ats_readiness(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Calculate the deterministic Demo ATS Readiness Score for the original resume."""
    job = state.get("selected_job")
    resume = state.get("resume")
    if job is None or resume is None:
        return {
            "errors": ["ATS readiness could not be scored without a selected job."],
            "progress_events": [event("Resume scored for ATS readiness", "error")],
        }
    if len(job.description.strip()) < 200:
        return {
            "errors": [
                "The selected job description is too short to assess ATS readiness."
            ],
            "progress_events": [event("Resume scored for ATS readiness", "error")],
        }

    assessment = assess_ats(
        job,
        resume,
        resume_version="original",
        now=deps.now(),
        threshold=deps.settings.ats_recommendation_threshold,
    )
    return {
        "ats_assessment": assessment,
        "ats_recommendations": list(assessment.recommendations),
        "progress_events": [
            event(
                "Selected resume scored for ATS readiness",
                detail=f"{assessment.total_score}/100 — {BAND_LABELS[assessment.band]}",
            )
        ],
    }


def recommend_ats_changes(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Publish the prioritized “what to change first” panel."""
    assessment = state.get("ats_assessment")
    if assessment is None:
        return {}
    safe = [rec for rec in assessment.recommendations if rec.safe_to_apply]
    gaps = [rec for rec in assessment.recommendations if not rec.safe_to_apply]
    warnings: list[str] = []
    if assessment.total_score < deps.settings.ats_recommendation_threshold and not safe:
        warnings.append(
            "No safe, evidence-backed changes are available for this posting. "
            "The remaining differences are genuine gaps, not wording problems."
        )
    return {
        "ats_recommendations": list(assessment.recommendations),
        "warnings": warnings,
        "progress_events": [
            event(
                "Prioritized safe changes prepared",
                detail=f"{len(safe)} safe change(s), {len(gaps)} unsupported gap(s)",
            )
        ],
    }


# --------------------------------------------------------------------------
# Drafting
# --------------------------------------------------------------------------


def draft_application(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Draft the tailored package using only safe, evidence-backed changes."""
    job = state.get("selected_job")
    resume = state.get("resume")
    assessment = state.get("ats_assessment")
    if job is None or resume is None or assessment is None:
        return {"errors": ["The application could not be drafted."]}

    safe = safe_recommendations(assessment)
    gaps = list(assessment.unsupported_job_gaps)
    draft = _generate_draft(
        job, resume, assessment, safe, gaps, deps, feedback=state.get("approval_feedback")
    )
    application = _finalize_application(draft, job, resume, assessment, safe, gaps)

    return {
        "tailored_application": application,
        "safe_ats_recommendations_applied": bool(
            application.applied_ats_recommendation_ids
        ),
        "warnings": _quota_warnings(deps),
        "progress_events": [
            event(
                "Tailored application drafted",
                detail=(
                    f"{len(application.applied_ats_recommendation_ids)} safe "
                    f"recommendation(s) applied; {len(gaps)} gap(s) refused"
                ),
            )
        ],
    }


def _generate_draft(
    job: JobPosting,
    resume: ResumeProfile,
    assessment: AtsAssessment,
    safe: list[AtsRecommendation],
    gaps: list[str],
    deps: AgentDeps,
    feedback: str | None = None,
) -> TailoredDraft:
    """Generate the draft with Gemini, or deterministically when unavailable."""
    if getattr(deps.llm, "available", False):
        prompt = render_prompt(
            "tailor_application",
            JOB_TITLE=job.title,
            COMPANY=job.company,
            JOB_DESCRIPTION=job.description[:8000],
            MASTER_RESUME=resume.as_plain_text(),
            ALLOWED_BULLET_IDS=", ".join(sorted(resume.bullet_index())),
            SAFE_RECOMMENDATIONS="\n".join(
                f"- {rec.recommendation_id} ({rec.target_section}): "
                f"{rec.recommended_change} [evidence: {', '.join(rec.evidence_resume_ids)}]"
                for rec in safe
            )
            or "- none",
            UNSUPPORTED_GAPS=", ".join(gaps) or "none",
            REVISION_FEEDBACK=feedback or "none",
        )
        draft = deps.llm.generate_structured(prompt, TailoredDraft, temperature=0.2)
        if draft is not None:
            return draft
    return _deterministic_draft(job, resume, assessment, safe, gaps)


def _deterministic_draft(
    job: JobPosting,
    resume: ResumeProfile,
    assessment: AtsAssessment,
    safe: list[AtsRecommendation],
    gaps: list[str],
) -> TailoredDraft:
    """Build a truthful draft without a model.

    Used offline and as the fallback path. It only reorders and re-states facts
    that already exist in the master resume.
    """
    supported = assessment.supported_but_missing_keywords
    summary = (
        f"{job.title} candidate. {resume.professional_summary.rstrip('.')}"
        f", working with {', '.join(resume.skills[:4])}."
    )
    summary_clause = _lower_first(resume.professional_summary.rstrip("."))

    bullets: list[RevisedBullet] = []
    index = list(resume.bullet_index().items())
    for position, (bullet_id, text) in enumerate(index[:2]):
        keyword = supported[position] if position < len(supported) else None
        revised = text if not keyword else f"{text.rstrip('.')}, including {keyword}."
        bullets.append(
            RevisedBullet(
                source_bullet_id=bullet_id, original_text=text, revised_text=revised
            )
        )

    matched = set(assessment.matched_required_keywords)
    ordered = sorted(
        resume.skills, key=lambda skill: (canonical_skill(skill) not in matched,)
    )

    # Resume bullets drop the subject; a letter needs it back.
    evidence_sentences = (
        " ".join(f"I {_lower_first(text.rstrip('.'))}." for _, text in index[:2])
        or "I support recurring analysis and reporting work."
    )

    cover_letter = (
        f"Dear Hiring Team,\n\n"
        f"I am applying for the {job.title} position at {job.company}. I am an "
        f"{summary_clause}, and that work lines up closely with what this role "
        f"asks for. {evidence_sentences} My strongest tools are "
        f"{', '.join(resume.skills[:4])}, and I use them to keep analysis "
        f"reproducible and reporting easy for a reader to follow. I would welcome "
        f"the chance to bring that same care to your team, and I am eager to keep "
        f"learning the tools your analysts rely on.\n\n"
        f"Thank you for your time and consideration.\n\n"
        f"Sincerely,\n{resume.name}"
    )

    return TailoredDraft(
        revised_summary=summary,
        revised_bullets=bullets,
        reordered_skills=ordered,
        keywords_used=sorted(matched | set(supported)),
        applied_ats_recommendation_ids=[rec.recommendation_id for rec in safe],
        unsupported_ats_gaps_not_applied=gaps,
        missing_requirements=gaps,
        cover_letter=cover_letter,
    )


def _lower_first(text: str) -> str:
    """Lowercase only the first character, preserving acronyms after it."""
    return text[:1].lower() + text[1:] if text else text


def _finalize_application(
    draft: TailoredDraft,
    job: JobPosting,
    resume: ResumeProfile,
    assessment: AtsAssessment,
    safe: list[AtsRecommendation],
    gaps: list[str],
) -> TailoredApplication:
    """Apply the deterministic guardrails the model is not trusted to enforce.

    Skill lists are constrained to the master resume, applied recommendation IDs
    are constrained to the safe set, and the unsupported-gap list is taken from
    the rubric rather than from the model.
    """
    bullet_index = resume.bullet_index()

    bullets: list[RevisedBullet] = []
    for bullet in draft.revised_bullets:
        if bullet.source_bullet_id not in bullet_index:
            continue
        bullets.append(
            RevisedBullet(
                source_bullet_id=bullet.source_bullet_id,
                original_text=bullet_index[bullet.source_bullet_id],
                revised_text=bullet.revised_text.strip()
                or bullet_index[bullet.source_bullet_id],
            )
        )
        if len(bullets) == 2:
            break
    for bullet_id, text in bullet_index.items():
        if len(bullets) >= 2:
            break
        if any(existing.source_bullet_id == bullet_id for existing in bullets):
            continue
        bullets.append(
            RevisedBullet(
                source_bullet_id=bullet_id, original_text=text, revised_text=text
            )
        )

    by_lower = {skill.lower(): skill for skill in resume.skills}
    ordered: list[str] = []
    for skill in draft.reordered_skills:
        canonical = by_lower.get(skill.strip().lower())
        if canonical and canonical not in ordered:
            ordered.append(canonical)
    for skill in resume.skills:
        if skill not in ordered:
            ordered.append(skill)

    safe_ids = {rec.recommendation_id for rec in safe}
    applied = [
        rec_id for rec_id in draft.applied_ats_recommendation_ids if rec_id in safe_ids
    ]

    return TailoredApplication(
        job_id=job.job_id,
        revised_summary=draft.revised_summary.strip()
        or resume.professional_summary,
        revised_bullets=bullets,
        reordered_skills=ordered,
        keywords_used=[kw for kw in draft.keywords_used if kw],
        applied_ats_recommendation_ids=applied,
        unsupported_ats_gaps_not_applied=gaps,
        missing_requirements=draft.missing_requirements or gaps,
        cover_letter=draft.cover_letter.strip(),
    )


def rescore_proposed_resume(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Re-score the proposed resume under the identical deterministic rubric."""
    job = state.get("selected_job")
    resume = state.get("resume")
    application = state.get("tailored_application")
    original = state.get("ats_assessment")
    if job is None or resume is None or application is None or original is None:
        return {}

    projected = rescore_proposed(
        job,
        resume,
        application,
        now=deps.now(),
        threshold=deps.settings.ats_recommendation_threshold,
    )
    warnings: list[str] = []
    if projected.total_score < original.total_score:
        warnings.append(
            f"The projected score ({projected.total_score}/100) is lower than the "
            f"original score ({original.total_score}/100). These changes are not an "
            "improvement under this rubric."
        )
    return {
        "projected_ats_assessment": projected,
        "warnings": warnings,
        "progress_events": [
            event(
                "Proposed resume re-scored under the same rubric",
                "ok" if projected.total_score >= original.total_score else "warn",
                detail=(
                    f"{original.total_score}/100 → {projected.total_score}/100 "
                    f"({BAND_LABELS[projected.band]})"
                ),
            )
        ],
    }


# --------------------------------------------------------------------------
# Validation, approval, export
# --------------------------------------------------------------------------


def validate_application(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Validate every revised claim against the fictional master resume."""
    application = state.get("tailored_application")
    resume = state.get("resume")
    assessment = state.get("ats_assessment")
    job = state.get("selected_job")
    if application is None or resume is None:
        return {"errors": ["Validation could not run without a drafted application."]}

    report = run_validation(
        application,
        resume,
        applied_recommendations=list(assessment.recommendations) if assessment else [],
        unsupported_gaps=list(assessment.unsupported_job_gaps) if assessment else [],
        allowed_organizations=[job.company] if job else [],
        llm=deps.llm,
    )
    failed = len(report.failed_checks()) + len(report.unsupported_claims)
    return {
        "validation_report": report,
        "progress_events": [
            event(
                "Truthfulness validation complete",
                "ok" if report.passed else "warn",
                detail="All checks passed"
                if report.passed
                else f"{failed} issue(s) found",
            )
        ],
    }


def revise_application(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Revise the draft once, using validation findings or human feedback."""
    report = state.get("validation_report")
    feedback_parts: list[str] = []
    if report is not None:
        for review in report.unsupported_claims:
            feedback_parts.append(f"Remove or rewrite: {review.claim} ({review.reason})")
        for check in report.failed_checks():
            feedback_parts.append(f"Fix: {check['name']} — {check['detail']}")
    if state.get("approval_feedback"):
        feedback_parts.append(f"Reviewer feedback: {state['approval_feedback']}")

    job = state.get("selected_job")
    resume = state.get("resume")
    assessment = state.get("ats_assessment")
    if job is None or resume is None or assessment is None:
        return {}

    safe = safe_recommendations(assessment)
    gaps = list(assessment.unsupported_job_gaps)
    draft = _generate_draft(
        job, resume, assessment, safe, gaps, deps, feedback="\n".join(feedback_parts)
    )
    application = _finalize_application(draft, job, resume, assessment, safe, gaps)

    return {
        "tailored_application": application,
        "revision_count": state.get("revision_count", 0) + 1,
        "approval_feedback": None,
        # Cleared so the approval node pauses again instead of replaying the
        # previous decision.
        "approval_decision": None,
        "progress_events": [
            event("Draft revised", detail=f"{len(feedback_parts)} issue(s) addressed")
        ],
    }


def human_approval(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Pause for explicit human approval before any side effect."""
    decision = state.get("approval_decision")
    if decision not in {"approve", "request_changes", "reject"}:
        job = state.get("selected_job")
        application = state.get("tailored_application")
        report = state.get("validation_report")
        projected = state.get("projected_ats_assessment")
        original = state.get("ats_assessment")
        payload = {
            "kind": "approval",
            "message": (
                "Application package ready.\n"
                "No application has been submitted.\n"
                "Waiting for human approval."
            ),
            "allowed_decisions": ["approve", "request_changes", "reject"],
            "job": job.model_dump(mode="json") if job else None,
            "application": application.model_dump(mode="json") if application else None,
            "validation": report.model_dump(mode="json") if report else None,
            "ats_original": original.model_dump(mode="json") if original else None,
            "ats_projected": projected.model_dump(mode="json") if projected else None,
        }
        response = interrupt(payload)
        if isinstance(response, dict):
            decision = response.get("decision")
            feedback = response.get("feedback")
        else:
            decision = response
            feedback = None
        return {
            "approval_decision": decision,
            "approval_feedback": feedback,
            "progress_events": [event(f"Human decision recorded: {decision}")],
        }
    return {"approval_decision": decision}


def export_package(state: CareerAgentState, deps: AgentDeps) -> dict[str, Any]:
    """Write the approved Markdown and JSON package. The only side effect."""
    resume = state.get("resume")
    job = state.get("selected_job")
    application = state.get("tailored_application")
    original = state.get("ats_assessment")
    report = state.get("validation_report")
    matches = {match.job_id: match for match in state.get("ranked_matches", [])}
    if not all([resume, job, application, original, report]):
        return {"errors": ["The package could not be exported; the run is incomplete."]}

    approved_at = deps.now()
    try:
        files = write_package(
            resume=resume,
            job=job,
            match=matches.get(job.job_id)
            or MatchResult(
                job_id=job.job_id,
                total_score=0,
                skill_score=0,
                similarity_score=0,
                role_score=0,
                experience_score=0,
                preference_score=0,
            ),
            original_ats=original,
            projected_ats=state.get("projected_ats_assessment"),
            application=application,
            validation=report,
            output_dir=deps.settings.output_path,
            approved_at=approved_at,
            data_mode=state.get("data_mode", "live"),
            resume_descriptor=deps.settings.resume_descriptor,
        )
    except OSError as exc:
        logger.error("Export failed: %s", type(exc).__name__)
        return {
            "errors": [
                "The application package could not be written. Check that the "
                "output directory is writable."
            ],
            "progress_events": [event("Application package exported", "error")],
        }

    return {
        "output_files": files,
        "approved_at": approved_at,
        "progress_events": [
            event(
                "Application package exported",
                detail="No application was submitted.",
            )
        ],
    }
