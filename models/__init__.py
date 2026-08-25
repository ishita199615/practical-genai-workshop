"""Validated data models shared by every stage of the Cougar Career Agent."""

from models.application import RevisedBullet, TailoredApplication, TailoredDraft
from models.ats import (
    ATS_DISCLAIMER,
    ATS_LONG_DISCLAIMER,
    BAND_LABELS,
    PRIORITY_ORDER,
    AtsAssessment,
    AtsBand,
    AtsPriority,
    AtsRecommendation,
    AtsRecommendationCategory,
)
from models.job import (
    FRESHNESS_LABELS,
    SOURCE_CATEGORY_LABELS,
    ExtractedJobFields,
    FreshnessEvidence,
    FreshnessStatus,
    FreshnessWindow,
    JobPosting,
    RawJobResult,
    SourceCategory,
    WorkMode,
)
from models.match import MatchResult
from models.resume import (
    ExperienceEntry,
    ResumeBullet,
    ResumeProfile,
    render_resume_text,
)
from models.validation import ClaimReview, ClaimReviewBatch, ValidationReport

__all__ = [
    "ATS_DISCLAIMER",
    "ATS_LONG_DISCLAIMER",
    "BAND_LABELS",
    "FRESHNESS_LABELS",
    "PRIORITY_ORDER",
    "SOURCE_CATEGORY_LABELS",
    "AtsAssessment",
    "AtsBand",
    "AtsPriority",
    "AtsRecommendation",
    "AtsRecommendationCategory",
    "ClaimReview",
    "ClaimReviewBatch",
    "ExperienceEntry",
    "ExtractedJobFields",
    "FreshnessEvidence",
    "FreshnessStatus",
    "FreshnessWindow",
    "JobPosting",
    "MatchResult",
    "RawJobResult",
    "ResumeBullet",
    "ResumeProfile",
    "RevisedBullet",
    "SourceCategory",
    "TailoredApplication",
    "TailoredDraft",
    "ValidationReport",
    "WorkMode",
    "render_resume_text",
]
