# Cougar Career Agent — Demo 2

## Purpose of this file

This file is the source of truth for building **Demo 2: The Job Hunter Agent** for the University of Houston System's 60-minute Practical Generative AI workshop.

Build the smallest reliable application that clearly demonstrates this agent pattern:

```text
Retrieve current information
→ use deterministic tools
→ generate a tailored artifact
→ validate the result
→ pause for human approval
```

The complete Demo 2 presentation must fit in approximately **8 minutes** because the workshop allocates 22 minutes to three agent demonstrations.

---

## 1. Project objective

Build a Streamlit application called **Cougar Career Agent** that:

1. Loads a fictional sample resume.
2. Accepts a target role, location, work-mode preference, **freshness window** (Last 1 hour, Last 24 hours, Last 3 days, or Last 7 days), and **job source/query category**.
3. Searches current public job pages with Firecrawl.
4. Cleans, validates, classifies, and deduplicates the retrieved postings.
5. Calculates an explainable job-match score in Python.
6. Ranks and displays the top three jobs with both the selected query category and the actual detected source.
7. Shows a validated, clickable **Open job posting ↗** link for every result so the instructor can inspect the original public page.
8. Shows an expandable, cleaned **full job description** for every result, plus responsibilities, required skills, preferred skills, and the original source link.
9. Lets the user select one job.
10. Calculates a separate deterministic **Demo ATS Readiness Score** for the fictional resume against the selected job description.
11. When the ATS readiness score is below 80, shows prioritized, section-specific changes that can improve the resume without inventing experience or skills.
12. Generates a small, truthful resume patch and short cover letter with Gemini using only safe, evidence-backed recommendations.
13. Re-scores the proposed resume patch and displays a clearly labeled projected score under the same demo rubric.
14. Validates every revised claim against the fictional master resume.
15. Pauses for explicit human approval.
16. Exports a Markdown and JSON application package after approval.

The final state must clearly say:

```text
Application package ready.
No application has been submitted.
Waiting for human approval.
```

---

## 2. Core teaching message

The UI and code should make this separation obvious:

```text
Firecrawl retrieves current public information.
Python calculates the Job Match Score and ATS Readiness Score.
Gemini extracts, explains, recommends, and drafts.
The validator checks truthfulness.
A human approves the result.
```

The two scores serve different purposes:

- **Demo Job Match Score** ranks multiple jobs by candidate fit.
- **Demo ATS Readiness Score** evaluates how well the current resume is likely to be parsed and matched against one selected job description under this project's transparent rubric.

Neither score may be invented or changed by the LLM. The ATS readiness score is an educational estimate, not the score of any employer's proprietary applicant-tracking system and not a guarantee of interview selection.

The language model is only one component of the agent. The agent also includes tools, state, routing, validation, and permissions.

---

## 3. Non-negotiable workshop constraints

### Build for the live demo

- Synthetic applicant data only.
- Public job listings only.
- Maximum eight search results.
- Top three ranked results shown.
- Freshness choices are **Last 1 hour**, **Last 24 hours**, **Last 3 days**, and **Last 7 days**; the live-demo default is Last 24 hours for reliability.
- A **Job source / query category** selector with: All Public Sources, LinkedIn, Indeed, Google Jobs / Web, and Direct Company Careers.
- Direct Company Careers is the default category for live-demo reliability.
- Visible query category, detected source category, requested freshness window, posting timestamp/date evidence, and retrieval timestamp for every result.
- A validated clickable **Open job posting ↗** button for every displayed job, backed by the final canonical public job URL.
- An expandable cleaned full job description for every displayed result; do not show only a short search snippet.
- Deterministic Job Match and ATS Readiness scoring in Python.
- A visible ATS disclaimer: “Estimated using this demo rubric; not an official employer ATS score.”
- A score band of Strong, Needs Targeted Changes, or Low.
- When ATS readiness is below 80, a prioritized “What to change first” panel with exact resume sections, evidence IDs, and safe/unsafe labels.
- No keyword stuffing, hidden text, false titles, invented metrics, or unsupported skills.
- One revised professional summary.
- Two revised experience bullets.
- One reordered skills list.
- One cover letter of approximately 100–140 words.
- Claim-level truthfulness validation.
- Human approval with **Approve**, **Request Changes**, and **Reject** choices.
- Clearly labeled live and cached modes.
- Complete workflow target: under 60 seconds on the instructor machine.

### Do not build for the live demo

- No real student resume uploads.
- No UHS-protected or confidential data.
- No LinkedIn login.
- No job-board credentials.
- No CAPTCHA handling.
- No automatic job submission.
- No application-form browser automation.
- No email sending.
- No demographic-question completion.
- No work-authorization decisions.
- No user accounts.
- No production database.
- No multi-agent conversation architecture.
- No full resume redesign.
- No DOCX or PDF rendering in the MVP.
- No self-hosting Firecrawl during the workshop.
- No Docker requirement for the first working version.

Do not add excluded features unless explicitly requested after the MVP is stable.

---

## 4. Technology stack

Use Python 3.11 or newer.

| Layer | Tool | Responsibility |
|---|---|---|
| User interface | Streamlit | Inputs, progress, job cards, comparison, validation, approval |
| Workflow | LangGraph | State, nodes, conditional routing, pause/resume |
| Live web retrieval | Firecrawl Search API v2 | Find and scrape current public job pages |
| LLM | Gemini via `google-genai` | Structured extraction, match explanation, resume patch, cover letter, claim review |
| Validation | Pydantic | Validate all structured inputs and outputs |
| Scoring | Python, scikit-learn, RapidFuzz | Repeatable Job Match and ATS Readiness scores plus deterministic re-scoring |
| Data handling | pandas | Tabular ranking and presentation |
| Configuration | python-dotenv | Local secrets and runtime settings |
| Testing | pytest | Unit and integration tests |

### Provider boundaries

- Use Firecrawl Cloud for the live workshop because it is simpler and more reliable than self-hosting during an eight-minute demo.
- Keep Firecrawl behind a small adapter so a self-hosted base URL can be added later.
- Keep Gemini behind an `LLMClient` interface so a local Ollama provider can be added later without changing the graph.
- Do not hard-code a preview Gemini model. Read the model ID from `GEMINI_MODEL`.

---

## 5. Repository structure

Create this structure:

```text
cougar-career-agent/
│
├── SPECIFICATION.md
├── README.md
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   ├── state.py
│   ├── nodes.py
│   └── routing.py
│
├── models/
│   ├── __init__.py
│   ├── resume.py
│   ├── job.py
│   ├── match.py
│   ├── ats.py
│   ├── application.py
│   └── validation.py
│
├── tools/
│   ├── __init__.py
│   ├── firecrawl_search.py
│   ├── job_normalizer.py
│   ├── job_filter.py
│   ├── job_scorer.py
│   ├── ats_scorer.py
│   ├── claim_validator.py
│   └── exporter.py
│
├── services/
│   ├── __init__.py
│   ├── llm_interface.py
│   └── gemini_client.py
│
├── prompts/
│   ├── extract_job.md
│   ├── explain_match.md
│   ├── explain_ats.md
│   ├── tailor_application.md
│   └── validate_claims.md
│
├── data/
│   ├── sample_resume.json
│   ├── cached_jobs.json
│   └── expected_demo_output.json
│
├── output/
│   └── .gitkeep
│
└── tests/
    ├── test_job_filter.py
    ├── test_deduplication.py
    ├── test_scoring.py
    ├── test_ats_scoring.py
    ├── test_ats_recommendations.py
    ├── test_claim_validator.py
    ├── test_cached_fallback.py
    └── test_graph_approval.py
```

Keep modules small and explicit. Avoid unnecessary abstractions.

---

## 6. Environment configuration

Create `.env.example` with:

```text
FIRECRAWL_API_KEY=
FIRECRAWL_BASE_URL=https://api.firecrawl.dev
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
DEMO_MODE=live
DEFAULT_SOURCE_CATEGORY=company_careers
DEFAULT_FRESHNESS_WINDOW=last_24_hours
MAX_JOB_RESULTS=8
MAX_JOB_DESCRIPTION_CHARS=20000
ATS_RECOMMENDATION_THRESHOLD=80
SEARCH_TIMEOUT_SECONDS=10
CACHE_FILE=data/cached_jobs.json
OUTPUT_DIR=output
```

Rules:

- Never commit `.env`.
- Never print API keys.
- Never place API keys in Streamlit state or rendered HTML.
- Validate required variables at startup.
- Allow `DEMO_MODE` values of `live`, `cached`, or `auto`.
- Allow `DEFAULT_SOURCE_CATEGORY` values of `all`, `linkedin`, `indeed`, `google_jobs`, or `company_careers`.
- Allow `DEFAULT_FRESHNESS_WINDOW` values of `last_hour`, `last_24_hours`, `last_3_days`, or `last_7_days`.
- Treat the source category as a search filter, not as proof of the actual result source; classify each result from its final URL.
- In `auto` mode, try Firecrawl first and fall back to cache on timeout or error.

---

## 7. Required Python dependencies

Create a minimal `requirements.txt`:

```text
streamlit
langgraph
firecrawl-py
google-genai
pydantic
pandas
scikit-learn
rapidfuzz
python-dotenv
requests
pytest
```

Pin versions only after the first successful build and test run.

---

## 8. Synthetic sample resume

Create `data/sample_resume.json` using fictional data. Use stable IDs for every experience and bullet.

```json
{
  "candidate_id": "demo_candidate_001",
  "name": "Alex Morgan",
  "email": "alex.morgan@example.com",
  "phone": "555-010-2026",
  "location": "Houston, TX",
  "professional_summary": "Information systems student with hands-on experience analyzing survey data and building recurring dashboards.",
  "target_roles": [
    "Data Analyst Intern",
    "Junior Data Analyst",
    "Business Intelligence Intern"
  ],
  "skills": [
    "Python",
    "SQL",
    "Excel",
    "Tableau",
    "Pandas",
    "Statistics"
  ],
  "education": [
    {
      "id": "education_1",
      "degree": "Bachelor of Science in Information Systems",
      "institution": "Example University",
      "graduation": "May 2027"
    }
  ],
  "experience": [
    {
      "id": "experience_1",
      "title": "Student Data Assistant",
      "organization": "Example Research Lab",
      "dates": "August 2025–Present",
      "bullets": [
        {
          "id": "experience_1_bullet_1",
          "text": "Analyzed survey data using Python and Excel."
        },
        {
          "id": "experience_1_bullet_2",
          "text": "Created Tableau dashboards for weekly reporting."
        }
      ]
    }
  ],
  "projects": [
    {
      "id": "project_1",
      "name": "Campus Event Attendance Analysis",
      "bullets": [
        {
          "id": "project_1_bullet_1",
          "text": "Used SQL and Pandas to clean and summarize fictional event attendance data."
        }
      ]
    }
  ],
  "document_features": {
    "source_format": "synthetic_plain_text",
    "parse_success": true,
    "single_column": true,
    "standard_headings": true,
    "uses_tables_for_layout": false,
    "uses_text_boxes": false,
    "uses_images_for_critical_text": false
  }
}
```

Do not use a real UH logo, real student name, real student email, real course record, or real department data in the applicant profile.

---

## 9. Pydantic data models

Implement explicit Pydantic models.

### Resume models

```python
class ResumeBullet(BaseModel):
    id: str
    text: str

class ExperienceEntry(BaseModel):
    id: str
    title: str
    organization: str
    dates: str | None = None
    bullets: list[ResumeBullet]

class ResumeProfile(BaseModel):
    candidate_id: str
    name: str
    email: str
    phone: str | None = None
    location: str
    professional_summary: str
    target_roles: list[str]
    skills: list[str]
    education: list[dict]
    experience: list[ExperienceEntry]
    projects: list[dict] = []
    document_features: dict = {}
```

### Job models

```python
SourceCategory = Literal[
    "all",
    "linkedin",
    "indeed",
    "google_jobs",
    "company_careers",
    "other",
]

FreshnessWindow = Literal[
    "last_hour",
    "last_24_hours",
    "last_3_days",
    "last_7_days",
]

FreshnessEvidence = Literal[
    "exact_timestamp",
    "date_only",
    "search_filter_only",
    "unavailable",
]

class RawJobResult(BaseModel):
    title: str | None = None
    description: str | None = None
    url: str  # original URL returned by Firecrawl
    final_url: str | None = None  # resolved URL after redirects, when available
    markdown: str | None = None
    metadata: dict = {}
    query_category: SourceCategory = "all"
    freshness_window: FreshnessWindow = "last_24_hours"
    detected_source_category: SourceCategory = "other"
    retrieved_at: datetime

class JobPosting(BaseModel):
    job_id: str
    title: str
    company: str
    location: str | None = None
    work_mode: Literal["remote", "hybrid", "onsite", "unknown"] = "unknown"
    query_category: SourceCategory
    source_category: SourceCategory
    source_label: str
    description: str  # cleaned full job description, not merely a search snippet
    description_excerpt: str
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    minimum_experience_years: float | None = None
    education_requirement: str | None = None
    responsibilities: list[str] = []
    source_url: str  # final canonical direct job URL rendered as a clickable link
    original_source_url: str | None = None
    apply_url: str | None = None  # optional; only when explicitly present on the public page
    source_domain: str
    freshness_window: FreshnessWindow
    posted_at: datetime | None = None  # exact source timestamp when available
    posting_date: date | None = None  # date-only fallback
    posting_age_hours: float | None = None
    freshness_evidence: FreshnessEvidence = "unavailable"
    retrieved_at: datetime
    is_closed: bool = False
    freshness_status: Literal[
        "verified_recent",
        "date_unavailable",
        "possibly_stale"
    ]
```

### Match models

```python
class MatchResult(BaseModel):
    job_id: str
    total_score: int
    skill_score: int
    similarity_score: int
    role_score: int
    experience_score: int
    preference_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    concerns: list[str]
    explanation: str | None = None
```

### ATS readiness models

```python
AtsBand = Literal["strong", "needs_targeted_changes", "low"]
AtsPriority = Literal["high", "medium", "low"]
AtsRecommendationCategory = Literal[
    "keyword_alignment",
    "summary",
    "skills",
    "experience",
    "education",
    "section_completeness",
    "format_and_parseability",
    "unsupported_gap",
]

class AtsRecommendation(BaseModel):
    recommendation_id: str
    priority: AtsPriority
    category: AtsRecommendationCategory
    target_section: str
    current_text: str | None = None
    recommended_change: str
    reason: str
    evidence_resume_ids: list[str] = []
    safe_to_apply: bool
    projected_effect: Literal["high", "medium", "low"]

class AtsAssessment(BaseModel):
    job_id: str
    resume_version: Literal["original", "proposed"]
    total_score: int
    band: AtsBand
    keyword_score: int
    qualification_score: int
    evidence_score: int
    section_score: int
    structure_score: int
    contact_score: int
    matched_required_keywords: list[str]
    supported_but_missing_keywords: list[str]
    unsupported_job_gaps: list[str]
    recommendations: list[AtsRecommendation]
    disclaimer: str = (
        "Estimated using this demo rubric; not an official employer ATS score."
    )
```

### Tailoring models

```python
class RevisedBullet(BaseModel):
    source_bullet_id: str
    original_text: str
    revised_text: str

class TailoredApplication(BaseModel):
    job_id: str
    revised_summary: str
    revised_bullets: list[RevisedBullet]
    reordered_skills: list[str]
    keywords_used: list[str]
    applied_ats_recommendation_ids: list[str]
    unsupported_ats_gaps_not_applied: list[str]
    missing_requirements: list[str]
    cover_letter: str
```

### Validation models

```python
class ClaimReview(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "unclear"]
    supporting_resume_ids: list[str]
    reason: str

class ValidationReport(BaseModel):
    valid_source_ids: bool
    unsupported_claims: list[ClaimReview]
    unclear_claims: list[ClaimReview]
    passed: bool
```

Do not pass unvalidated dictionaries between graph nodes.

---

## 10. LangGraph state

Create a typed graph state similar to:

```python
class CareerAgentState(TypedDict, total=False):
    thread_id: str
    role: str
    location: str
    work_mode: str
    freshness_window: FreshnessWindow
    freshness_tbs: str
    freshness_cutoff_utc: datetime
    query_category: SourceCategory
    source_domains: list[str]
    search_query: str
    resume: ResumeProfile
    raw_jobs: list[RawJobResult]
    normalized_jobs: list[JobPosting]
    ranked_matches: list[MatchResult]
    selected_job_id: str | None
    selected_job: JobPosting | None
    ats_assessment: AtsAssessment | None
    ats_recommendations: list[AtsRecommendation]
    projected_ats_assessment: AtsAssessment | None
    safe_ats_recommendations_applied: bool
    tailored_application: TailoredApplication | None
    validation_report: ValidationReport | None
    revision_count: int
    approval_decision: str | None
    approval_feedback: str | None
    data_mode: str
    retrieval_timestamp: datetime | None
    output_files: list[str]
    progress_events: list[dict]
    warnings: list[str]
    errors: list[str]
```

Use JSON-serializable values in interrupt payloads.

---

## 11. Graph design

Implement one controlled graph, not multiple autonomous agents.

```text
START
  ↓
load_sample_resume
  ↓
build_search_query
  ↓
search_current_jobs
  ↓
normalize_jobs
  ↓
filter_and_deduplicate_jobs
  ↓
score_jobs
  ↓
explain_top_matches
  ↓
wait_for_job_selection
  ↓
score_ats_readiness
  ↓
ATS score below 80?
  ├── yes → recommend_ats_changes ─┐
  └── no  → optional refinements  ─┤
                                    ↓
                              draft_application
                                    ↓
                         rescore_proposed_resume
                                    ↓
                           validate_application
  ↓
unsupported claims?
  ├── yes and revision_count < 1
  │       ↓
  │   revise_application
  │       ↓
  │   validate_application
  │
  └── no or revision limit reached
          ↓
      human_approval
        ├── approve → export_package → END
        ├── request_changes → revise_from_feedback → validate_application
        └── reject → END
```

### Required nodes

- `load_sample_resume`
- `build_search_query`
- `search_current_jobs`
- `normalize_jobs`
- `filter_and_deduplicate_jobs`
- `score_jobs`
- `explain_top_matches`
- `select_job`
- `score_ats_readiness`
- `recommend_ats_changes`
- `draft_application`
- `rescore_proposed_resume`
- `validate_application`
- `revise_application`
- `human_approval`
- `export_package`

### Human approval

Use LangGraph checkpointing and an interrupt-based approval node.

- Use `InMemorySaver` for this workshop prototype.
- Generate a stable `thread_id` per Streamlit session.
- Resume with the same thread ID.
- Approval payload must include the selected job, tailored content, validation summary, and allowed decisions.
- Side effects must occur only after approval.
- Export is the only post-approval side effect.

---

## 12. Firecrawl search implementation

Use Firecrawl Search API v2. Search and scrape in one request.

### Search defaults

```text
Role: Data Analyst Intern
Location: Houston, TX
Work mode: Houston or Remote
Job source / query category: Direct Company Careers
Freshness window: Last 24 hours
Maximum results: 8
Country: US
Time filter: sbd:1,qdr:d
```

### Preferred job domains

Start with direct applicant-tracking-system domains:

```text
boards.greenhouse.io
job-boards.greenhouse.io
jobs.lever.co
jobs.ashbyhq.com
jobs.smartrecruiters.com
careers.workday.com
```

Do not rely on LinkedIn as the only source.

### Job source / query categories

Expose these user-facing choices:

```text
All Public Sources
LinkedIn — public job pages only
Indeed — public job pages only
Google Jobs / Web
Direct Company Careers
```

Map them internally as follows:

| UI label | Internal value | Query/domain behavior |
|---|---|---|
| All Public Sources | `all` | Broad web query; no forced single domain. Prefer direct opening URLs and classify the actual source from the final URL. |
| LinkedIn — public job pages only | `linkedin` | Add `site:linkedin.com/jobs/view` to the query and use `linkedin.com` as the domain filter when supported. Never log in or bypass access controls. |
| Indeed — public job pages only | `indeed` | Add `site:indeed.com/viewjob` to the query and use `indeed.com` as the domain filter when supported. |
| Google Jobs / Web | `google_jobs` | Use a broad web query oriented to Google-indexed job pages. Do not claim the result is from Google Jobs unless the final URL or metadata supports that label. |
| Direct Company Careers | `company_careers` | Restrict to the preferred ATS and employer-career domains listed above. This is the workshop default. |

### Freshness windows

Expose a separate freshness selector. Query category and freshness are independent controls and must be combined by the query builder.

| UI label | Internal value | Firecrawl `tbs` | Local verification cutoff |
|---|---|---|---|
| Last 1 hour | `last_hour` | `sbd:1,qdr:h` | `retrieved_at - 1 hour` |
| Last 24 hours | `last_24_hours` | `sbd:1,qdr:d` | `retrieved_at - 24 hours` |
| Last 3 days | `last_3_days` | `sbd:1,cdr:1,cd_min:<MM/DD/YYYY>,cd_max:<MM/DD/YYYY>` | `retrieved_at - 72 hours` |
| Last 7 days | `last_7_days` | `sbd:1,qdr:w` | `retrieved_at - 168 hours` |

The current Firecrawl Search API v2 supports `qdr:h`, `qdr:d`, `qdr:w`, custom date ranges, and `sbd:1`. Generate custom dates at runtime in UTC; do not hard-code them.

Freshness honesty rules:

- A search filter narrows search results but is not itself proof of the exact posting time.
- Label a result **Verified in last hour** only when the source exposes an exact timestamp and `posted_at >= cutoff`.
- When only a date is available, label it **Date shown; exact time unavailable**.
- When no posting date/time is available, label it **Search-filtered; source timestamp unavailable**.
- If the user selected Last 1 hour and zero verifiable results remain, show an explicit **Expand to Last 24 hours** action that preserves the selected query category. Never expand silently.

Rules:

- The selected query category controls how the search is formed.
- The **actual source category** must be detected from the final result URL, not copied blindly from the selected filter.
- Show both values when useful: `Query category: LinkedIn` and `Actual source: LinkedIn`.
- A result found during an `all` search may still be classified as LinkedIn, Indeed, Google Jobs, Direct Company Careers, or Other.
- If LinkedIn blocks full-page retrieval, keep the source URL only when verifiable and do not fabricate a job description. Prefer returning a clear warning or using the visibly labeled cache.
- Never broaden a user-selected category silently. Offer the user a choice to retry with All Public Sources or Direct Company Careers.

### Query examples

```text
all:
("data analyst intern" OR "entry level data analyst") (Houston OR remote)

linkedin:
site:linkedin.com/jobs/view ("data analyst intern" OR "entry level data analyst") (Houston OR remote)

indeed:
site:indeed.com/viewjob ("data analyst intern" OR "entry level data analyst") (Houston OR remote)

google_jobs:
("data analyst intern" OR "entry level data analyst") (Houston OR remote) jobs

company_careers:
("data analyst intern" OR "entry level data analyst") (Houston OR remote)
```

### Category + freshness examples

```text
LinkedIn + Last 1 hour:
query = site:linkedin.com/jobs/view ("data analyst intern" OR "entry level data analyst") (Houston OR remote)
includeDomains = ["linkedin.com"]
tbs = sbd:1,qdr:h

Indeed + Last 24 hours:
query = site:indeed.com/viewjob ("data analyst intern" OR "entry level data analyst") (Houston OR remote)
includeDomains = ["indeed.com"]
tbs = sbd:1,qdr:d

Direct Company Careers + Last 24 hours:
query = ("data analyst intern" OR "entry level data analyst") (Houston OR remote)
includeDomains = [preferred ATS/company-career domains]
tbs = sbd:1,qdr:d
```

### Request behavior

The request should approximately contain:

```json
{
  "query": "data analyst internship OR entry level data analyst Houston remote",
  "queryCategory": "company_careers",
  "freshnessWindow": "last_24_hours",
  "limit": 8,
  "sources": ["web"],
  "includeDomains": [
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "jobs.smartrecruiters.com"
  ],
  "tbs": "sbd:1,qdr:d",
  "location": "Houston,Texas,United States",
  "country": "US",
  "safe": true,
  "timeout": 10000,
  "ignoreInvalidURLs": true,
  "scrapeOptions": {
    "formats": ["markdown"],
    "onlyMainContent": true
  }
}
```

`queryCategory` and `freshnessWindow` are application metadata for the adapter and should not be sent to Firecrawl unless the current SDK explicitly supports them. The adapter must translate the category into a query/domain filter and the freshness window into the correct `tbs` value.

Keep the actual request implementation isolated in `tools/firecrawl_search.py`.

### Firecrawl result rules

For every result:

- Preserve the original URL and resolve redirects when possible.
- Validate the final URL as public `http` or `https`; reject `javascript:`, `data:`, malformed, or credential-bearing URLs.
- Canonicalize the final direct opening URL and save it as `source_url`.
- Preserve the raw result URL separately as `original_source_url`.
- Extract `apply_url` only when the public page explicitly exposes a direct application link; never guess one.
- Record `retrieved_at` locally in UTC.
- Preserve returned Markdown.
- Preserve the complete cleaned job-description text used for matching.
- Create a short deterministic `description_excerpt` for cards, but retain the full description for the expandable details view.
- Record the selected `query_category` and `freshness_window`.
- Detect `source_category` and a human-readable `source_label` from the final URL/domain.
- Parse an exact `posted_at` timestamp when present; otherwise preserve a date-only `posting_date` or mark freshness evidence unavailable.
- Compute `posting_age_hours` in Python from source evidence, never from LLM guesswork.
- Preserve useful metadata.
- Reject missing or invalid URLs.
- Reject generic career homepages without a specific opening.
- Reject empty pages.
- Reject closed or expired jobs.
- Do not infer a posting date when none is present.
- Do not claim a result is recent solely because the search request used a time filter.
- Render the validated `source_url` with `st.link_button("Open job posting ↗", source_url)` and also show a Markdown link in exported files.
- If the direct link is unavailable or fails validation, do not display a fake button; mark the record unusable for the live ranking.

### Fallback behavior

In `auto` mode:

1. Try the live Firecrawl request.
2. Stop waiting after `SEARCH_TIMEOUT_SECONDS`.
3. On timeout, API error, zero valid jobs, or invalid response, load `data/cached_jobs.json`.
4. Add a visible warning.
5. Never silently represent cached data as live.
6. If the selected freshness is Last 1 hour and zero valid results remain, show **Expand to Last 24 hours**. Keep role, location, work mode, and query category unchanged, and run only after the user clicks.

UI labels:

```text
🟢 LIVE PUBLIC RESULTS
Retrieved at: <timestamp>
```

or:

```text
🟡 CACHED DEMONSTRATION RESULTS
Originally retrieved at: <timestamp>
Live retrieval was unavailable.
```

---

## 13. Job normalization with Gemini

Use Gemini structured output to convert each Firecrawl page into a `JobPosting` while preserving a cleaned full job description for display and scoring.

The extraction instruction must state:

```text
Extract only information explicitly present in SOURCE_CONTENT.

Do not invent or infer a posting date.
Do not infer a company from unrelated navigation content.
Do not convert preferred qualifications into required qualifications.
Use null when information is unavailable.
Set is_closed=true when the page says the position is unavailable,
filled, expired, or no longer accepting applications.
Return the complete main job-description content in `description`; do not replace it with a brief summary or search snippet.
```

Requirements:

- Use `google-genai`.
- Use Pydantic response schemas.
- Set JSON response format.
- Validate the response before saving it.
- Retry once on invalid structured output.
- Reject the record after the second failure.
- Keep temperature low where the selected model supports it.
- Do not duplicate the entire JSON schema as prose in the prompt.
- Compute `description_excerpt` in Python from the cleaned description; do not spend an LLM call on it.
- Classify the actual source from the URL in Python using explicit domain rules. Use Gemini only when the URL is truly ambiguous.
- Preserve the selected query category separately from the detected source category.

Recommended source-classification rules:

```python
SOURCE_DOMAIN_RULES = {
    "linkedin.com": ("linkedin", "LinkedIn"),
    "indeed.com": ("indeed", "Indeed"),
    "google.com": ("google_jobs", "Google Jobs / Web"),
    "boards.greenhouse.io": ("company_careers", "Greenhouse"),
    "job-boards.greenhouse.io": ("company_careers", "Greenhouse"),
    "jobs.lever.co": ("company_careers", "Lever"),
    "jobs.ashbyhq.com": ("company_careers", "Ashby"),
    "jobs.smartrecruiters.com": ("company_careers", "SmartRecruiters"),
    "careers.workday.com": ("company_careers", "Workday Careers"),
}
```

Process no more than eight pages.

---

## 14. Filtering and deduplication

Perform filtering in Python.

Reject a posting when:

- `is_closed` is true.
- Title or company is missing.
- Description is missing or too short to score meaningfully.
- Only a search-result snippet is available and a full public job description could not be retrieved.
- URL is a duplicate.
- Page is a generic job-search page rather than a specific opening.
- Explicit posting date is older than the requested age.
- Source indicates that applications are closed.

Deduplicate using a normalized key:

```text
normalized company
+ normalized title
+ normalized location
```

Also deduplicate exact and canonicalized URLs.

Posting freshness logic:

```text
Exact timestamp present and within selected cutoff → verified_recent + exact_timestamp
Date-only value inside the selected calendar range → verified_recent + date_only
No source date/time but returned by the requested Firecrawl time filter → date_unavailable + search_filter_only
Exact timestamp/date outside the selected cutoff → reject
Ambiguous or conflicting source evidence → possibly_stale
```

For **Last 1 hour**, only `exact_timestamp` evidence may receive a “Verified in last hour” badge. A date-only or missing timestamp may still be shown as search-filtered, but it must not be described as verified within one hour.

The UI must show the difference between requested freshness, source posting evidence, and retrieval time.

---

## 15. Deterministic scoring

The LLM must not determine the numeric score.

Use this 100-point formula:

```text
45 points — Required-skill coverage
20 points — Resume/job text similarity
15 points — Role-title alignment
10 points — Experience alignment
10 points — Location and work-mode alignment
```

Equivalent normalized formula:

```python
total_score = round(
    0.45 * skill_coverage
    + 0.20 * text_similarity
    + 0.15 * role_alignment
    + 0.10 * experience_alignment
    + 0.10 * preference_alignment
)
```

Each component must be between 0 and 100.

### Skill matching

- Normalize skill names to lowercase canonical forms.
- Use a small alias map, for example:

```python
{
    "ms excel": "excel",
    "microsoft excel": "excel",
    "postgresql": "sql",
    "structured query language": "sql",
    "data viz": "data visualization",
    "powerbi": "power bi"
}
```

- Use exact canonical matching first.
- Use RapidFuzz only for conservative near-matches.
- Do not treat Tableau as Power BI.
- Do not treat Python as every Python library.

### Text similarity

- Build a plain-text resume representation.
- Use TF-IDF cosine similarity.
- Keep this score explainable and deterministic.

### Role alignment

Compare normalized target titles with the job title using token overlap and RapidFuzz.

### Experience alignment

Use simple rules based on explicitly extracted minimum years and the fictional resume. Do not fabricate years of experience.

### Preference alignment

Use location and work-mode compatibility.

### Output label

Call the result:

```text
Demo Job Match Score
```

Never call it an official ATS score.

---


## 16. Demo ATS Readiness scoring and improvement guidance

The application must calculate a second score after the user selects a job. This score evaluates the fictional resume against the selected full job description.

### Naming and disclaimer

Call it:

```text
Demo ATS Readiness Score
```

Always display:

```text
Estimated using this demo rubric; not an official employer ATS score.
Different employers and applicant-tracking systems use different proprietary rules.
```

Do not claim that the score predicts interview selection, recruiter behavior, or a real ATS outcome.

### Deterministic 100-point rubric

The LLM must not calculate or modify the score.

```text
40 points — Required keyword and skill coverage
20 points — Required qualification alignment
15 points — Evidence and specificity in experience/project bullets
10 points — Standard section completeness
10 points — ATS-safe text structure and parseability
 5 points — Contact and application essentials
```

Equivalent normalized formula:

```python
ats_total_score = round(
    0.40 * keyword_coverage
    + 0.20 * qualification_alignment
    + 0.15 * evidence_quality
    + 0.10 * section_completeness
    + 0.10 * structure_parseability
    + 0.05 * contact_completeness
)
```

Each component must be between 0 and 100.

### Score bands

```text
80–100 → Strong
65–79  → Needs Targeted Changes
0–64   → Low
```

Use neutral language. Never say that a resume “will pass” or “will fail” an ATS.

### Component rules

#### Required keyword and skill coverage

- Use required skills and explicit job-description terms extracted from the selected posting.
- Count a keyword as present only when it appears in the resume or maps through the conservative alias dictionary.
- Separate `supported_but_missing_keywords` from `unsupported_job_gaps`.
- A supported-but-missing keyword may be added only when the master resume contains evidence for the concept.
- An unsupported gap must never be inserted into the resume.
- Do not reward keyword repetition or stuffing.

#### Required qualification alignment

- Compare explicit education, experience, certification, and work requirement fields.
- Preferred qualifications must not be scored as required qualifications.
- Unknown requirements must not be treated as failures.
- Never fabricate experience years, degrees, certifications, or work authorization.

#### Evidence and specificity

- Reward matched skills that are supported by experience or project bullet IDs, not merely listed in the skills section.
- Reward clear action + task/context wording.
- Do not require numeric metrics when the master resume has no verified metric.
- Never recommend inventing numbers to increase the score.

#### Section completeness

Check for standard, parseable sections such as:

```text
Contact information
Professional summary
Skills
Education
Experience
Projects, when present
```

Missing optional sections must not be treated as mandatory.

#### ATS-safe structure and parseability

Use the synthetic `document_features` metadata and the canonical plain-text rendering.

Reward:

- Successful text parsing.
- Standard headings.
- Single-column structure.
- No layout tables.
- No text boxes.
- No images containing critical text.

Because the live demo uses a structured fictional resume rather than a real uploaded file, label this component as a simulation of ATS-safe structure.

#### Contact and application essentials

Check only for reasonable presence of name, email, phone, and location. Do not score protected or demographic information.

### “What to change first” recommendations

When the original ATS readiness score is below 80, display prioritized recommendations.

- Score 65–79: show up to three targeted changes.
- Score below 65: show up to five targeted changes.
- Score 80 or above: show no more than two optional refinements.

Every recommendation must include:

```text
Priority
Target resume section
Current text, when applicable
Exact recommended change
Why it matters for this job
Supporting resume IDs
Safe to apply: Yes or No
Expected effect: High, Medium, or Low
```

Recommendations must be specific, for example:

```text
HIGH · Skills
Add the phrase “data visualization” next to Tableau.
Evidence: experience_1_bullet_2
Safe to apply: Yes
Reason: The selected job uses “data visualization,” and the resume already proves this through Tableau dashboard work.
```

Unsupported-gap example:

```text
HIGH · Missing qualification
The job requests Power BI, but the master resume contains no Power BI evidence.
Safe to apply: No
Action: Do not add Power BI. Keep it in the learning-gap list.
```

Never recommend:

- Hidden white text.
- Keyword stuffing.
- Copying the entire job description.
- Fake titles.
- Fake metrics.
- Unsupported tools or certifications.
- Removing truthful experience merely to mimic a posting.

### Safe-change drafting and projected re-score

Gemini may use only recommendations where `safe_to_apply=true` when creating the proposed resume patch.

After drafting:

1. Render a canonical plain-text proposed resume using the revised summary, exactly two revised bullets, and reordered existing skills.
2. Re-run the same deterministic ATS rubric.
3. Store the result as `projected_ats_assessment` with `resume_version="proposed"`.
4. Display the comparison:

```text
Original estimated ATS readiness: 62/100 — Low
Projected after proposed safe changes: 76/100 — Needs Targeted Changes
```

Label the second value as projected under the demo rubric. It is not a promise that a real ATS score will improve.

If the projected score is lower than the original score, show a warning and do not describe the changes as an improvement.

---

## 17. Match explanation

After Python calculates the Job Match Score, Gemini may create a concise explanation using only the calculated components. Keep this explanation separate from the ATS Readiness assessment.

Input to Gemini must include:

- Component scores.
- Matched skills.
- Missing skills.
- Experience concern.
- Location/work-mode result.
- Selected job title and company.

Output should be no more than approximately 90 words.

The explanation must not change the numeric score or add unsupported qualifications.

Example UI:

```text
Demo Job Match Score: 82/100

Strong matches
✓ Python
✓ SQL
✓ Tableau
✓ Excel

Missing or unclear
△ Power BI
△ A/B testing

Possible concern
! One year of related experience is preferred
```

---

## 18. Tailored application generation

Generate only a small application package for the selected job:

- One revised professional summary.
- Exactly two revised bullets.
- Reordered existing skills.
- One cover letter of 100–140 words.
- Missing requirements list.
- Keywords-used list.
- Safe ATS recommendations applied, with recommendation IDs.
- Unsupported ATS gaps intentionally not applied.

### Tailoring rules

Gemini may:

- Improve clarity.
- Reorder content.
- Emphasize relevant facts.
- Shorten or combine phrasing.
- Use terminology from the job description when it remains truthful.

Gemini may not invent:

- Skills.
- Employers.
- Job titles.
- Degrees.
- Certifications.
- Dates.
- Metrics.
- Responsibilities.
- Accomplishments.
- Work authorization.

Every revised bullet must contain a valid `source_bullet_id`.

Use only ATS recommendations marked `safe_to_apply=true`. When a job requirement is absent from the resume, place it in `missing_requirements`; do not add it to the tailored resume. Do not use keyword stuffing merely to raise the projected demo score.

---

## 19. Truthfulness validation

Validation is a required demo feature, not an optional enhancement.

### Deterministic validation

Check in Python that:

- Every `source_bullet_id` exists.
- `original_text` matches the referenced source.
- No new employer appears.
- No new degree or institution appears.
- No new date appears.
- No new numeric metric appears without source evidence.
- Reordered skills are a subset of the original skills.
- Every applied ATS recommendation is marked `safe_to_apply=true` and cites supporting resume evidence.
- Unsupported ATS gaps are not added to the summary, skills, bullets, or cover letter.
- Cover letter does not claim an unsupported skill.

### LLM claim review

Break revised content into claims and ask Gemini to classify each as:

```text
supported
unsupported
unclear
```

Each review must include:

- The claim.
- Status.
- Supporting resume IDs.
- Concise reason.

### Revision loop

- If unsupported claims exist and `revision_count == 0`, revise once automatically.
- Validate the revised draft again.
- Do not loop more than once automatically.
- If unsupported or unclear claims remain, show them to the human reviewer.

### Required demo example

The fictional resume includes Tableau but not Power BI. If a selected job asks for Power BI, the interface should visibly show:

```text
Job requirement: Power BI
Evidence in master resume: None
Decision: Power BI was not added to the resume
Recommendation: Treat it as a learning gap
```

---

## 20. Human approval node

The graph must pause before export.

Show:

```text
Application package ready

[Approve]
[Request Changes]
[Reject]
```

### Approve

- Export Markdown and JSON.
- Record approval timestamp.
- Display download buttons.
- Clearly state that no application was submitted.

### Request Changes

- Collect short human feedback.
- Return to a revision node.
- Re-run validation.
- Pause again.

### Reject

- Stop the graph.
- Do not export an approved package.
- Preserve the review screen for teaching purposes.

No external write, submission, or email action exists in this MVP.

---

## 21. Streamlit interface

Build one polished page rather than a complex multi-page application.

### Header

```text
Cougar Career Agent
Fresh jobs. Explainable matching. ATS-ready, truthful tailoring.
```

Add a visible workshop disclaimer:

```text
Demo uses a fictional resume and public job data. It does not submit applications.
ATS readiness is estimated using a transparent workshop rubric, not an employer's proprietary ATS.
```

### Section A — Search preferences

Controls:

- Target role text input.
- Location text input.
- Work-mode selectbox: Remote, Hybrid, On-site, Any.
- **Job source / query category** selectbox: Direct Company Careers, LinkedIn, Indeed, Google Jobs / Web, All Public Sources.
- **Freshness** selectbox: Last 1 hour, Last 24 hours, Last 3 days, Last 7 days.
- Run button.

Default values:

```text
Target role: Data Analyst Intern
Location: Houston, TX
Work mode: Any
Job source / query category: Direct Company Careers
Freshness: Last 24 hours
```

After the user clicks **Run Career Agent**, render a persistent search summary above the results:

```text
Searching: Direct Company Careers · Last 24 hours · Data Analyst Intern · Houston, TX
```

For a LinkedIn query, the same summary should read:

```text
Searching: LinkedIn · Last 1 hour · Data Analyst Intern · Houston, TX
```

### Section B — Agent activity

Show observable events, not hidden chain-of-thought.

Examples:

```text
✓ Fictional resume loaded
✓ Search query created for Direct Company Careers · Last 24 hours
✓ Firecrawl searched public job pages
✓ 8 pages retrieved and source categories classified
✓ 3 invalid or closed pages removed
✓ 5 jobs scored
✓ Top 3 matches prepared
✓ Selected resume scored for ATS readiness
✓ Prioritized safe changes prepared
```

Use `st.status`, progress text, or a compact event log.

Never expose private chain-of-thought or internal reasoning tokens.

### Section C — Top three job cards

Each card must show:

- Rank.
- Title.
- Company.
- Location/work mode.
- Demo Job Match Score.
- Selected query category.
- Actual detected source badge, such as LinkedIn, Indeed, Greenhouse, Lever, or Other.
- Requested freshness window.
- Source posting timestamp/date or “not available.”
- Freshness evidence badge: Exact timestamp, Date only, or Search-filtered/unverified.
- Retrieval timestamp.
- Live or cached badge.
- A 2–3 line description excerpt.
- **View job description** expander or details panel.
- A prominent clickable **Open job posting ↗** link button using the validated final URL.
- Optional **Apply on source ↗** button only when an explicit `apply_url` exists.
- A plain-text/copyable URL directly beneath the button for presentation reliability.
- Select button.

Implementation pattern:

```python
from urllib.parse import urlparse

def is_safe_public_job_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not parsed.username

if is_safe_public_job_url(job.source_url):
    st.link_button(
        "Open job posting ↗",
        job.source_url,
        key=f"open_job_{job.job_id}",
        help=f"Open the original {job.source_label} posting in a new tab",
        use_container_width=True,
    )
    st.caption(job.source_url)
```

Do not render raw scraped HTML to create links. Use Streamlit's link component after URL validation. `st.link_button` opens the external URL in a new browser tab, which lets the instructor verify the posting without losing the running demo.

Inside **View job description**, show:

1. The cleaned full job description in readable Markdown.
2. Responsibilities.
3. Required skills.
4. Preferred skills.
5. Education and experience requirements when available.
6. Query category and detected source.
7. Requested freshness, source posting evidence, retrieval time, and original public URL.
8. A clickable **Open job posting ↗** link.

Do not make the user open the external website merely to understand what the job requires.

### Section D — Job details and match explanation

Keep the selected job's full description visible or expandable while showing component scores and concise strengths, gaps, and concerns. The audience should be able to compare the resume against the actual job text on the same screen.

### Section E — ATS readiness and “What to change first”

After the user selects a job, show:

```text
Demo ATS Readiness Score: 62/100 — Low
Estimated using this demo rubric; not an official employer ATS score.
```

Display the six component scores and a prioritized recommendation panel.

For each recommendation, show:

- Priority.
- Resume section.
- Exact recommended change.
- Evidence resume IDs.
- `Safe to apply: Yes/No`.
- Expected effect.

Use separate visual treatment for:

- Safe wording or structure changes that the resume evidence supports.
- Unsupported job gaps that must not be added.

After the resume patch is generated, show:

```text
Original estimated score: 62/100
Projected under the same rubric: 76/100
```

Do not use red/green pass/fail language. Use Strong, Needs Targeted Changes, and Low.

### Section F — Resume patch

Display original and revised text side by side.

Show:

- Original summary and revised summary.
- Original bullet and revised bullet.
- Source bullet ID.
- ATS recommendation IDs applied.
- Missing requirements and unsupported gaps not applied.
- Cover letter.

### Section G — Validation

Show:

```text
✓ Source IDs verified
✓ No unsupported employer or degree added
✓ Skills remain grounded in the master resume
✓ Only safe ATS recommendations were applied
```

When a claim fails, show it clearly.

### Section H — Human approval

Render approval controls only after validation completes.

---

## 22. Export format

After approval, create:

```text
output/application_package_<timestamp>.md
output/application_package_<timestamp>.json
```

Markdown should contain:

1. Candidate name marked as fictional.
2. Selected job, selected query category, detected source category, requested freshness window, and clickable source URL.
3. Retrieval time, exact posting timestamp/date when available, posting age in hours when calculable, and freshness-evidence label.
4. Cleaned selected job description.
5. Job Match Score and explanation.
6. Original Demo ATS Readiness Score, band, component breakdown, and disclaimer.
7. Prioritized ATS recommendations with safe/unsafe labels and supporting evidence IDs.
8. Revised summary.
9. Two revised bullets with source IDs and applied recommendation IDs.
10. Reordered skills.
11. Missing requirements and unsupported ATS gaps.
12. Projected ATS Readiness Score under the same demo rubric.
13. Cover letter.
14. Validation report.
15. Approval timestamp.
16. Statement: “No application was submitted.”

Never include API keys, raw prompts, or hidden reasoning in exports.

---

## 23. Cached demo data

Create `data/cached_jobs.json` from a successful Firecrawl run before the workshop.

The cache must include:

- Original retrieval timestamp.
- Validated canonical direct job URL plus the original raw result URL.
- Optional explicit apply URL.
- Selected query category, requested freshness window, detected source category, and freshness evidence.
- Cleaned full job description and description excerpt.
- Raw Firecrawl fields needed by the pipeline.
- At least five valid postings across at least two source categories.
- At least one posting that is a strong match.
- At least one posting that requests Power BI, which the sample resume lacks.
- Expected original ATS assessment, prioritized recommendations, and projected assessment for the selected rehearsal job.
- At least one clearly labeled LinkedIn public-page example and one Direct Company Careers example, when those pages were publicly retrievable. If a cached LinkedIn page lacks a full description, mark it unusable for scoring rather than inventing one.

The cache must be clearly labeled synthetic or previously retrieved public demonstration data.

Do not manually alter cached job claims without recording the change.

---

## 24. Error handling

The app must fail gracefully.

Handle:

- Missing environment variables.
- Firecrawl authentication failure.
- Firecrawl timeout.
- Empty search response.
- No valid results for the selected query category and freshness window.
- Zero Last 1 hour results; offer an explicit one-click expansion to Last 24 hours without changing category.
- Invalid, unsafe, redirected-to-generic, or inaccessible job links.
- Invalid URLs.
- Failed page scrape.
- Full description unavailable because the source blocks public retrieval.
- Gemini authentication failure.
- Invalid structured LLM output.
- Zero valid jobs after filtering.
- Missing selected job.
- ATS scoring failure or incomplete selected-job description.
- No safe recommendations available for a low score.
- Projected ATS score lower than the original score.
- Failed validation.
- Export-directory permission error.

User-facing errors must be concise and actionable. For example: “No publicly readable LinkedIn job descriptions were available. Retry with Direct Company Careers or All Public Sources.” Never broaden the category silently.

Do not display stack traces by default in the Streamlit UI.

Log errors locally without logging resume text, cover letters, API keys, or full prompts.

---

## 25. Tests and acceptance criteria

### Unit tests

Implement tests for:

- Query construction for every source category.
- Domain-filter mapping for every source category.
- Freshness-window-to-`tbs` mapping, including `qdr:h` and `qdr:d`.
- Runtime custom-date-range construction for Last 3 days.
- Category + freshness combinations, such as LinkedIn + Last 1 hour and Indeed + Last 24 hours.
- URL-based actual-source classification.
- URL canonicalization and redirect resolution.
- Public `http`/`https` job-link validation and unsafe-scheme rejection.
- Preservation of both canonical direct URL and original result URL.
- Closed-job detection.
- Last-hour and last-day cutoff filtering.
- Exact timestamp, date-only, and search-filter-only freshness labeling.
- Missing-date behavior.
- Duplicate detection.
- Full-description preservation and excerpt generation.
- Skill alias normalization.
- Deterministic Job Match scoring.
- Deterministic ATS Readiness scoring.
- Both score bounds from 0 to 100.
- ATS score-band boundaries.
- Low-score recommendation count and priority ordering.
- Supported-but-missing keyword detection.
- Unsupported ATS gap recommendations marked `safe_to_apply=false`.
- Projected re-score using the same rubric.
- Unsupported-skill detection.
- Source-bullet-ID validation.
- Live-to-cache fallback.
- Approval routing.

### Integration tests

Use mocked Firecrawl and Gemini responses. Do not require real API calls in the default test suite.

### Definition of done

The MVP is complete only when:

- Firecrawl results include validated, direct, clickable source URLs.
- Every ranked job renders an **Open job posting ↗** button and the link opens the stored direct public job URL.
- The UI includes a Job source / query category selector with LinkedIn, Indeed, Google Jobs / Web, Direct Company Careers, and All Public Sources.
- The UI includes Last 1 hour and Last 24 hours freshness choices, plus Last 3 days and Last 7 days.
- The selected query category and freshness window are combined in the Firecrawl request; neither is silently broadened.
- Each result shows both the selected query category and the actual source detected from its URL.
- Every ranked job has an expandable cleaned full job description; a mere search snippet is not accepted for scoring.
- The app records retrieval timestamps.
- Closed jobs are removed.
- Missing posting timestamps/dates are labeled honestly, and Last 1 hour is called verified only with exact timestamp evidence.
- Duplicate postings are removed.
- The same inputs produce the same Job Match and ATS Readiness scores.
- Gemini cannot change either numeric score.
- The selected job displays a Demo ATS Readiness Score, six-component breakdown, score band, and explicit disclaimer.
- A score below 80 produces prioritized, section-specific recommendations.
- Unsupported job requirements are labeled as gaps and never recommended as resume claims.
- Safe recommendations include supporting resume IDs.
- The proposed resume is re-scored with the same deterministic rubric and labeled as projected.
- Exactly three ranked jobs are displayed when at least three valid jobs exist.
- Every revised bullet has a valid source ID.
- An unsupported Power BI claim is blocked.
- The graph pauses at approval.
- No application is submitted.
- Cached fallback is visibly labeled.
- The complete workflow runs in under approximately 60 seconds.
- The instructor can complete the live narrative in approximately 8 minutes.

---

## 26. Development sequence

Build in this order. Do not start with UI polish.

### Milestone 1 — Foundation

- Create repository structure.
- Add requirements and environment loading.
- Add Pydantic models.
- Add fictional resume.
- Add basic tests.

Success condition: sample resume loads and validates.

### Milestone 2 — Live retrieval and cache

- Implement source-category enum, freshness-window enum, query builder, `tbs` mapping, domain mapping, and URL-based source classifier.
- Implement Firecrawl adapter.
- Implement direct-link validation, redirect resolution, and canonical URL preservation.
- Implement timeout handling.
- Save/load cached data.
- Add live/cached status metadata.

Success condition: raw jobs are available even when the live API fails.

### Milestone 3 — Normalization and filtering

- Implement Gemini job extraction.
- Validate structured output.
- Preserve and clean full job descriptions.
- Generate deterministic description excerpts.
- Detect closed jobs.
- Filter stale jobs.
- Deduplicate results.

Success condition: at least three clean `JobPosting` objects are produced.

### Milestone 4 — Deterministic ranking and ATS readiness

- Implement canonical skill matching.
- Implement TF-IDF similarity.
- Implement role, experience, and preference scores.
- Rank top three.
- Implement the six-component ATS Readiness rubric.
- Implement score bands and prioritized recommendations.
- Distinguish supported-but-missing keywords from unsupported gaps.
- Add Job Match and ATS score tests.

Success condition: job rankings and original ATS readiness assessment work without an LLM.

### Milestone 5 — Tailoring, projected re-score, and validation

- Generate summary, two bullets, skills ordering, and cover letter using only safe ATS recommendations.
- Re-score the proposed resume under the same rubric.
- Add source mappings.
- Implement deterministic and LLM validation.
- Add one automatic correction loop.

Success condition: Power BI is not added when unsupported.

### Milestone 6 — LangGraph

- Assemble nodes and conditional edges.
- Add state persistence.
- Add selection and approval interrupts.
- Test approve/change/reject routes.

Success condition: the graph pauses and resumes correctly.

### Milestone 7 — Streamlit

- Add controls, including Job source / query category and Last 1 hour / Last 24 hours freshness.
- Add agent activity display.
- Add job cards with source badges, freshness-evidence badges, clickable Open job posting buttons, description excerpts, and View job description expanders.
- Add ATS score breakdown, “What to change first,” and original-versus-projected comparison.
- Add resume comparison.
- Add validation and approval controls.
- Add exports.

Success condition: the entire workflow is usable from one page.

### Milestone 8 — Rehearsal hardening

- Create cached workshop data.
- Warm the Gemini request before presenting.
- Test without internet.
- Test with an invalid Firecrawl key.
- Test with an invalid Gemini key.
- Measure total runtime.
- Rehearse the exact eight-minute script.

---

## 27. Exact eight-minute demo script

### 0:00–0:40 — Frame the agent

Say:

> A chatbot gives job-search advice. This agent retrieves current jobs, uses tools, calculates a score, modifies a document, validates its own output, and pauses for approval.

### 0:40–1:40 — Start the workflow

Use:

```text
Role: Data Analyst Intern
Location: Houston, TX
Work mode: Any
Job source: Direct Company Careers
Freshness: Last 24 hours
```

Click **Run Career Agent**.

### 1:40–2:40 — Show live retrieval

Point out:

- Job source / query category selector, including LinkedIn.
- Actual detected source badge.
- Clickable Open job posting link.
- Requested freshness window.
- Source posting timestamp/date and evidence badge.
- Retrieval time.
- Live/cached indicator.
- View job description expander showing the complete cleaned posting.

Say:

> The LLM did not remember these jobs. Firecrawl retrieved them from the current web.

Briefly point out that **Last 1 hour** is also available. If it returns zero verified postings, the app must ask before expanding to Last 24 hours.

### 2:40–3:30 — Show ranking

Show the top three jobs, click **Open job posting ↗** on one card to prove the source link is real, then open **View job description** and point out that the Job Match Score is calculated against the displayed source text rather than against a fabricated summary.

Say:

> Python calculates the number. Gemini explains the number.

### 3:30–4:30 — Select a job and show ATS readiness

Show:

```text
Demo ATS Readiness Score: 62/100 — Low
Estimated using this demo rubric; not an official employer ATS score.
```

Open **What to change first** and point out one safe recommendation and one unsupported gap.

Say:

> This is a transparent readiness estimate, not a secret employer ATS score. The agent tells us exactly what it measured and what it would change.

### 4:30–5:30 — Generate the safe resume patch

Generate:

- One revised summary.
- Two revised bullets.
- One short cover letter.
- A projected ATS readiness score under the same rubric.

### 5:30–6:30 — Show the guardrail

Highlight the Power BI gap.

Say:

> The job asks for Power BI. The resume does not support it, so the agent refuses to add it.

### 6:30–7:20 — Show approval

Show:

```text
Approve · Request Changes · Reject
```

Do not submit anything.

### 7:20–8:00 — Reveal the architecture

Display:

```text
Firecrawl retrieves
Python matches and scores ATS readiness
Gemini explains and drafts safe changes
Validator checks
Human approves
```

End with:

> The model is not the agent. The agent is the model plus tools, state, validation, decisions, and permissions.

---

## 28. Security and UHS workshop rules

- Use a fictional resume only.
- Use public job pages only.
- Do not use Level 1 confidential or Level 2 protected data.
- Do not use a live UHS account or production UHS system.
- Do not use a UHS-issued secret in a public repository.
- Use least privilege.
- Make all external actions read-only.
- Require human approval before export.
- Never submit an application.
- Do not store participant inputs after the session.
- Do not log protected data.
- Respect source websites' terms, robots directives, and applicable policies.
- Open source does not by itself imply institutional approval.

---

## 29. Coding standards for Claude Code

When implementing this project:

1. Treat this file as the authoritative specification.
2. Inspect existing files before editing.
3. Prefer the simplest working design.
4. Avoid speculative features.
5. Use type hints throughout.
6. Add docstrings to public functions.
7. Keep pure logic separate from Streamlit rendering.
8. Keep API clients behind adapters.
9. Validate all external responses.
10. Use dependency injection for Firecrawl and Gemini clients so tests can mock them.
11. Keep both Job Match and ATS Readiness scoring deterministic.
12. Treat ATS readiness as an educational estimate and never represent it as a real employer score.
13. Generate actionable changes only from supported resume evidence; unsupported gaps must stay gaps.
14. Never expose hidden chain-of-thought.
15. Never claim a job is current without source support.
16. Never label a result LinkedIn, Indeed, or another source merely because that category was queried; detect the actual source from the final URL.
17. Never score a job from a title/snippet alone when the full description is unavailable.
18. Never invent a resume fact.
19. Run `pytest -q` after each milestone.
20. Report failures honestly instead of bypassing tests.
21. Do not delete unrelated files.
22. Do not introduce a database, frontend framework, or container stack unless explicitly requested.

---

## 30. Local setup commands

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy environment template:

```bash
cp .env.example .env
```

Run tests:

```bash
pytest -q
```

Run the app:

```bash
streamlit run app.py
```

---

## 31. First implementation request

When asked to build this project, proceed in small milestones.

Start by creating:

1. The repository structure.
2. `requirements.txt`.
3. `.env.example` and `.gitignore`.
4. Pydantic models, including source/query category, freshness-window, canonical job-link, posting-time evidence, job-description, and ATS readiness fields.
5. The fictional sample resume with synthetic document-feature metadata.
6. Initial tests for query-category + freshness mapping, `qdr:h` and `qdr:d`, canonical URL safety, clickable-link preservation, source classification, posting-time filtering, description preservation, Job Match scoring boundaries, ATS scoring boundaries, and safe/unsafe recommendation behavior.

Then stop and report:

- Files created.
- Tests run.
- Test results.
- The next milestone.

Do not jump directly to a large monolithic `app.py`.
