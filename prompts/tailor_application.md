Tailor a small application package for one job using only facts that already exist in MASTER_RESUME.

You may:

- Improve clarity.
- Reorder content.
- Emphasize relevant facts.
- Shorten or combine phrasing.
- Use terminology from the job description when it stays truthful.

You may not invent skills, employers, job titles, degrees, certifications, dates, metrics, responsibilities, accomplishments, or work authorization.

Hard requirements:

- `revised_summary`: one professional summary, at most 45 words.
- `revised_bullets`: exactly two entries. Each `source_bullet_id` must be one of ALLOWED_BULLET_IDS, and `original_text` must be copied exactly from that bullet.
- `reordered_skills`: the master resume's skills reordered for relevance. Use exactly the same skill strings, add nothing, and remove nothing.
- `keywords_used`: job-description terms you actually used, all of which must be supported by the resume.
- `applied_ats_recommendation_ids`: only IDs listed in SAFE_RECOMMENDATIONS.
- `unsupported_ats_gaps_not_applied`: copy UNSUPPORTED_GAPS verbatim. These must never appear in the summary, skills, bullets, or cover letter.
- `missing_requirements`: job requirements the resume cannot evidence.
- `cover_letter`: 100 to 140 words, addressed generally, no invented details, and no claim of a skill outside the master resume.

Never use keyword stuffing, hidden text, false titles, or invented numbers.

JOB_TITLE: {JOB_TITLE}
COMPANY: {COMPANY}
JOB_DESCRIPTION:
{JOB_DESCRIPTION}

MASTER_RESUME:
{MASTER_RESUME}

ALLOWED_BULLET_IDS: {ALLOWED_BULLET_IDS}

SAFE_RECOMMENDATIONS:
{SAFE_RECOMMENDATIONS}

UNSUPPORTED_GAPS: {UNSUPPORTED_GAPS}

REVISION_FEEDBACK:
{REVISION_FEEDBACK}
