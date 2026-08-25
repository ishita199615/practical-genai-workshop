You are reading one public job posting page and returning structured fields.

Extract only information explicitly present in SOURCE_CONTENT.

Do not invent or infer a posting date.
Do not infer a company from unrelated navigation content.
Do not convert preferred qualifications into required qualifications.
Use null when information is unavailable.
Set is_closed=true when the page says the position is unavailable,
filled, expired, or no longer accepting applications.
Return the complete main job-description content in `description`; do not replace it with a brief summary or search snippet.

Additional rules:

- `posted_at_text` must be copied verbatim from the page (for example "Posted 3 hours ago" or "2026-08-20"). Use null when the page shows no posting date or time.
- `apply_url` must be an absolute link that the page itself presents as the application link. Use null when no such link is visible.
- `is_specific_opening` is false when the page is a search page, a category index, or a general careers homepage rather than one specific opening.
- `required_skills` may contain only skills the page marks as required or minimum.
- `preferred_skills` may contain only skills the page marks as preferred, nice to have, or a plus.
- `minimum_experience_years` must be a number the page states explicitly. Use null otherwise.

SOURCE_URL:
{SOURCE_URL}

SOURCE_TITLE:
{SOURCE_TITLE}

SOURCE_CONTENT:
{SOURCE_CONTENT}
