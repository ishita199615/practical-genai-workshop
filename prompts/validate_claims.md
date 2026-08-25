Check each claim in DRAFT_CLAIMS against MASTER_RESUME and classify it.

Use exactly one status per claim:

- `supported`: every fact in the claim appears in the master resume.
- `unsupported`: the claim asserts a skill, tool, employer, degree, date, metric, or accomplishment the master resume does not contain.
- `unclear`: the claim is vague or only partially traceable to the master resume.

For each claim return the claim text, the status, the supporting resume IDs (empty when unsupported), and a concise one-sentence reason.

Rules:

- Judge only against MASTER_RESUME. Job-description content is not evidence about the candidate.
- Generic motivation or interest statements that assert no new fact are `supported` with no IDs.
- A claim naming a tool that is absent from the master resume is `unsupported`, even if a similar tool is present.
- Return one review per claim, in the same order.

MASTER_RESUME:
{MASTER_RESUME}

RESUME_IDS:
{RESUME_IDS}

DRAFT_CLAIMS:
{DRAFT_CLAIMS}
