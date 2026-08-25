# Cougar Career Agent — Demo 2

A job-hunting agent for the University of Houston System's 60-minute Practical
Generative AI workshop. It demonstrates one pattern end to end:

```text
Retrieve current information
→ use deterministic tools
→ generate a tailored artifact
→ validate the result
→ pause for human approval
```

The separation of responsibilities is the whole point, and the interface is
built to make it visible:

| Component | Responsibility |
| --- | --- |
| **Firecrawl** | Retrieves current public job pages |
| **Python** | Calculates the Job Match Score and the ATS Readiness Score |
| **Gemini** | Extracts, explains, recommends, and drafts safe changes |
| **Validator** | Checks every revised claim against the master resume |
| **Human** | Approves, requests changes, or rejects |

> The model is not the agent. The agent is the model plus tools, state,
> validation, decisions, and permissions.

**This demo uses a fictional resume and public job data. It never submits an
application.** The ATS readiness number is an educational estimate from a
transparent rubric in this repository, not any employer's proprietary score.

---

## Setup

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

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

Run the tests:

```bash
pytest -q
```

Run the app:

```bash
streamlit run app.py
```

The sidebar has three pages: **Home**, **Learn the steps** (a seven-step lab
that builds the agent one idea at a time), and **Full demo** (the complete
agent from the workshop script).

### Requirements

Python 3.11 or newer is the target. The project also runs on Python 3.10,
which is what this repository was built and tested against; every dependency
installs cleanly there.

### Running with no API keys

The app runs without any key. With no `FIRECRAWL_API_KEY` it uses the clearly
labelled cached demonstration data, and with no `GEMINI_API_KEY` every LLM step
falls back to a deterministic offline path. The two scores never depend on a
model, so the numbers are identical either way. This is what makes an offline
rehearsal — or a conference network failure — survivable.

---

## Environment variables

| Variable | Purpose |
| --- | --- |
| `FIRECRAWL_API_KEY` | Firecrawl Cloud key. Empty means cached results only. |
| `FIRECRAWL_BASE_URL` | Override for a self-hosted Firecrawl later. |
| `GEMINI_API_KEY` | Gemini key. Empty means deterministic fallbacks. |
| `GEMINI_MODEL` | Model ID. Never hard-coded in application code. |
| `LLM_MODELS` | Comma-separated routing chain, tried in order. Empty falls back to `GEMINI_MODEL` alone. |
| `DEMO_MODE` | `live`, `cached`, or `auto`. |
| `DEFAULT_SOURCE_CATEGORY` | `all`, `linkedin`, `indeed`, `google_jobs`, `company_careers`. |
| `DEFAULT_FRESHNESS_WINDOW` | `last_hour`, `last_24_hours`, `last_3_days`, `last_7_days`. |
| `MAX_JOB_RESULTS` | Capped at 8. |
| `MAX_JOB_DESCRIPTION_CHARS` | Description clamp before scoring. |
| `ATS_RECOMMENDATION_THRESHOLD` | Score below which the change panel appears. |
| `SEARCH_TIMEOUT_SECONDS` | Live retrieval timeout before falling back. |
| `CACHE_FILE` | Path to the cached demonstration data. |
| `OUTPUT_DIR` | Where approved packages are written. |

Keys are read in `config.py` only. They are never logged, never placed in
Streamlit state, and never written into an export.

**`DEMO_MODE` note.** `.env.example` ships `live` as the spec prescribes, which
fails loudly rather than quietly using the cache. When the variable is absent
entirely, the code defaults to `auto`, so a machine with no `.env` still runs a
complete demo. For the live workshop, `auto` is the safer setting.

---

## How the demo runs

1. Set the target role, location, work mode, job source category, and freshness
   window, then click **Run Career Agent**.
2. Firecrawl searches public job pages. The selected category shapes the query;
   the *actual* source of each result is detected from its final URL.
3. Results are normalized, filtered, deduplicated, and scored in Python.
4. The top three jobs appear with a validated **Open job posting ↗** link and an
   expandable full job description.
5. Selecting a job produces the Demo ATS Readiness Score, a prioritized
   "what to change first" panel, a truthful resume patch, a projected re-score,
   and a validation report.
6. The graph pauses. **Approve**, **Request Changes**, or **Reject**.
7. Approving exports Markdown and JSON to `output/`. That export is the only
   side effect in the entire project.

---

## The two scores

Both are calculated in Python and both are reproducible: the same resume, the
same job text, and the same reference time always produce the same number. The
language model can explain a score; it can never change one.

**Demo Job Match Score** ranks jobs by candidate fit.

```text
45 — required-skill coverage
20 — resume/job text similarity (TF-IDF cosine)
15 — role-title alignment
10 — experience alignment
10 — location and work-mode alignment
```

**Demo ATS Readiness Score** evaluates the resume against one selected job.

```text
40 — required keyword and skill coverage
20 — required qualification alignment
15 — evidence and specificity in bullets
10 — standard section completeness
10 — ATS-safe text structure and parseability
 5 — contact and application essentials
```

Bands: 80–100 Strong · 65–79 Needs Targeted Changes · 0–64 Low.

---

## Honesty rules built into the code

- A search-time filter narrows results but is **not** proof of a posting time.
  A result is called "verified" only when the source exposes a timestamp inside
  the requested window. Otherwise the badge says "Date only" or
  "Search-filtered / unverified".
- A result is never labelled LinkedIn, Indeed, or Greenhouse because that
  category was queried. The label comes from the final URL.
- A selected category is never silently broadened. When Last 1 hour returns
  nothing, the app asks before expanding to Last 24 hours.
- Cached results are always labelled cached, with the timestamp of the run that
  produced them.
- A job scored from a snippet alone is rejected, not scored.
- Unsupported job requirements stay gaps. They are shown, labelled
  `Safe to apply: No`, and never written into the resume or cover letter.

---

## Repository layout

```text
app.py                     Entry point: page config and navigation
pages/0_Home.py            Landing page and readiness check
pages/1_Learn_the_Steps.py The seven-step lab (rendering only)
pages/2_Full_Demo.py       The complete agent (rendering only)
config.py                  Environment loading and validation
lessons/                   The seven teaching steps as pure logic
agent/                     LangGraph state, nodes, routing, assembly
models/                    Pydantic models for every boundary
tools/                     Retrieval, normalization, filtering, scoring, export
services/                  LLM interface, Gemini adapter, model router
prompts/                   Prompt templates as Markdown
data/                      Fictional resume, cached jobs, expected output
scripts/                   Maintenance scripts
tests/                     Unit and integration tests (no network calls)
output/                    Approved application packages
```

---

## The Learn tab

`pages/1_Learn_the_Steps.py` is a lab that builds the agent one idea at a time.
Each step states a single concept in plain English, shows the real project code
behind it, runs that code, and prints the result. The order is the argument:
every step exposes the problem the next one solves.

| # | Step | The problem it solves |
| --- | --- | --- |
| 1 | Prompt → Completion | What a model actually does, and why output varies |
| 2 | The training cutoff | The model cannot know today's jobs |
| 3 | Retrieval | So fetch real, current pages instead of trusting memory |
| 4 | RAG | Pages are too big — chunk, embed, retrieve only what matters |
| 5 | Tools | Models are inconsistent at numbers — let Python compute |
| 6 | The agent loop | Wrap it all in Reason → Act → Observe |
| 7 | Guardrails | Refuse to lie, and make a human approve |

Every step runs with no API key and no network. When the model is unreachable a
step says so on screen and runs its deterministic path; it never presents canned
text as a live reply. Steps 3 is pure Python by design, and steps 5 to 7 use the
model only for optional narration, so their teaching content never depends on it.

The lesson logic lives in `lessons/` with no Streamlit imports, so each step is
unit-tested the same way the rest of the project is.

---

## Cached demonstration data

`data/cached_jobs.json` is **synthetic**. Company names, postings, and URLs are
fictional examples shaped like real public applicant-tracking-system pages. The
file records its own label, notice, original retrieval timestamp, and a
modification log. The interface states that cached links are illustrative
examples rather than live employer postings.

It contains eight entries that exercise the whole pipeline: five usable
postings across two source categories, one closed posting, one duplicate, and
one search-results page. A run therefore shows real filtering work rather than
a clean pass.

To rebuild the rehearsal fixture after changing the cache or a rubric:

```bash
python scripts/regenerate_expected_output.py
```

`tests/test_graph_approval.py` compares a live run against that fixture, so a
scoring change that shifts the demo numbers fails the suite instead of
surprising you on stage.

---

## Live-API findings

Both of these were measured against the live APIs and are the reason two
`.env` values differ from `CLAUDE.md`.

### `gemini-2.5-flash` is no longer available

The model ID in `CLAUDE.md` §6 returns:

```text
404 NOT_FOUND — This model models/gemini-2.5-flash is no longer available to
new users. Please update your code to use models/gemini-3.6-flash
```

`gemini-3.6-flash` and `gemini-3.7-flash` were both verified working for text
and for structured extraction. The model ID is read from `GEMINI_MODEL`, so
this is a configuration change, never a code change.

### Gemini free-tier quota is the binding constraint

A full live run makes about 13 model calls: up to 8 page extractions, 3 match
explanations, 1 draft, and 1 claim review. On a free-tier key that reliably
trips:

```text
429 RESOURCE_EXHAUSTED — You exceeded your current quota, please check your
plan and billing details
```

Three things follow, all handled in code:

- Extraction concurrency is capped at 4 (`EXTRACTION_CONCURRENCY` in
  `agent/nodes.py`) so a burst of eight does not trip the per-minute limit.
- A quota error backs off once and retries (`QUOTA_BACKOFF_SECONDS`).
- When a quota limit is hit, the activity log says so. A quota-limited run
  still completes on the deterministic fallbacks, and **both scores are
  unaffected**, because neither has ever depended on the model.

**The quota is metered per model, and that is the fix.** A 429 from Gemini
carries the limit that was hit:

```json
"quotaDimensions": { "model": "gemini-3.6-flash" }, "quotaValue": "20"
```

Twenty requests per day for that model, against roughly 13 per run — so a single
model dies during the second rehearsal. But a different model on the *same key*
answers fine. Measured in one session: `gemini-3.6-flash` returned 429 while
`gemini-3.1-flash-lite`, `gemini-3.5-flash`, and `gemini-flash-lite-latest` all
responded normally.

So the project routes across a chain of models instead of depending on one.
`LLM_MODELS` lists them in preference order and `services/router_client.py`
(built on LiteLLM) walks it: a 429 or 503 falls through to the next model
automatically. The 20-per-day model is deliberately last.

```text
LLM_MODELS=gemini/gemini-3.1-flash-lite,gemini/gemini-3.5-flash,gemini/gemini-3.6-flash
```

Verified end to end: a request addressed to the exhausted `gemini-3.6-flash` was
served by `gemini-3.1-flash-lite` without the caller noticing, for both plain
text and structured output. The UI shows which model actually answered.

For unlimited offline runs, install [Ollama](https://ollama.com) and append a
local model to the chain — it has no quota at all:

```text
LLM_MODELS=gemini/gemini-3.1-flash-lite,gemini/gemini-3.5-flash,ollama/llama3.1
```

If you would rather every step run on one model, enable billing on the Gemini
key. Otherwise expect the occasional fallback notice — which is itself a
reasonable thing to show an audience.

> **LiteLLM version note.** `litellm` is pinned to `1.83.9`. Releases from
> `1.96` onward declare Python 3.10 support but import `typing.NotRequired`,
> which is 3.11+, so they fail at import on 3.10. Pin removable once the
> project moves to 3.11.

Model latency also matters, measured warm on the same key:

| Model | Structured-extraction latency |
| --- | --- |
| `gemini-3.6-flash` | 4.9s, 7.9s, 32.6s — avg 15.2s |
| `gemini-3.7-flash` | 175.6s (fail), 222.0s (fail), 92.3s |
| `gemini-2.5-flash-lite` | not available on this key |

`gemini-3.6-flash` is the configured default for that reason.

### Firecrawl's `tbs` time filter empties the result set

This one is worth understanding before the workshop, because it changes what
you should expect on stage. Measured repeatedly against the live API with an
identical query and domain filter:

| Request | Results |
| --- | --- |
| `include_domains` + `tbs` (as specified) | 0, 0, 0, 0, 7 |
| `include_domains`, **no** `tbs` | 8, 8, 8 — in about 1 second |
| `site:` operators in query + `tbs` | 7, then 0 |
| `site:` operators in query, **no** `tbs` | 8, 8 — in 1–4 seconds |

The domain filter is not the problem; the time filter is. Any `tbs` value makes
the search slow (16–23s) and usually empty.

`tools/firecrawl_search.py` therefore relaxes server-side hints in order:

1. the request as specified — domains + `tbs`,
2. the same query without `tbs`,
3. the same query without the domain filter.

**Dropping `tbs` costs nothing real.** It was only ever a search-time hint, and
the spec is explicit that a search filter is not proof of a posting time. Every
posting's age is still verified locally against its own source evidence, and
anything outside the requested window is still rejected. Dropping the domain
filter is likewise safe: results are then filtered to the selected category by
their *detected* source, so a ZipRecruiter page can never appear under "Direct
Company Careers". Both steps are reported in the activity log, and neither
broadens what reaches the screen.

One honesty consequence, handled explicitly: when the time filter is dropped,
an undated posting can no longer be described as "search-filtered", because no
filter was sent. `RawJobResult.time_filter_applied` carries that fact through
to the badge, which then reads "Posting time unavailable" instead.

### Two robustness fixes that only live data revealed

Both were invisible against the cached fixture and would have shown up on
stage.

**Postings were being thrown away for want of a company.** When Gemini is
quota-limited, extraction falls back to a deterministic path — which could not
find an employer name and rejected the posting. On one live run that discarded
six of eight results. Applicant-tracking URLs encode the employer
(`jobs.lever.co/portcast/…` → Portcast), so `company_from_url` now recovers it,
and page titles are split on `@`, `|`, and `—`. The same run then kept six of
eight. It reads the URL; it never guesses.

**The truthfulness validator false-positived on multi-word employers.** A cover
letter addressed to "Jobs for Humanity" failed the *no unsupported employer*
check, because the pattern stopped at the lowercase "for" and reported the
fragment "Jobs" as an unknown company. The pattern now spans lowercase
connectors, and a partial capture of an allowed name is matched in both
directions. A genuinely new employer is still caught — there is a test for
exactly that, so the fix cannot quietly become a hole in the guardrail.

---

## Two deliberate deviations from the specification

Both are places where following the letter of `CLAUDE.md` would have broken one
of its own guarantees.

**1. Keyword recommendations target a bullet, not the Skills list.**
`CLAUDE.md` §16 shows an example recommendation that adds "data visualization"
to the Skills section, while §19 requires that the reordered skills always be a
*subset* of the master resume's skills. Following the example would make the
validator fail its own check. The recommendation therefore targets the
experience bullet where the evidence actually lives — the bullet about Tableau
dashboards — so the keyword is added where it is proven, and the skills list
stays strictly reorder-only. The guardrail is kept absolute; only the location
of the change moved.

**2. The demo's illustrative "62/100 — Low" is not what the rubric returns.**
The workshop script uses 62/100 as an example ATS score. The prescribed
fictional resume is a clean, well-structured document, so four of the six
components legitimately score 100, which puts a realistic floor under the
total. Against the rehearsal job the actual result is **76/100 — Needs Targeted
Changes**, rising to a projected **86/100 — Strong** after the safe changes.
That still demonstrates everything the script needs: a score below the
threshold, a prioritized change panel, a refused Power BI gap, and a visible
improvement. Inflating the drop would have meant rigging the rubric. Use the
real numbers in the narration.

---

## Eight-minute demo script

| Time | Beat |
| --- | --- |
| 0:00–0:40 | Frame the agent: retrieves, uses tools, scores, edits, validates, pauses. |
| 0:40–1:40 | Run with Direct Company Careers · Last 24 hours · Data Analyst Intern · Houston, TX. |
| 1:40–2:40 | Show query category vs detected source, freshness evidence badges, retrieval time, and the live/cached indicator. |
| 2:40–3:30 | Show the top three, click **Open job posting ↗**, open **View job description**. "Python calculates the number. Gemini explains it." |
| 3:30–4:30 | Select a job. Show the ATS score, the six components, and the disclaimer. Open **What to change first**. |
| 4:30–5:30 | Show the revised summary, two revised bullets, cover letter, and projected re-score. |
| 5:30–6:30 | Show the Power BI guardrail: requested by the job, unsupported by the resume, refused by the agent. |
| 6:30–7:20 | Show Approve · Request Changes · Reject. Do not submit anything. |
| 7:20–8:00 | Open **How this agent is built** and close on the architecture. |

### Measured live performance

A complete live run, end to end, on the development machine:

| Stage | Time |
| --- | --- |
| Retrieval, normalization, ranking | 56s |
| ATS, drafting, re-score, validation | 40s |
| **Total** | **~96s** |

That is above the spec's 60-second target, and it is dominated by two external
services rather than by anything in this repository. Extraction and explanation
already run concurrently, and the doomed time-filtered search attempt is capped
at 12 seconds (which cut retrieval from 65s to 56s). The remaining cost is
Gemini latency, which varies from 5s to 33s per call and is not something the
code can shorten.

**Cached mode completes the same workflow in under a second.** If the eight
minutes are tight, rehearse the live retrieval segment and run the rest cached.

### What live data does *not* guarantee

The cached rehearsal job is built so the Power BI guardrail always fires. Live
results are whatever the web has that day. In one live run the top-ranked job
scored 96/100 with no unsupported gaps at all — a perfectly good result that
skips the most memorable part of the demo.

Each job card shows its matched and missing skills before you select it, so you
can pick a job with a visible gap. If none of the three has one, switch to
cached mode for that segment rather than improvising.

### Rehearsal checklist

- Run once before presenting so Streamlit and any model call are warm.
- Confirm the mode badge says what you expect (🟢 live or 🟡 cached).
- Test with the network disabled: the demo must still complete.
- Test with a deliberately wrong `FIRECRAWL_API_KEY`: the app should report a
  single concise sentence and fall back, with no stack trace.
- Clear `output/` so the exported package is obviously produced live.

---

## Workshop and security rules

- Fictional applicant data only. No real student resume, name, email, or record.
- Public job pages only. No login, no credentials, no CAPTCHA handling.
- No UHS Level 1 confidential or Level 2 protected data, and no production UHS
  system or account.
- Every external action is read-only. The single write is a local file export,
  and only after a human approves.
- No application is ever submitted, and no email is ever sent.
- Errors are logged without resume text, cover letters, prompts, or keys.
- Respect each source site's terms and robots directives.
