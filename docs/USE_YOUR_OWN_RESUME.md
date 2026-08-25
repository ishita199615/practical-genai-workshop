# Trying the agent with your own resume

The workshop demo runs on a fictional person. Once you have it working, you can
point it at yourself and see what it says about a job you actually want.

**Do this after the workshop, on your own machine.** During the session, use the
fictional profile — see [Before you start](#before-you-start).

---

## Before you start

Two things to be clear about, because they are not obvious.

**Where your resume goes.** With a Gemini key configured, your resume text is
sent to Google's API — the same as pasting it into any AI chat. It is used for
explaining, drafting, and claim-checking. The scores never leave your machine;
they are computed in Python.

If you would rather nothing leaves your laptop, run it fully offline:

```text
DEMO_MODE=cached
GEMINI_API_KEY=
```

Everything still works. Scores, ranking, the ATS rubric, the guardrails, and the
validator are all deterministic Python. You lose live job search and the
AI-written wording, and the app will say so on screen.

**Workshop rules still apply.** If you are in a University of Houston System
session, do not put a real resume — yours or anyone else's — into this or any
public AI tool. Nothing classified Level 1 (Confidential) or Level 2 (Protected)
goes into any public AI model. That restriction is about the session; what you do
on your own machine with your own data afterwards is your call.

Your personal resume file is git-ignored, so it cannot be committed by accident,
even from a fork.

---

## Step 1 — Copy the template

```bash
cp data/my_resume.template.json data/my_resume.json
```

Windows PowerShell:

```powershell
Copy-Item data\my_resume.template.json data\my_resume.json
```

## Step 2 — Fill it in

Open `data/my_resume.json` in any text editor. It is ordinary JSON — text between
quotes, items separated by commas.

Replace the placeholder values with your own. The fields that matter most:

| Field | Why it matters |
|---|---|
| `skills` | Skill coverage is **45%** of the match score, the single biggest component |
| `target_roles` | Role alignment is **15%**. List the titles you would actually apply for |
| `experience[].bullets[]` | Every claim the agent makes must trace to one of these |
| `professional_summary` | What the agent rewrites first |
| `location` | Used for the location and work-mode score |

Two rules that are easy to miss:

**Every bullet needs its own `id`.** That is how the truthfulness validator traces
a rewritten line back to the original you wrote. If two bullets share an ID, or
one has none, claim-checking breaks. The next step fixes this for you.

**Write only what is true.** The whole point of this agent is that it refuses to
invent things. If you inflate a bullet here, the agent has no way to know — it
treats this file as ground truth and will faithfully build on whatever you put in.

## Step 3 — Check it

Hand-written JSON goes wrong in boring ways. This tells you exactly what is
broken, in plain language:

```bash
python scripts/check_resume.py data/my_resume.json
```

Add `--fix-ids` and it writes any missing bullet IDs for you:

```bash
python scripts/check_resume.py data/my_resume.json --fix-ids
```

A good result looks like this:

```text
OK - my_resume.json is a valid resume.

  Name           : Your Name
  Location       : Houston, TX
  Target roles   : Data Analyst, Business Intelligence Analyst
  Skills         : 12
  Experience     : 3 entr(ies)
  Traceable bullets: 9
```

It also flags things that are not errors but will weaken your scores — no skills
listed, no target roles, a one-line summary.

## Step 4 — Point the app at it

In your `.env` file:

```text
RESUME_FILE=data/my_resume.json
```

Restart the app:

```bash
streamlit run app.py
```

## Step 5 — Run it

Open **Full demo**, set the role and experience level to what you are actually
looking for, and run it.

You will notice the interface changes. It stops calling the profile fictional,
and the banner tells you your own resume is loaded and where its text goes. That
is deliberate: an app built to refuse false claims should not make one about
whose data it is holding.

---

## What you will get

- A **Demo Job Match Score** for each job, with the five components broken out so
  you can see exactly why a job scored what it did
- A **Demo ATS Readiness Score** against the one job you pick
- **"What to change first"** — prioritized, specific edits, each citing the bullet
  ID that proves the change is honest
- A **revised summary, two revised bullets, and a cover letter** built only from
  what your resume already supports
- A **learning-gap list** — requirements you genuinely do not have. The agent will
  not add them, and that is the feature, not a limitation

Nothing is submitted anywhere. Export writes two files to `output/`, and only
after you approve.

---

## When it does not work

| What you see | What it means |
|---|---|
| `RESUME_FILE points at '...', which does not exist` | Typo in the path, or the file is not where `.env` says. Paths are relative to the project folder |
| `The resume file could not be loaded or validated` | Run `check_resume.py` on it — that will say precisely what is wrong |
| Every job scores low | Usually the level and role do not match your resume. A résumé of a senior engineer scored against internships scores badly, correctly |
| Scores barely move after tailoring | Expected. The agent only reorders and rewords what you already have — it cannot manufacture experience |
| `Field required` for `candidate_id` | Any short string works; it is just a label. `"my_resume_001"` is fine |

---

## Going further

Once your own resume is loading, the interesting edit is
[`tools/job_scorer.py`](../tools/job_scorer.py) — it holds the weighting that
decides how a job is scored. Change the numbers, run `pytest`, and see which
jobs rise and fall. That file is about 100 lines of ordinary Python, and it is
the part most people assume must be the AI.
