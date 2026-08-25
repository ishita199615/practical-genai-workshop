# Demo 2- Cougar Career Agent 

A working AI agent you can run on your own laptop, read the source of, and take
apart. Built for the University of Houston System's 60-minute *Practical
Generative AI* workshop.

It runs with **no API key and no internet**. Nothing to sign up for.

```bash
git clone https://github.com/ishita199615/practical-genai-workshop.git
cd practical-genai-workshop

python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

No git? Use the green **Code** button above, then **Download ZIP**.

Full instructions, including installing Python: **[SETUP.md](SETUP.md)**

---

## What is this?

Most people meet generative AI as a chat box. You type, it answers, and that is
the whole loop. An **agent** is a different thing: it goes and gets current
information, runs real tools, produces a document, checks its own work, and then
stops and asks a human before doing anything that matters.

This repository is one complete agent — the **Cougar Career Agent** — plus a lab
that builds it up one idea at a time so you can see where each piece comes from.

The agent searches real, current job postings, scores them against a résumé,
drafts a tailored application, and refuses to write any claim the résumé does not
support. Then it stops and waits for you.

> **The résumé is fictional and the agent never applies to anything.** It reads
> public pages and writes a draft. There is no submit button, by design.

---

## Two ways in

Open the app and the sidebar has three pages.

### 📚 Learn the steps

Seven small steps. Each one states a single idea in plain English, shows the real
code behind it, runs that code, and prints what came back. The order is the
argument — every step exposes the problem the next one solves.

| # | Step | The problem it solves |
|---|------|----------------------|
| 1 | Prompt → Completion | What a model actually does, and why the same prompt gives different answers |
| 2 | The training cutoff | The model cannot know about jobs posted today |
| 3 | Retrieval | So fetch real pages instead of trusting the model's memory |
| 4 | **RAG** | Those pages are too big — chunk them, turn them into numbers, retrieve only what matters |
| 5 | Tools | Models are unreliable with numbers — let Python do the arithmetic |
| 6 | The agent loop | Wrap it all in Reason → Act → Observe |
| 7 | Guardrails | Refuse to lie, and make a human approve |

Every step runs offline. When the AI model is unreachable, a step says so on
screen and runs a deterministic version instead — it never pretends a call
happened that did not.

The RAG step is worth the price of admission on its own: it shows the chunks, the
vector dimensions, the similarity scores, **and the fully assembled prompt** that
gets sent to the model. That last part is the thing most tutorials never show you.

### 🎯 Full demo

The finished agent, running the whole pipeline end to end:

1. **Searches** current public job pages through Firecrawl
2. **Cleans and de-duplicates** what comes back, dropping closed and stale postings
3. **Scores** every job against the résumé with a transparent 100-point rubric
4. **Ranks** the top three and explains why
5. **Scores the résumé** for ATS readiness against the one job you pick
6. **Tells you what to change first**, citing which résumé line proves each change is honest
7. **Drafts** a revised summary, two revised bullets, and a short cover letter
8. **Re-scores** the draft under the identical rubric
9. **Validates** every claim against the original résumé
10. **Stops** and asks you to approve, request changes, or reject

Only after you approve does it write anything to disk.

---

## What it can do

**Find jobs that actually match you.** Choose your seniority — Internship, Entry,
Mid, Senior, Staff/Principal, or Manager — and the search is built for that level.
Ask for Senior and you get senior roles, not a pile of internships.

**Search recent postings.** Last hour, 24 hours, 3 days, or 7 days, across
LinkedIn, Indeed, Google, or company career sites (Greenhouse, Lever, Ashby,
SmartRecruiters, Workday).

**Show its work.** Every job card carries a real, clickable link to the original
posting, the full cleaned description, when it was retrieved, and where the
seniority reading came from.

**Explain every number.** Both scores are plain Python you can read and change:

| Demo Job Match Score | Demo ATS Readiness Score |
|---|---|
| Skills 45% | Keywords 40% |
| Text similarity 20% | Qualifications 20% |
| Job title 15% | Evidence in bullets 15% |
| Experience 10% | Sections present 10% |
| Location & work mode 10% | Parseable formatting 10% |
| | Contact details 5% |

**The AI model never touches either number.** It explains them; Python computes
them. Run the same search twice and you get the same scores.

---

## What makes it honest

This is the part worth stealing for your own projects.

**It will not invent a skill you don't have.** The fictional résumé lists Tableau
but not Power BI. When a job asks for Power BI, the agent shows you this instead
of quietly adding it:

```text
Job requirement:  Power BI
Evidence in résumé:  None
Decision:  Power BI was not added
Recommendation:  treat it as a learning gap
```

**A filter is not evidence.** Searching "posted in the last hour" narrows what
comes back; it does not prove any particular posting is an hour old. So a result
only reads *Verified* when the source actually published a timestamp. Otherwise it
says *Search-filtered; source timestamp unavailable*. The same rule applies to
seniority: a posting that never states a level is labelled **Level not stated**,
never quietly stamped with the level you searched for.

**Every revised line traces back to a real one.** A validator checks each claim
against the original résumé before you ever see it, and again after any revision.

**It degrades honestly.** When the AI is rate-limited or offline, the app says so
in plain language and continues on its deterministic path. Both scores are
unaffected, because they never depended on the model.

---

## The idea underneath

```text
Firecrawl  retrieves current public information
Python     calculates the scores
Gemini     explains, recommends, and drafts
Validator  checks every claim for truthfulness
A human    approves before anything is written
```

The model is one component out of five. **The model is not the agent — the agent
is the model plus tools, state, validation, decisions, and permissions.**

---

## What's in here

```text
app.py               Entry point — page config and navigation
pages/               The three screens (rendering only)
lessons/             The seven teaching steps, as plain testable logic
agent/               The LangGraph state machine that runs the demo
tools/               Retrieval, filtering, scoring, validation, export
services/            The AI model boundary, and multi-model routing
models/              Typed data structures for every boundary
data/                Fictional résumé and saved job postings
tests/               1,146 tests — no network, no API key
docs/                Runbook, engineering notes, and the specification
```

Built with Streamlit, LangGraph, Pydantic, scikit-learn, and Google Gemini.

---

## Make it yours

The point of having the source is changing it. Easiest first:

1. **Search your own career.** Change the role and set the experience level to
   where you actually are.
2. **Run it on your own résumé.** Copy the template, fill it in, check it with
   `python scripts/check_resume.py`, and point `RESUME_FILE` at it. Full steps,
   including what does and does not leave your machine, are in
   **[docs/USE_YOUR_OWN_RESUME.md](docs/USE_YOUR_OWN_RESUME.md)**. Your file is
   git-ignored, and the app stops calling the profile fictional once it is
   loaded.
3. **Change what counts.** [`tools/job_scorer.py`](tools/job_scorer.py) holds the
   weighting. Shift it, run `pytest`, and see what you broke.

Confirm you haven't broken anything at any point:

```bash
python -m pytest -q
```

---

## Ground rules

- **Fictional data only.** Do not paste your own résumé, or anyone else's, into
  this or any public AI tool during the workshop.
- **No university data.** Nothing classified Level 1 (Confidential) or Level 2
  (Protected) goes into any public AI model.
- **Public job listings only**, read-only, respecting each site's terms.
- **Keys stay local.** `.env` is git-ignored. Never paste a key into a chat, a
  screenshot, or a commit.
- This is an educational demonstration. The ATS readiness score is a transparent
  teaching rubric, not any employer's real applicant-tracking system, and it
  predicts nothing about a real application.
- Publishing this openly does not imply institutional endorsement.

---

## For instructors

- **[docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md)** — the eight-minute demo,
  minute by minute: what to click, what to say, and what to do when a live
  API misbehaves mid-session
- **[docs/RUNNING_WITHOUT_PAID_APIS.md](docs/RUNNING_WITHOUT_PAID_APIS.md)** —
  free and local model options (Ollama needs no key at all), plus where the
  saved job data actually comes from
- **[SETUP.md](SETUP.md)** — the student-facing install guide, Windows and macOS
- **[docs/ENGINEERING_NOTES.md](docs/ENGINEERING_NOTES.md)** — architecture, the
  eight-minute demo script, and what testing against the live APIs actually
  revealed
- **[docs/SPECIFICATION.md](docs/SPECIFICATION.md)** — the full specification this was built from
- `scripts/make_student_bundle.py` — packages a ZIP for students with secrets
  excluded and the archive scanned for keys before it is written

A note from the rehearsals, in case it saves you a bad five minutes: free AI tiers
meter quota **per model**, and some allow only 20 requests a day while one full
run uses about 13. That is why `LLM_MODELS` in `.env` lists several models — when
one is exhausted the app falls through to the next automatically. If they all run
dry, the demo keeps working on its offline path and simply says so.

---

## Licence

[MIT](LICENSE) — take it, change it, teach with it, build on it. Attribution is
appreciated but not required.

The licence covers the code. It does not extend to the University of Houston
System's name or branding, and publishing this openly does not imply
institutional endorsement.
