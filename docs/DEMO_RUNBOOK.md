# Demo 2 runbook — Cougar Career Agent

Eight minutes, start to finish. The trick is starting the search early and
talking over it: the agent needs about ninety seconds, and that time is where
the teaching happens.

| | |
| --- | --- |
| Runtime | 8 minutes |
| Live search | ~90 seconds |
| Cached | instant |
| Set up by | T-15 minutes |

---

## Before the room fills

Fifteen minutes ahead. The one that matters is the rehearsal run — it warms the
connection and tells you whether your quota is alive today.

- [ ] Start the app: `streamlit run app.py`
- [ ] **Do one complete run** — search, select a job, generate the patch
- [ ] Confirm the badge reads **🟢 LIVE PUBLIC RESULTS**. If 🟡, see [When it goes wrong](#when-it-goes-wrong)
- [ ] Empty `output/` so the export you produce is visibly new
- [ ] Reload, leave it on **Full demo** with defaults filled in
- [ ] Browser zoom ~110%, close every other tab — you will click a real job link

> **One full run costs about 13 AI calls.** The free tier meters quota per model
> and some allow only 20 a day. Rehearsal plus demo is ~26. Budget for exactly
> two runs, or accept the fallback — which makes a good point on its own.

---

## The eight minutes

### 0:00 — Frame what an agent is *(40 sec)*

Don't show the app yet. Land the distinction first; everything after is evidence.

> A chatbot gives job-search advice. This agent retrieves current jobs, uses
> tools, calculates a score, modifies a document, validates its own output, and
> pauses for approval.

### 0:40 — Set the inputs and start it running *(30 sec)*

```text
Target role       Data Analyst Intern
Location          Houston, TX
Work mode         Any
Experience level  Internship
Job source        Direct Company Careers
Freshness         Last 24 hours
```

Press **Run Career Agent** *now*, then keep talking. Do not wait in silence —
you have ninety seconds of runway and the next beat fills it.

### 1:10 — Narrate the activity log while it works *(70 sec)*

The log fills in live. Read it out; it is the agent showing its work.

> Watch what it is doing. It built a search query for internships specifically.
> It went to the live web. It threw away three pages that were closed or
> duplicates. Now it is scoring what survived — in Python, not in the model.

Still running? Mention **Learn the steps**: the same ideas broken into seven,
and what they will install afterwards.

### 2:20 — The results are real, prove it *(50 sec)*

- Point at the **🟢 LIVE** badge and retrieval timestamp
- Point at the source badges — Greenhouse, Ashby, LinkedIn — detected from the
  URL, not from what you searched
- **Click `Open job posting ↗`** on the top card. A real posting opens. Come back.

> The model did not remember these jobs. It could not — they were posted after
> it finished training. Firecrawl went and got them, minutes ago.

Clicking the link is the most convincing thing in the demo. Don't skip it.

### 3:10 — Open the description, then select the job *(40 sec)*

- Expand **View job description** — the full cleaned posting, not a snippet
- Select the **Lakeside Analytics** job (the one asking for Power BI)

> The score is calculated against this text — the actual posting — not against a
> summary the model wrote about it. Python calculates the number. The model only
> explains it.

### 3:50 — ATS readiness *(50 sec)*

```text
Demo ATS Readiness Score: 83/100 — Strong
Estimated using this demo rubric; not an official employer ATS score.
```

> This is a transparent readiness estimate, not a secret employer score. Six
> components, all visible, all arithmetic you could do by hand.

### 4:40 — What to change first *(50 sec)*

- Open **What to change first**
- Read one **safe** recommendation aloud — note the résumé ID it cites
- Then point at one marked **Safe to apply: No**

> Every suggested change points at the line of the résumé that proves it is
> honest. If it cannot find that evidence, it will not suggest the change.

### 5:30 — Generate the patch and re-score *(50 sec)*

Show the revised summary, the two bullets beside their originals, the projection.

```text
Original estimated score:    83/100
Projected under same rubric: 92/100
```

> Same rubric, re-run on the rewritten résumé. Nothing was invented — every
> bullet still traces to the one it came from.

### 6:20 — The guardrail, your strongest moment *(40 sec)*

```text
Job requirement:     Power BI
Evidence in résumé:  None
Decision:            Power BI was not added
Recommendation:      treat it as a learning gap
```

> The job asks for Power BI. The résumé cannot support it. So the agent refuses
> to add it — and tells you why. That refusal is not a limitation. It is the
> entire point.

Slow down here. This is the beat people remember.

### 7:00 — The approval gate *(30 sec)*

Show **Approve · Request Changes · Reject**. Hover Approve. Don't click yet.

> It has stopped. It will not write a file, let alone send anything, until a
> human says so. There is no submit button anywhere in this application — that
> is a design decision, not a missing feature.

Then click **Approve** and show the download buttons appear.

### 7:30 — Reveal the architecture *(30 sec)*

```text
Firecrawl   retrieved current public information
Python      calculated every score
Gemini      explained and drafted
Validator   checked each claim
A human     approved
```

> The model is one of five parts. The model is not the agent — the agent is the
> model plus tools, state, validation, decisions, and permissions.

Close with the repo link on screen. Fifteen minutes to install, no API key.

---

## When it goes wrong

None of these end the demo. Two arguably improve it.

| What happens | What you do, and what you say |
| --- | --- |
| 🟡 Cached instead of live | Switch **Job source** to **All Public Sources** and re-run — three cards across LinkedIn, Ashby, Greenhouse. *"The live search failed, so it fell back to saved data and told us so rather than pretending."* |
| Amber quota banner | Carry on. Every score is identical; only the wording is blunter. *"The model just went offline mid-demo. The scores, the ranking, the guardrail, the approval gate — all still working. That is the point."* |
| Only two job cards | Expected on cached data with Direct Company Careers. Point at the removal notes: *"it dropped an entry-level posting because we asked for internships, and it told us."* |
| No results at all | **Freshness** to **Last 7 days**, or **Experience level** to **Any level**. Note that the app never widens a search behind your back. |
| Everything misbehaving | Set `OFFLINE=true` in `.env`, restart, use All Public Sources. Runs in 0.03s with no network at all, and every beat still lands. See [Running standalone](#running-standalone-with-no-apis-at-all). |
| Running out of time | Skip 5:30. Jump from the ATS panel straight to the Power BI guardrail and the approval gate. Those two are the demo. |

---

## If you only get four minutes

```text
0:00  Frame it — a chatbot advises, an agent acts
0:20  Run with the defaults, keep talking
1:30  Live badge, source badges, click a real job link
2:10  Select the job, show the ATS score and disclaimer
2:50  The Power BI refusal
3:20  Approve · Request Changes · Reject — nothing submitted
3:45  Five parts. The model is not the agent.
```

Run it with `OFFLINE=true` for this version: no waiting, no quota risk.

---

## Questions you will get

| Question | Answer |
| --- | --- |
| Does it apply for me? | No. It reads public pages and writes a draft. There is no submit button, deliberately. |
| Can I use my own résumé? | Yes, after today on your own machine — [USE_YOUR_OWN_RESUME.md](USE_YOUR_OWN_RESUME.md). Not during the session, and not with anyone else's. |
| Is that a real ATS score? | No. A transparent rubric written for teaching. Real systems are proprietary and all different. |
| What does it cost? | Nothing to run. Free tiers for both APIs, and it works with no key at all. |
| Could it lie on my résumé? | It is built specifically not to. Every claim traces to a line you wrote, and a validator checks each one. |
| Do I need to know Python? | To run it, no. To change it, a little — the scoring file is about a hundred readable lines. |

---

## Running standalone, with no APIs at all

The most reliable way to present this. One line in `.env`:

```text
OFFLINE=true
```

Restart the app. That is the whole procedure. Your keys can stay in the file —
they are simply not used.

What you get: no network call, no quota to exhaust, no conference wifi to
depend on, and a **complete run in about 0.03 seconds** instead of ninety.

| Still works | Changes |
| --- | --- |
| Both scores, identical numbers | Postings come from saved data, badged 🟡 |
| Ranking and the top three | Wording is templated, not AI-written |
| ATS rubric and all six components | No "posted an hour ago" moment |
| "What to change first" | |
| The Power BI refusal | |
| Claim validation | |
| Approval gate and export | |
| All seven Learn steps | |

Nothing about the teaching is lost. Every number is identical because no number
ever came from the model — which is the argument the whole demo is making.

**Set `Job source` to `All Public Sources`** for three cards across LinkedIn,
Ashby, and Greenhouse. Direct Company Careers yields two from the saved data.

**One line to add if you present this way**, when someone asks whether it is
really working:

> This is running with the internet switched off. Every score you see was
> calculated on this laptop. The only thing I gave up was the live search and
> some nicer phrasing — not one number moved.
