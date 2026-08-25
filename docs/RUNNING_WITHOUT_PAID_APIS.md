# Running it without paid APIs

Two questions:

1. [I want the AI parts, but cheaper or free](#1-cheaper-model-options)
2. [I want to run with no APIs at all](#2-no-apis-at-all) — and *where does the
   saved data come from?*

---

## 1. Cheaper model options

The app routes through [LiteLLM](https://docs.litellm.ai), so it already speaks
to most providers. You change one line in `.env` — no code edits.

```text
LLM_MODELS=<provider>/<model>,<provider>/<model>
```

Models are tried in order. A rate limit or outage falls through to the next.

### Ollama — free, local, no key at all

The only option that needs no account and has no quota. Models run on your own
machine.

```bash
# install from https://ollama.com, then:
ollama pull llama3.1
```

```text
LLM_MODELS=ollama/llama3.1
```

That is the whole setup. **Verified:** the router recognises `ollama/…` as
keyless and reports itself available with no API key present.

Trade-offs worth telling students honestly: it needs a reasonably capable
machine (8 GB RAM for a small model, more for a good one), it is slower than a
hosted API on most laptops, and smaller models are noticeably worse at the
structured extraction step — you will see more extractions fail and fall back.
Nothing breaks; the deterministic paths take over.

### OpenRouter — one key, many models, some free

Free-tier models exist alongside paid ones, and one key reaches all of them.

```text
OPENROUTER_API_KEY=sk-or-...
LLM_MODELS=openrouter/meta-llama/llama-3.3-70b-instruct:free
```

### Groq — free tier, very fast

```text
GROQ_API_KEY=gsk_...
LLM_MODELS=groq/llama-3.3-70b-versatile
```

### Mixing them

The point of the chain is that it crosses providers. A robust free setup:

```text
LLM_MODELS=groq/llama-3.3-70b-versatile,openrouter/meta-llama/llama-3.3-70b-instruct:free,ollama/llama3.1
```

Hosted models first because they are faster; the local model last because it can
never run out.

### What about OpenCode, Cursor, Claude Code?

Those are coding assistants, not model providers — they help you *write* the
app, they do not serve it a model at run time. Not applicable here.

### One thing that does not change

**No model choice affects either score.** Both are computed in Python. A weaker
model produces blunter wording and more failed extractions; every number stays
identical. That is worth demonstrating rather than asserting: switch models and
re-run the same search.

---

## 2. No APIs at all

One line:

```text
OFFLINE=true
```

No network call of any kind. A complete run takes about 0.03 seconds. Keys can
stay in `.env` — they are simply not read.

Everything that matters still runs: both scores, ranking, the ATS rubric, the
"what to change first" panel, the Power BI refusal, claim validation, the
approval gate, export, and all seven Learn steps.

### So where does the job data come from?

Fair question, and the honest answer has two halves.

**The data shipped in this repository is synthetic.** `data/cached_jobs.json`
contains fictional companies — Lakeside Analytics, Gulf Coast Retail Group,
Northline Energy Services — written for the workshop. They are shaped like real
Greenhouse, Ashby, and LinkedIn pages, and one deliberately asks for Power BI so
the guardrail has something to refuse. **The URLs do not go anywhere.**

The file says so about itself:

```json
"synthetic": true,
"data_notice": "Synthetic demonstration data written for the ... workshop.
   Company names, postings, and URLs are fictional examples shaped like real
   public applicant-tracking-system pages ... No real employer posting is
   reproduced here."
```

and the app repeats it on screen whenever cached data is in use, rather than
letting you assume the links are live.

**You can replace it with real postings.** If you have a Firecrawl key, capture
a genuine run and freeze it:

```bash
python scripts/capture_live_cache.py --category company_careers --freshness last_7_days
```

That writes `data/cached_jobs.local.json`. Point the app at it:

```text
CACHE_FILE=data/cached_jobs.local.json
```

Now offline mode replays real postings with links that open. The app notices the
difference and changes what it claims: a captured cache is described as *"real
postings captured in an earlier live run, so the links are genuine but the
postings may since have closed"* rather than as synthetic examples.

Two cautions, both learned by running it:

- **Use `--category company_careers`.** A capture with `--category all` pulls in
  Glassdoor and Indeed *search-result* pages, aggregator sites, and — genuinely —
  an Instagram post. The filters reject those, but you are left with few usable
  postings. Restricting to applicant-tracking domains gives a much cleaner set.
- **The captured file is git-ignored on purpose.** It holds other organisations'
  posting text. Keep it local rather than republishing it.

### Which combination should a student use?

| Situation | Setting |
| --- | --- |
| No accounts, no setup, just wants it working | `OFFLINE=true` |
| Wants the AI parts, has a decent laptop | Ollama, and leave Firecrawl unset |
| Wants live jobs but no AI cost | Firecrawl key + `LLM_MODELS=ollama/llama3.1` |
| Wants the full live experience | Both free tiers, as in [SETUP.md](../SETUP.md) |

The first row is the honest default. The demo was built so that the parts that
teach the lesson never depended on a paid service.
