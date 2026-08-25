# Setting up the Cougar Career Agent

Practical Generative AI · University of Houston System workshop.

About fifteen minutes, most of it waiting on a download. **You do not need an API
key** — the app ships with saved job postings and works completely offline. Keys
are optional and only add live web search.

| | |
| --- | --- |
| Time | ~15 minutes |
| Disk space | ~1 GB |
| Python | 3.11 or 3.12 (3.10 works too) |
| API keys | None required |
| Code | [github.com/ishita199615/practical-genai-workshop](https://github.com/ishita199615/practical-genai-workshop) |

---

## 1. Install Python

Check what you have first:

```bash
python --version
```

macOS: use `python3 --version`.

If that prints 3.11 or higher, skip to step 2. Otherwise install from
<https://www.python.org/downloads/>.

**Windows: on the first screen of the installer, tick "Add python.exe to PATH".**
It is easy to miss and nothing below works without it. If you already installed
without it, run the installer again and choose Modify.

Close and reopen your terminal afterwards, then check the version again.

## 2. Get the code

The project lives here:

**<https://github.com/ishita199615/practical-genai-workshop>**

Pick whichever route suits you. Both give you the same files.

### Option A — Download ZIP (no extra software)

1. Open the link above
2. Click the green **Code** button, then **Download ZIP**
3. Unzip it to your Desktop

You get a folder named **`practical-genai-workshop-main`**. Note the `-main` on
the end — GitHub adds it, and you will need it in the next step.

### Option B — Clone it (if you have git)

```bash
git clone https://github.com/ishita199615/practical-genai-workshop.git
```

You get a folder named **`practical-genai-workshop`**, with no suffix.

> **No internet?** Your instructor can hand you
> `practical-genai-workshop-offline.zip`, which unzips to
> `practical-genai-workshop` — the same as Option B.

## 3. Open a terminal in that folder

Use the folder name from whichever option you chose above. These examples assume
Option A, so drop the `-main` if you cloned.

Windows PowerShell:

```powershell
cd "$HOME\Desktop\practical-genai-workshop-main"
```

macOS Terminal:

```bash
cd ~/Desktop/practical-genai-workshop-main
```

Run `dir` (Windows) or `ls` (macOS). You should see `app.py`, `requirements.txt`,
and folders named `lessons`, `agent`, and `tools`. If you get "no such file or
directory", the folder name is probably the other one — check with `ls ~/Desktop`
(macOS) or `dir $HOME\Desktop` (Windows).

## 4. Create a virtual environment

This keeps the project's packages separate from your system Python. Run both
lines — the second turns it on.

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell says *"running scripts is disabled on this system"*, run this and
then activate again. It affects only the current window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Your prompt should now start with `(.venv)`. If you close the terminal, run the
activate line again before continuing.

## 5. Install the packages

About 700 MB, so this is the slow part.

```bash
python -m pip install -r requirements.txt
```

Yellow warnings are fine. Only a red `ERROR` matters. The last line should read
`Successfully installed ...`.

## 6. Run it

```bash
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Three pages appear in the sidebar:

```text
🎓  Home
📚  Learn the steps
🎯  Full demo
```

Open **Learn the steps** and click *Run step 1*, then work down through all
seven. Press `Ctrl + C` in the terminal to stop the app.

Each step shows the real code it just ran. When one interests you, open the
matching file in `lessons/` — step 4 is `lessons/step_4_rag.py`. The finished
agent is assembled in `agent/graph.py`.

## 7. Optional — live mode

Everything already works. Adding free API keys makes the agent search the live
web for jobs posted today instead of using saved ones, and lets the AI write the
explanations. **The scores do not change** — they are always calculated in
Python, never by the model.

1. Gemini key: <https://aistudio.google.com/apikey> (free, Google account)
2. Firecrawl key: <https://firecrawl.dev> (free tier, no card)
3. Copy `.env.example` to a new file named exactly `.env`
4. Paste each key after the `=`, no quotes, no spaces

```powershell
Copy-Item .env.example .env    # Windows
```

```bash
cp .env.example .env           # macOS
```

Restart the app to pick up the keys. On the Full demo page the badge should read
🟢 **LIVE PUBLIC RESULTS** instead of 🟡 cached.

**The free Gemini tier runs out fast, and that is a lesson rather than a bug.**
Google meters quota *per model* — some models allow only 20 requests a day, and
one full run uses about 13. That is why `.env` lists several models in
`LLM_MODELS`: when one is exhausted the app falls through to the next. If they
all run dry you get an amber notice and the app continues on its offline path.

---

## Prove it works

```bash
python -m pytest -q
```

Expect `1103 passed`. It makes no network calls and needs no API key. This is
also the fastest way to check you have not broken anything once you start
editing.

---

## When something goes wrong

| What you see | What to do |
| --- | --- |
| `'python' is not recognized` | Python is not on your PATH. Reinstall and tick *Add python.exe to PATH*, or try `py`. On macOS use `python3`. |
| `running scripts is disabled on this system` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `'streamlit' is not recognized` | The virtual environment is not active — your prompt should start with `(.venv)`. Activate it, or run `python -m streamlit run app.py`. |
| `cannot import name 'NotRequired'` | An installed package is too new for your Python. Re-run `python -m pip install -r requirements.txt`; the file pins a compatible version. |
| `Port 8501 is already in use` | The app is running in another window. Close it, or use `streamlit run app.py --server.port 8502`. |
| `429 RESOURCE_EXHAUSTED` | Free Gemini quota for that model is spent. The app falls through to the next model, then to its offline path. Carry on. |
| `404 ... no longer available to new users` | That model is retired. Remove it from `LLM_MODELS` in `.env`. |
| 🟡 cached instead of 🟢 live | Expected without a Firecrawl key. With one, the search failed or timed out and the app fell back rather than showing nothing. |
| Nothing found for your search | Widen it: Freshness *Last 7 days*, or Experience level *Any level*. The app never quietly widens a search for you. |

---

## Ground rules

- **Use the fictional résumé.** Do not paste your own, or anyone else's, into
  this or any public AI tool during the workshop.
- **No university data.** Nothing classified Level 1 (Confidential) or Level 2
  (Protected) goes into any public AI model.
- **The agent never applies to anything.** It reads public pages and writes a
  draft. There is no submit button, by design.
- **Your keys are yours.** `.env` is excluded from sharing. Never paste a key
  into a chat, a screenshot, or a repository.

---

## Then make it yours

- **Search your own career.** On the Full demo page, change the role and set the
  experience level to where you actually are.
- **Rewrite the résumé.** `data/sample_resume.json` is plain text with a made-up
  person in it. Invent a different fictional candidate and watch every score move.
- **Change what counts.** `tools/job_scorer.py` decides how a job is scored —
  skills 45%, text similarity 20%, title 15%, experience 10%, location 10%.
  Shift the weights, run `pytest`, see what you broke.
