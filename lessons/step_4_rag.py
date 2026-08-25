"""Step 4 - RAG: chunk -> embed -> retrieve -> generate.

This is the step that shows *how* a model answers a question about documents it
was never trained on. Four visible stages, in order:

1. **Chunk** - every retrieved job description is cut into small overlapping
   pieces, because a whole document is too big and too vague to search.
2. **Embed** - each chunk becomes a vector of numbers with
   :class:`~sklearn.feature_extraction.text.TfidfVectorizer`. TF-IDF is used
   deliberately: it is already a project dependency, it runs offline, and it is
   perfectly deterministic, so the lab produces the same numbers in every room.
3. **Retrieve** - the question is turned into a vector the same way, then
   compared against every chunk with cosine similarity. Only the closest few
   survive.
4. **Generate** - the question plus *only* those chunks becomes the prompt. The
   assembled prompt is printed on screen, because seeing it is the whole lesson.

The step never requires the model. When Gemini is unavailable - which on the
free tier is most of the time - stages 1 to 3 are unchanged (they were never
model work in the first place) and stage 4 falls back to a deterministic
extractive answer that quotes the retrieved chunks and says so plainly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lessons.base import LessonContext, LessonResult, LessonStep, approx_tokens
from models.job import JobPosting
from tools.firecrawl_search import canonicalize_job_url
from tools.job_filter import dedup_key
from tools.job_normalizer import make_excerpt
from tools.job_scorer import canonical_skill, canonical_skill_set, skills_from_text

# Chunk geometry. Small enough that one chunk is about one idea, with an
# overlap so a sentence split across a boundary still appears whole somewhere.
CHUNK_CHARS = 500
CHUNK_OVERLAP = 80

# How many chunks are pasted into the prompt. Three is enough to answer the
# demo question and small enough to read on a projector.
TOP_K = 3

# The concrete question the lesson answers. It has a real, checkable answer in
# the cached job data, and it is the same gap the guardrail step uses later.
QUESTION = "Which of these jobs requires Power BI?"

# The skill the offline extractive path looks for, in canonical form.
TARGET_SKILL = canonical_skill("Power BI")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One searchable piece of one job description."""

    chunk_id: str
    job_id: str
    job_title: str
    company: str
    text: str

    @property
    def citation(self) -> str:
        """A short human-readable source label used in prompts and answers."""
        return f"{self.chunk_id} · {self.job_title} — {self.company}"


def split_into_chunks(
    text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Cut ``text`` into overlapping windows of roughly ``size`` characters.

    Whitespace is flattened first so the chunks read cleanly on screen. The
    windows step forward by ``size - overlap``, so consecutive chunks share
    their edges and no sentence is lost at a boundary.
    """
    flat = " ".join((text or "").split())
    if not flat:
        return []
    size = max(50, size)
    # Cap the overlap at half a window. Without this, an overlap set equal to
    # the window size would advance one character at a time and produce tens of
    # thousands of chunks - a hang on the projector if an instructor edits the
    # constants live to show what overlap does.
    step = max(1, size - max(0, min(overlap, size // 2)))
    chunks: list[str] = []
    for start in range(0, len(flat), step):
        window = flat[start : start + size].strip()
        if window:
            chunks.append(window)
        if start + size >= len(flat):
            break
    return chunks


def unique_postings(jobs: list[JobPosting]) -> list[JobPosting]:
    """Drop postings that are the same document as an earlier one.

    A retrieval index must not hold the same document twice: the copies compete
    for the same few retrieval slots and paste identical text into the prompt,
    which is the opposite of the point this step makes. The cached
    demonstration data deliberately contains one repeated posting - the same
    Greenhouse job under two different tracking URLs - so this runs for real.

    Uses the project's own duplicate rules (:func:`tools.job_filter.dedup_key`
    and :func:`tools.firecrawl_search.canonicalize_job_url`) rather than a
    lesson-local guess, so the lab and the agent agree on what a duplicate is.
    """
    seen_urls: set[str] = set()
    seen_keys: set[str] = set()
    unique: list[JobPosting] = []
    for job in jobs:
        canonical = canonicalize_job_url(job.source_url)
        key = dedup_key(job)
        if canonical in seen_urls or key in seen_keys:
            continue
        seen_urls.add(canonical)
        seen_keys.add(key)
        unique.append(job)
    return unique


def build_chunks(
    jobs: list[JobPosting], *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Chunk every job description and tag each piece with its source job."""
    chunks: list[Chunk] = []
    for job in jobs:
        for text in split_into_chunks(job.description, size=size, overlap=overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{len(chunks) + 1:03d}",
                    job_id=job.job_id,
                    job_title=job.title,
                    company=job.company,
                    text=text,
                )
            )
    return chunks


def retrieve(
    question: str, chunks: list[Chunk], *, top_k: int = TOP_K
) -> tuple[list[tuple[Chunk, float]], int, tuple[int, int], str | None]:
    """Embed the chunks and the question, then return the closest chunks.

    Returns ``(ranked, vocabulary_size, matrix_shape, warning)``. ``warning`` is
    a user-facing message when vectorization was impossible and a plain
    word-overlap fallback was used instead, so the screen never implies a
    similarity search that did not happen.
    """
    if not chunks:
        return [], 0, (0, 0), None
    texts = [chunk.text for chunk in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        matrix = vectorizer.fit_transform(texts)
        question_vector = vectorizer.transform([question])
        scores = cosine_similarity(question_vector, matrix)[0]
        vocabulary = len(vectorizer.vocabulary_)
        shape = (matrix.shape[0], matrix.shape[1])
    except Exception:  # noqa: BLE001 - a lesson must never crash the app
        scores = [_word_overlap(question, chunk.text) for chunk in chunks]
        ranked = _rank(chunks, scores, top_k)
        return (
            ranked,
            0,
            (len(chunks), 0),
            "Vectorization was unavailable, so retrieval used simple word overlap.",
        )
    return _rank(chunks, list(scores), top_k), vocabulary, shape, None


def _rank(
    chunks: list[Chunk], scores: list[float], top_k: int
) -> list[tuple[Chunk, float]]:
    """Pair chunks with scores and keep the highest, ties broken by order."""
    paired = list(zip(chunks, (float(score) for score in scores)))
    paired.sort(key=lambda item: (-item[1], item[0].chunk_id))
    return paired[: max(1, top_k)]


def _word_overlap(question: str, text: str) -> float:
    """Fraction of the question's words that appear in ``text``."""
    words = {word for word in re.findall(r"[a-z0-9]+", question.lower()) if len(word) > 2}
    if not words:
        return 0.0
    found = {word for word in words if word in text.lower()}
    return len(found) / len(words)


def build_augmented_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    """Assemble the prompt: instructions, the retrieved chunks, the question.

    This string is the entire difference between "ask a model what it
    remembers" and "ask a model about your documents".
    """
    lines = [
        "Answer the question using ONLY the context below.",
        "Cite the chunk id you used. If the context does not contain the answer,",
        'say "The retrieved context does not say." Do not use outside knowledge.',
        "",
        "CONTEXT",
    ]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.citation}] (similarity {score:.3f})")
        lines.append(chunk.text)
        lines.append("")
    lines.append(f"QUESTION: {question}")
    lines.append("ANSWER:")
    return "\n".join(lines)


def extractive_answer(
    retrieved: list[tuple[Chunk, float]], jobs: list[JobPosting]
) -> str:
    """Answer the question from the retrieved chunks with no model at all.

    Deterministic and quotable: it finds the sentence in each retrieved chunk
    that names the target skill, then reports whether that job lists the skill
    as required or preferred. It reads nothing outside the retrieved chunks.
    """
    by_id = {job.job_id: job for job in jobs}
    lines: list[str] = []
    seen: set[str] = set()
    for chunk, _score in retrieved:
        if TARGET_SKILL not in skills_from_text(chunk.text):
            continue
        # One line per job: two retrieved chunks may quote the same posting.
        if chunk.job_id in seen:
            continue
        seen.add(chunk.job_id)
        sentence = _sentence_mentioning(chunk.text, TARGET_SKILL)
        status = _requirement_status(by_id.get(chunk.job_id))
        lines.append(
            f"- **{chunk.job_title} — {chunk.company}** ({status})  \n"
            f'  "{sentence}"  \n'
            f"  Source: `{chunk.chunk_id}`"
        )
    if not lines:
        return (
            "The retrieved context does not say. None of the "
            f"{len(retrieved)} retrieved chunk(s) mentions Power BI, so the honest "
            "answer is that this question cannot be answered from the context that "
            "was supplied — not that no such job exists."
        )
    header = "Based only on the retrieved chunks, Power BI is named by:"
    return "\n".join([header, "", *lines])


def _sentence_mentioning(text: str, skill: str) -> str:
    """Return the first sentence in ``text`` that names ``skill``."""
    for sentence in _SENTENCE_RE.split(text):
        if skill in sentence.lower():
            return make_excerpt(sentence.strip(" -•"), 240)
    return make_excerpt(text, 240)


def _requirement_status(job: JobPosting | None) -> str:
    """Say whether the posting lists the target skill as required or preferred."""
    if job is None:
        return "listed in the description"
    if TARGET_SKILL in canonical_skill_set(job.required_skills):
        return "listed under required skills"
    if TARGET_SKILL in canonical_skill_set(job.preferred_skills):
        return "listed under preferred skills"
    return "named in the job description"


def run(ctx: LessonContext) -> LessonResult:
    """Run all four RAG stages and emit one labelled block per stage."""
    result = LessonResult()
    readable = [job for job in (ctx.jobs or []) if (job.description or "").strip()]
    jobs = unique_postings(readable)
    duplicates_dropped = len(readable) - len(jobs)
    if not jobs:
        result.add(
            "warning",
            "Nothing to retrieve from",
            "No job descriptions are loaded, so there is no corpus to chunk. "
            "Run the search on the Demo tab, or check the cached data file.",
        )
        return result

    # ---------------------------------------------------------------- 1. CHUNK
    chunks = build_chunks(jobs)
    chunk_help = (
        f"~{CHUNK_CHARS} characters each with a ~{CHUNK_OVERLAP} character "
        "overlap, so a sentence split across a boundary still survives whole "
        "in the neighbouring chunk."
    )
    if duplicates_dropped:
        chunk_help += (
            f" {duplicates_dropped} repeated posting(s) were dropped before "
            "indexing: the same job under two URLs would compete for the same "
            "retrieval slots and paste the same text into the prompt twice."
        )
    result.add(
        "metric",
        "1 · CHUNK — job descriptions cut into searchable pieces",
        {
            "value": f"{len(chunks)} chunks from {len(jobs)} job descriptions",
            "help": chunk_help,
        },
    )
    example = chunks[0]
    result.add(
        "code",
        f"One chunk of {len(chunks)}, exactly as it will be searched "
        f"({example.citation})",
        example.text,
    )

    # ---------------------------------------------------------------- 2. EMBED
    retrieved, vocabulary, shape, warning = retrieve(QUESTION, chunks, top_k=TOP_K)
    result.add(
        "metric",
        "2 · EMBED — every chunk becomes numbers",
        {
            "value": f"{shape[0]} chunks × {shape[1]} dimensions",
            "help": (
                f"TF-IDF vocabulary: {vocabulary} distinct terms. Each chunk is now a "
                "row of numbers, and rows can be compared mathematically."
            ),
        },
    )
    if warning:
        result.add("warning", "Retrieval ran in fallback mode", warning)
    result.add(
        "note",
        "What production systems do differently",
        "A production system swaps TF-IDF for dense embeddings from an embedding "
        "model and stores them in a vector database (pgvector, Pinecone, Chroma). "
        "The idea is identical and it is the whole idea of RAG: text becomes "
        "numbers, and numbers can be compared. TF-IDF is used here because it is "
        "free, offline, and gives the same answer every single time.",
    )

    # ------------------------------------------------------------- 3. RETRIEVE
    rows = [
        {
            "Rank": rank,
            "Similarity": round(score, 3),
            "Chunk": chunk.chunk_id,
            "From job": f"{chunk.job_title} — {chunk.company}",
            "Chunk text": make_excerpt(chunk.text, 180),
        }
        for rank, (chunk, score) in enumerate(retrieved, start=1)
    ]
    result.add(
        "markdown",
        "3 · RETRIEVE — the question we are asking",
        f"**{QUESTION}**\n\nThe question is turned into a vector the same way the "
        f"chunks were, then compared against all {len(chunks)} of them with cosine "
        f"similarity. Only the top {len(retrieved)} move on.",
    )
    result.add("table", f"Top {len(retrieved)} chunks by cosine similarity", rows)

    # ------------------------------------------------------------- 4. GENERATE
    prompt = build_augmented_prompt(QUESTION, retrieved)
    result.add(
        "code",
        "4 · GENERATE — the augmented prompt (this is the whole trick)",
        prompt,
    )

    try:
        answer, available = ctx.llm_text(prompt, temperature=0.1)
    except Exception:  # noqa: BLE001 - a lesson must never crash the app
        answer, available = None, False
    # A reply only counts as an answer when it is text with something in it. A
    # blank or non-text reply must take the offline path: rendering it as the
    # model's grounded answer would put an empty box on screen under a caption
    # claiming the model spoke, which is the one thing this lab must not do.
    reply = answer.strip() if isinstance(answer, str) else ""
    if available and reply:
        result.used_llm = True
        result.add("success", "Grounded answer from the model", reply)
        result.add(
            "note",
            "What the model was allowed to see",
            f"Only the {len(retrieved)} chunks above. It was not shown the other "
            f"{max(0, len(chunks) - len(retrieved))} chunks, and it was told to say "
            "so if the context did not contain the answer.",
        )
    else:
        result.llm_unavailable = True
        result.add(
            "note",
            "No answer came back from the model — the deterministic path ran instead",
            "The prompt above is exactly what would have been sent. Because no usable "
            "answer came back from the provider — no key, no quota, or an empty reply "
            "— the answer below was produced by Python reading the same retrieved "
            "chunks. It is a real answer over the real retrieved context, not a canned "
            "model response.",
        )
        result.add(
            "markdown",
            "Grounded answer (offline extractive path)",
            extractive_answer(retrieved, jobs),
        )

    # ------------------------------------------------- The cost of not chunking
    everything = "\n\n".join(job.description for job in jobs)
    full_tokens = approx_tokens(everything)
    prompt_tokens = approx_tokens(prompt)
    if prompt_tokens < full_tokens:
        percent = round(100 * (full_tokens - prompt_tokens) / full_tokens)
        cost_value = (
            f"{full_tokens:,} tokens → {prompt_tokens:,} tokens ({percent}% smaller)"
        )
        cost_help = (
            f"Pasting all {len(jobs)} full job descriptions costs about "
            f"{full_tokens:,} tokens every single question. Pasting only the top "
            f"{len(retrieved)} chunks costs about {prompt_tokens:,}. You pay less, "
            "you wait less, and the model has less irrelevant text to be distracted by."
        )
    else:
        # Retrieval only pays off once the corpus is bigger than the prompt. Say
        # so rather than printing "0% smaller" over a prompt that is larger.
        cost_value = f"{full_tokens:,} tokens → {prompt_tokens:,} tokens (no saving yet)"
        cost_help = (
            f"With only {len(jobs)} short posting(s) loaded, the instructions plus "
            f"{len(retrieved)} chunk(s) come to about {prompt_tokens:,} tokens against "
            f"{full_tokens:,} for the whole corpus, so retrieval saves nothing here. "
            "The saving is real once the corpus is larger than one prompt — which is "
            "the situation every real document set is in."
        )
    result.add(
        "metric",
        "Why retrieve at all — the cost argument",
        {"value": cost_value, "help": cost_help},
    )
    return result


TEACHING_CODE = '''# 1. CHUNK — long job descriptions become small, overlapping pieces.
jobs = unique_postings(ctx.jobs)         # never index the same document twice
chunks = build_chunks(jobs)              # ~500 chars, ~80 char overlap

# 2. EMBED — every chunk becomes a row of numbers.
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])

# 3. RETRIEVE — embed the question the same way, keep the closest chunks.
question = "Which of these jobs requires Power BI?"
scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
top_3 = sorted(zip(chunks, scores), key=lambda pair: -pair[1])[:3]

# 4. GENERATE — the model sees the question and ONLY those three chunks.
prompt = build_augmented_prompt(question, top_3)
answer, available = ctx.llm_text(prompt)

if not available or not answer:          # free tiers rate limit constantly
    answer = extractive_answer(top_3, jobs)       # same chunks, no model
'''


STEP = LessonStep(
    number=4,
    title="RAG: chunk → embed → retrieve → generate",
    subtitle="How a model answers questions about documents it never saw in training",
    concept=(
        "A language model only knows what it was trained on, and it cannot read a "
        "pile of documents you hand it all at once. RAG fixes this in four steps: "
        "cut the documents into small chunks, turn each chunk into numbers, find "
        "the few chunks closest to the question, and paste only those into the "
        "prompt."
    ),
    why=(
        "This is how nearly every real 'chat with your documents' product works, "
        "and it is why the answer can cite a source instead of guessing. It is also "
        "far cheaper: you send the model three relevant paragraphs instead of five "
        "entire job postings, for every single question."
    ),
    deck_reference="Slide 8 — Retrieval-Augmented Generation",
    code=TEACHING_CODE,
    run=run,
    takeaway=(
        "RAG is not magic: it is search that runs before the prompt, so the model "
        "answers from your text instead of from memory."
    ),
    needs_llm=False,
)
