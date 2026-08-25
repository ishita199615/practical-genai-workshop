"""Freeze a real Firecrawl run into a cache file for offline use.

The cache the project ships with is *synthetic* — fictional companies and URLs
written for the workshop. It is honest, and it is labelled as such everywhere it
appears, but the links do not go anywhere.

With a working Firecrawl key you can capture a genuine run instead. The result
behaves identically offline, except the postings are real and their links open.

    python scripts/capture_live_cache.py --role "Data Analyst Intern"

Writes ``data/cached_jobs.local.json`` by default, which is git-ignored, then
point the app at it::

    CACHE_FILE=data/cached_jobs.local.json

Why not overwrite the shipped cache: a captured file contains real employers'
posting text. Keeping it out of the repository avoids republishing someone
else's content, and leaves students with the fictional data instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings  # noqa: E402
from models.job import ExtractedJobFields  # noqa: E402
from services.router_client import RouterClient  # noqa: E402
from tools.firecrawl_search import (  # noqa: E402
    FirecrawlError,
    FirecrawlSearchAdapter,
    build_search_request,
    raw_results_to_models,
    search_with_domain_retry,
)
from tools.job_normalizer import clean_description, extract_job_fields  # noqa: E402

DEFAULT_OUTPUT = "data/cached_jobs.local.json"


def build_notice(query: str, when: datetime) -> str:
    """Describe the capture honestly inside the file itself."""
    return (
        "Real public job postings captured from a live Firecrawl run on "
        f"{when.date().isoformat()} for the query {query!r}. These are genuine "
        "employer pages, frozen at capture time: the links are real but the "
        "postings may since have closed. Not synthetic. Do not republish this "
        "file, as it contains other organisations' posting text."
    )


def main() -> int:
    """Run one live search and write the results as a cache file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", default="Data Analyst Intern")
    parser.add_argument("--location", default="Houston, TX")
    parser.add_argument("--work-mode", default="Any")
    parser.add_argument(
        "--category",
        default="all",
        choices=["all", "linkedin", "indeed", "google_jobs", "company_careers"],
        help="'all' captures the widest mix of sources, which demos best",
    )
    parser.add_argument(
        "--freshness",
        default="last_7_days",
        choices=["last_hour", "last_24_hours", "last_3_days", "last_7_days"],
        help="a wider window captures more postings; freshness is re-derived at run time",
    )
    parser.add_argument("--level", default="internship")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="skip the AI extraction pass (faster, but the app must extract at run time)",
    )
    args = parser.parse_args()

    settings = load_settings()
    if settings.offline:
        print("OFFLINE=true is set in .env. Turn it off to capture a live run.")
        return 1
    if not settings.firecrawl_api_key:
        print("FIRECRAWL_API_KEY is not set, so there is nothing to capture from.")
        return 1

    adapter = FirecrawlSearchAdapter(
        api_key=settings.firecrawl_api_key, base_url=settings.firecrawl_base_url
    )
    now = datetime.now(timezone.utc)
    request = build_search_request(
        role=args.role,
        location=args.location,
        work_mode=args.work_mode,
        query_category=args.category,
        freshness_window=args.freshness,
        experience_level=args.level,
        limit=settings.max_job_results,
        timeout_seconds=max(settings.search_timeout_seconds, 30),
        now=now,
    )

    print(f"Searching: {request.query}")
    print(f"  tbs={request.tbs}  limit={request.limit}")
    try:
        outcome = search_with_domain_retry(adapter, request)
    except FirecrawlError as exc:
        print(f"Firecrawl failed: {exc}")
        return 1
    results = outcome.results
    for note in outcome.notes:
        print(f"  note: {note}")
    if not outcome.time_filter_applied:
        print("  note: the time filter was dropped to get results back.")
    if not results:
        print("No results. Try a wider --freshness, or --category all.")
        return 1

    raw_jobs, warnings = raw_results_to_models(
        results,
        query_category=args.category,
        freshness_window=args.freshness,
        retrieved_at=now,
    )
    for warning in warnings:
        print(f"  {warning}")
    print(f"Retrieved {len(raw_jobs)} usable page(s).")

    llm = None
    if not args.no_extract and settings.has_gemini:
        llm = RouterClient(models=settings.model_chain, api_key=settings.gemini_api_key)
        print(f"Extracting fields with {settings.model_chain[0]}...")

    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_jobs, start=1):
        description = clean_description(
            raw.markdown or raw.description, settings.max_job_description_chars
        )
        extracted: ExtractedJobFields | None = None
        if llm is not None:
            extracted = extract_job_fields(raw, description, llm)
            label = extracted.title if extracted else "extraction failed"
            print(f"  {index}. {label}")

        metadata = dict(raw.metadata)
        if extracted is not None:
            metadata["cached_extraction"] = extracted.model_dump(mode="json")
        entries.append(
            {
                "url": raw.url,
                "final_url": raw.final_url,
                "title": raw.title,
                "description": description or raw.description,
                "markdown": raw.markdown,
                "detected_source_category": raw.detected_source_category,
                "detected_source_label": raw.detected_source_label,
                "metadata": metadata,
            }
        )

    payload = {
        "cache_label": "CACHED DEMONSTRATION RESULTS",
        "synthetic": False,
        "data_notice": build_notice(request.query, now),
        "originally_retrieved_at": now.isoformat(),
        "capture": {
            "role": args.role,
            "location": args.location,
            "work_mode": args.work_mode,
            "query_category": args.category,
            "freshness_window": args.freshness,
            "experience_level": args.level,
            "query": request.query,
        },
        "modification_log": [
            f"{now.date().isoformat()} - captured from a live Firecrawl run; "
            "no posting text was edited."
        ],
        "jobs": entries,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    relative = (
        out_path.relative_to(PROJECT_ROOT)
        if out_path.is_relative_to(PROJECT_ROOT)
        else out_path
    )
    print(f"\nWrote {len(entries)} posting(s) to {relative.as_posix()}")
    print("\nTo use it, add this to .env:")
    print(f"  CACHE_FILE={relative.as_posix()}")
    if relative.name != "cached_jobs.json":
        print("\nThis file is git-ignored. It holds real employers' posting text,")
        print("so keep it off the public repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
