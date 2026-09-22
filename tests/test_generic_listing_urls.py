"""Tests that search-result pages never reach the ranking as if they were jobs.

Every URL here came out of a real Firecrawl capture. A results page scoring
above genuine openings is worse than a missing result: it looks like an answer.
"""

from __future__ import annotations

import pytest

from tools.firecrawl_search import looks_like_generic_listing

LISTING_PAGES = [
    # Glassdoor search results, which ranked first before this was fixed.
    "https://www.glassdoor.co.uk/Job/remote-data-analyst-internship-jobs-SRCH_IL.0,6_IS11047.htm",
    "https://in.indeed.com/q-data-analyst-internship-l-remote-jobs.html",
    "https://www.virtualvocations.com/jobs/q-remote+data+analyst+intern+jobs/c-bachelors",
    "https://example.com/data-analyst-jobs",
    "https://example.com/analytics-internships",
    "https://example.com/current-vacancies",
    "https://www.linkedin.com/jobs/search?keywords=data%20analyst",
    "https://boards.greenhouse.io/",
    "https://example.com/careers",
    "https://example.com/jobs",
    # An employer's whole board on an ATS: the slug alone, with no posting
    # after it. These reached the ranking titled "Jobs" and were scored.
    "https://job-boards.greenhouse.io/anthropic",
    "https://job-boards.greenhouse.io/grafanalabs/",
    "https://jobs.lever.co/portcast?department=Data%20Science",
    "https://jobs.ashbyhq.com/console",
]

REAL_OPENINGS = [
    "https://job-boards.greenhouse.io/lakesideanalytics/jobs/4410021",
    "https://jobs.lever.co/acme/6f1a-data-analyst-intern",
    "https://www.linkedin.com/jobs/view/4188220561",
    "https://rayda.zohorecruit.com/jobs/Careers/759575000002968033/Data-Analyst-Intern",
    "https://www.remotefront.com/remote-jobs/ing-data-analyst-internship-2asor",
    "https://jobs.ashbyhq.com/gulfcoast/8b21-bi-intern",
    "https://www.indeed.com/viewjob?jk=abc123",
    # Stripe hosts its Greenhouse board on its own domain and gives every
    # opening a search path. The id is what names the job.
    "https://stripe.com/jobs/search?gh_jid=8172508",
    "https://www.linkedin.com/jobs/search?currentJobId=4188220561",
    "https://www.indeed.com/jobs?jk=abc123",
]


@pytest.mark.parametrize("url", LISTING_PAGES)
def test_listing_pages_are_rejected(url):
    assert looks_like_generic_listing(url) is True, f"should reject {url}"


@pytest.mark.parametrize("url", REAL_OPENINGS)
def test_real_openings_survive(url):
    assert looks_like_generic_listing(url) is False, f"should keep {url}"


def test_a_job_id_ending_in_the_word_jobs_is_not_a_listing():
    """The suffix rule must not fire mid-path.

    "/jobs/…" appears in almost every real posting URL, so only a trailing
    segment like "…-analyst-jobs" counts.
    """
    assert looks_like_generic_listing("https://x.com/jobs/senior-data-analyst") is False


def test_a_search_page_without_a_posting_id_is_still_a_listing():
    """Only an id rescues a search path — a query alone does not.

    The same LinkedIn path is a listing or an opening depending on whether it
    names a job, so the two must not be conflated.
    """
    assert looks_like_generic_listing(
        "https://www.linkedin.com/jobs/search?keywords=data%20analyst"
    ) is True
    assert looks_like_generic_listing("https://stripe.com/jobs/search") is True
    # A tracking parameter that merely resembles an id must not rescue it.
    assert looks_like_generic_listing(
        "https://stripe.com/jobs/search?gh_jid_src=linkedin"
    ) is True
