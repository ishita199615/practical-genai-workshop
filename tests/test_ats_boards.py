"""Reading postings from ATS APIs: parsing, filtering, and failure behaviour.

No network. Every board payload here is a fixture, so the suite stays offline.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from tools import ats_boards
from tools.ats_boards import (
    AtsPosting,
    Employer,
    _lever_work_mode,
    _parse_epoch_ms,
    _parse_iso,
    _parse_workday_age,
    _strip_html,
    fetch_board,
    load_employers,
    matches_location,
    read_greenhouse,
    read_lever,
    read_workday,
    search,
)

ANALYST = re.compile(r"analyst|data", re.IGNORECASE)
HOUSTON = ["houston", "sugar land"]


@pytest.fixture
def stub_request(monkeypatch):
    """Replace the HTTP layer with a canned response."""

    def _install(payload):
        monkeypatch.setattr(ats_boards, "_request", lambda *a, **k: payload)

    return _install


def posting(**overrides) -> AtsPosting:
    base = dict(
        title="Data Analyst",
        company="Acme",
        location="Houston, Texas, United States",
        url="https://example.test/1",
        ats="workday",
    )
    base.update(overrides)
    return AtsPosting(**base)


class TestDateParsing:
    """A board's own date is read; nothing is invented when it states none."""

    def test_iso_timestamp(self):
        parsed = _parse_iso("2026-08-19T14:02:07-04:00")
        assert parsed == datetime(2026, 8, 19, 18, 2, 7, tzinfo=timezone.utc)

    def test_naive_iso_is_treated_as_utc(self):
        assert _parse_iso("2026-08-19T14:02:07").tzinfo == timezone.utc

    def test_epoch_milliseconds(self):
        assert _parse_epoch_ms(1701703519526).year == 2023

    @pytest.mark.parametrize("value", [None, "", "not a date", {}])
    def test_unparseable_dates_are_none(self, value):
        assert _parse_iso(value) is None
        assert _parse_epoch_ms(value) is None

    def test_workday_today_and_yesterday(self):
        now = datetime.now(timezone.utc)
        today_age = (now - _parse_workday_age("Posted Today")).total_seconds()
        yesterday_age = (now - _parse_workday_age("Posted Yesterday")).total_seconds()
        assert abs(today_age) < 5
        assert abs(yesterday_age - 86400) < 5

    def test_workday_day_count(self):
        parsed = _parse_workday_age("Posted 8 Days Ago")
        assert 7 <= (datetime.now(timezone.utc) - parsed).days <= 8

    def test_workday_plus_form(self):
        assert _parse_workday_age("Posted 30+ Days Ago") is not None

    def test_workday_silence_is_not_a_date(self):
        assert _parse_workday_age("") is None
        assert _parse_workday_age("Posted") is None


class TestPayloadReading:
    """Each ATS reports the same facts under different names."""

    def test_greenhouse_reads_location_and_date(self, stub_request):
        stub_request({"jobs": [{
            "title": "Data Analyst",
            "company_name": "Acme",
            "location": {"name": "Houston, TX"},
            "absolute_url": "https://example.test/j/1",
            "first_published": "2026-08-19T14:02:07-04:00",
            "content": "&lt;p&gt;Build &amp; own dashboards&lt;/p&gt;",
        }]})
        [job] = read_greenhouse(Employer("Acme", "greenhouse", "acme"))
        assert job.location == "Houston, TX"
        assert job.posted_at.year == 2026
        assert "Build & own dashboards" in job.description

    def test_lever_collects_every_location(self, stub_request):
        stub_request([{
            "text": "Data Analyst",
            "categories": {
                "location": "Houston",
                "allLocations": ["Houston", "Austin"],
                "commitment": "Full-time: Remote",
            },
            "hostedUrl": "https://example.test/j/2",
            "createdAt": 1701703519526,
            "descriptionPlain": "Own reporting.",
        }])
        [job] = read_lever(Employer("Acme", "lever", "acme"))
        assert "Houston" in job.location and "Austin" in job.location
        assert job.work_mode == "remote"

    def test_workday_builds_an_absolute_url(self, stub_request):
        stub_request({"jobPostings": [{
            "title": "Land Analyst",
            "locationsText": "Houston, Texas, United States",
            "postedOn": "Posted Today",
            "externalPath": "/job/Houston/Land-Analyst_R123",
        }]})
        [job] = read_workday(Employer("Chevron", "workday", "chevron", site="jobs"))
        assert job.url.startswith("https://chevron.wd5.myworkdayjobs.com/jobs/job/")
        assert job.posted_text == "Posted Today"

    def test_an_unreachable_board_yields_nothing(self, stub_request):
        stub_request(None)
        assert read_greenhouse(Employer("Acme", "greenhouse", "acme")) == []
        assert read_lever(Employer("Acme", "lever", "acme")) == []

    def test_a_reader_that_raises_does_not_stop_the_run(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("board on fire")

        monkeypatch.setitem(ats_boards.READERS, "greenhouse", boom)
        assert fetch_board(Employer("Acme", "greenhouse", "acme")) == []


class TestFiltering:
    """Role, place, and age are matched against what the board actually said."""

    def test_location_must_be_stated_to_match(self):
        assert matches_location(posting(), HOUSTON)
        assert not matches_location(posting(location=""), HOUSTON)

    def test_a_different_city_does_not_match(self):
        assert not matches_location(posting(location="Austin, Texas"), HOUSTON)

    def test_search_filters_by_role_and_place(self):
        jobs = [
            posting(title="Data Analyst"),
            posting(title="Chef", url="https://example.test/2"),
            posting(title="Data Analyst", location="Austin, TX", url="https://example.test/3"),
        ]
        found = ats_boards._filter(jobs, HOUSTON, ANALYST, None)
        assert [j.title for j in found] == ["Data Analyst"]

    def test_an_undated_posting_survives_an_age_limit(self):
        """A board saying nothing is not evidence the posting is old."""
        jobs = [posting(posted_at=None)]
        assert ats_boards._filter(jobs, HOUSTON, ANALYST, 7)

    def test_an_old_posting_is_dropped(self):
        old = posting(posted_at=datetime.now(timezone.utc) - timedelta(days=30))
        assert not ats_boards._filter([old], HOUSTON, ANALYST, 7)

    def test_freshest_first(self):
        older = posting(title="Data Analyst A", posted_at=datetime.now(timezone.utc) - timedelta(days=5))
        newer = posting(title="Data Analyst B", posted_at=datetime.now(timezone.utc))
        assert [j.title for j in ats_boards._filter([older, newer], HOUSTON, ANALYST, None)] == [
            "Data Analyst B", "Data Analyst A"
        ]


class TestGrouping:
    """Results group by city and by country, with unknowns kept visible."""

    def test_group_by_city(self):
        jobs = [
            posting(location="Houston, Texas, United States"),
            posting(location="Houston, TX", url="https://example.test/2"),
            posting(location="Austin, TX", url="https://example.test/3"),
        ]
        grouped = ats_boards.group_by_city(jobs)
        assert list(grouped) == ["Houston, TX", "Austin, TX"]
        assert len(grouped["Houston, TX"]) == 2

    def test_a_multi_city_posting_appears_under_each(self):
        jobs = [posting(location="Mountain View, California; San Francisco, California")]
        grouped = ats_boards.group_by_city(jobs)
        assert set(grouped) == {"Mountain View, CA", "San Francisco, CA"}

    def test_group_by_country_nests_cities(self):
        jobs = [
            posting(location="Houston, TX"),
            posting(location="Austin, TX", url="https://example.test/2"),
            posting(location="London, United Kingdom", url="https://example.test/3"),
        ]
        tree = ats_boards.group_by_country(jobs)
        assert list(tree) == ["United States", "United Kingdom"]
        assert set(tree["United States"]) == {"Houston, TX", "Austin, TX"}

    def test_unknown_places_sort_last_and_are_not_hidden(self):
        jobs = [
            posting(location="Hybrid"),
            posting(location="Houston, TX", url="https://example.test/2"),
        ]
        grouped = ats_boards.group_by_city(jobs)
        assert list(grouped)[-1] == "Location not stated"
        assert len(grouped["Location not stated"]) == 1

    def test_country_filter_keeps_only_the_named_countries(self):
        jobs = [
            posting(location="Houston, TX"),
            posting(location="London, United Kingdom", url="https://example.test/2"),
        ]
        kept = ats_boards._filter(jobs, [], ANALYST, None, ["United Kingdom"])
        assert [j.location for j in kept] == ["London, United Kingdom"]

    def test_a_posting_with_no_country_never_matches_a_country_filter(self):
        """An unstated country is not evidence the job is in the one asked for."""
        assert not ats_boards.matches_country(posting(location="Hybrid"), ["United States"])

    def test_an_empty_country_filter_matches_everything(self):
        assert ats_boards.matches_country(posting(location="Hybrid"), [])


class TestRegistry:
    """A malformed registry row is skipped, never fatal."""

    def test_valid_rows_load(self, tmp_path):
        path = tmp_path / "e.json"
        path.write_text(json.dumps([
            {"name": "Chevron", "ats": "workday", "slug": "chevron", "site": "jobs"},
            {"name": "Acme", "ats": "greenhouse", "slug": "acme"},
        ]))
        assert [e.name for e in load_employers(path)] == ["Chevron", "Acme"]

    def test_rows_missing_fields_or_using_an_unknown_ats_are_skipped(self, tmp_path):
        path = tmp_path / "e.json"
        path.write_text(json.dumps([
            {"name": "NoSlug", "ats": "greenhouse"},
            {"name": "Bad", "ats": "carrier-pigeon", "slug": "x"},
            {"ats": "lever", "slug": "noname"},
            "not-a-dict",
            {"name": "Good", "ats": "lever", "slug": "good"},
        ]))
        assert [e.name for e in load_employers(path)] == ["Good"]

    def test_a_missing_registry_is_not_fatal(self, tmp_path):
        assert load_employers(tmp_path / "absent.json") == []

    def test_a_corrupt_registry_is_not_fatal(self, tmp_path):
        path = tmp_path / "e.json"
        path.write_text("{not json")
        assert load_employers(path) == []


class TestHtmlAndWorkMode:
    def test_entities_and_tags_are_reduced_to_text(self):
        assert _strip_html("&lt;p&gt;A &amp; B&lt;/p&gt;") == "A & B"

    @pytest.mark.parametrize(
        "commitment,expected",
        [("Full-time: Remote", "remote"), ("Hybrid", "hybrid"),
         ("On-site", "onsite"), ("Full-time", "unknown")],
    )
    def test_work_mode_is_read_not_guessed(self, commitment, expected):
        assert _lever_work_mode(commitment, "") == expected
