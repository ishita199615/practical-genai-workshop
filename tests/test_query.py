"""The one search both routes of the agent answer."""

from __future__ import annotations

import pytest

from models.query import SearchQuery


class TestDerivedViews:
    """Each route reads the fields it can honour."""

    def test_the_city_drops_the_state(self):
        assert SearchQuery(location="Houston, TX").city == "Houston"

    def test_a_bare_city_is_kept(self):
        assert SearchQuery(location="Houston").city == "Houston"

    def test_an_empty_location_has_no_city(self):
        assert SearchQuery(location="").city == ""

    @pytest.mark.parametrize(
        "window,days",
        [("last_hour", 1.0), ("last_24_hours", 1.0), ("last_3_days", 3.0),
         ("last_7_days", 7.0)],
    )
    def test_freshness_becomes_days(self, window, days):
        assert SearchQuery(freshness_window=window).max_age_days == days

    def test_an_unknown_window_has_no_age_limit(self):
        assert SearchQuery(freshness_window="whenever").max_age_days is None


class TestRolePattern:
    """A role is matched word by word, so near titles are not missed."""

    def test_every_word_is_an_alternative(self):
        pattern = SearchQuery(role="Data Analyst Intern").role_pattern()
        assert pattern.search("Data Analyst")
        assert pattern.search("Analyst Intern")
        assert pattern.search("Senior Data Scientist")

    def test_an_unrelated_title_does_not_match(self):
        pattern = SearchQuery(role="Data Analyst").role_pattern()
        assert not pattern.search("Head Chef")

    def test_filler_words_are_not_alternatives(self):
        """"of" and "the" would match nearly every title on their own."""
        pattern = SearchQuery(role="Head of the Data Team").role_pattern()
        alternatives = set(pattern.pattern.split("|"))
        assert "of" not in alternatives and "the" not in alternatives
        assert {"Head", "Data", "Team"} <= alternatives

    def test_a_role_of_only_filler_matches_everything(self):
        assert SearchQuery(role="of the").role_pattern().search("Anything")

    def test_an_empty_role_matches_everything(self):
        assert SearchQuery(role="").role_pattern().search("Anything")


class TestWorkMode:
    @pytest.mark.parametrize("mode,wanted", [
        ("any", True), ("remote", True), ("On-site", False), ("hybrid", False),
    ])
    def test_remote_is_wanted_for_any_and_remote(self, mode, wanted):
        assert SearchQuery(work_mode=mode).wants_remote() is wanted


class TestRoundTrip:
    """The query crosses a session boundary, so reading it back is defensive."""

    def test_a_query_survives_a_round_trip(self):
        original = SearchQuery(
            role="Data Analyst", location="Austin, TX", work_mode="onsite",
            query_category="all", freshness_window="last_7_days",
            experience_level="entry",
        )
        assert SearchQuery.from_dict(original.to_dict()) == original

    @pytest.mark.parametrize("junk", [None, "not a dict", 42, []])
    def test_junk_falls_back_to_defaults(self, junk):
        assert SearchQuery.from_dict(junk) == SearchQuery()

    def test_missing_fields_keep_their_defaults(self):
        query = SearchQuery.from_dict({"role": "Chef"})
        assert query.role == "Chef"
        assert query.location == SearchQuery().location

    def test_blank_and_wrongly_typed_values_are_ignored(self):
        query = SearchQuery.from_dict({"role": "  ", "location": {"a": 1}})
        assert query.role == SearchQuery().role
        assert query.location == SearchQuery().location
