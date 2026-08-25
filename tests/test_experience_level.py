"""Tests for seniority detection and level-aware query building.

The feature exists because the app used to return every seniority at once. Two
properties matter most and are pinned hard here:

* a posting that never states a level stays ``unknown`` and is never assigned
  the level the user happened to search for, and
* substring accidents ("Internal", "Seniority", "Leading") never register as a
  level.
"""

from __future__ import annotations

import pytest

from models.job import EXPERIENCE_LEVEL_LABELS, EXPERIENCE_LEVEL_YEARS
from tools.experience_level import (
    LEVEL_ORDER,
    detect_experience_level,
    detect_from_body,
    detect_from_title,
    detect_from_years,
    expected_years,
    level_query_terms,
    levels_conflict,
    strip_level_words,
)

REAL_LEVELS = ("internship", "entry", "mid", "senior", "staff_principal", "manager")


class TestStripLevelWords:
    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            ("Senior Data Analyst", "data analyst"),
            ("Data Analyst Intern", "data analyst"),
            ("Principal Data Scientist", "data scientist"),
            ("Junior Data Analyst", "data analyst"),
            ("Data Scientist", "data scientist"),
            ("Entry Level Data Analyst", "data analyst"),
        ],
    )
    def test_removes_seniority_words(self, role: str, expected: str) -> None:
        assert strip_level_words(role) == expected

    def test_empty_role_is_empty(self) -> None:
        assert strip_level_words("") == ""
        assert strip_level_words("   ") == ""

    def test_a_role_that_is_only_a_level_word_collapses(self) -> None:
        assert strip_level_words("Intern") == ""


class TestLevelQueryTerms:
    @pytest.mark.parametrize("level", REAL_LEVELS)
    def test_never_contradicts_a_level_already_in_the_role(self, level: str) -> None:
        # "Data Analyst Intern" + senior must not yield "senior data analyst intern".
        terms = level_query_terms("Data Analyst Intern", level)
        assert terms
        if level != "internship":
            assert not any("intern" in term for term in terms)

    def test_senior_terms(self) -> None:
        assert level_query_terms("Data Analyst", "senior") == [
            "senior data analyst",
            "sr data analyst",
        ]

    def test_internship_terms(self) -> None:
        assert level_query_terms("Data Analyst", "internship") == [
            "data analyst intern",
            "data analyst internship",
        ]

    def test_entry_terms_cover_common_spellings(self) -> None:
        terms = level_query_terms("Data Analyst", "entry")
        assert "entry level data analyst" in terms
        assert "junior data analyst" in terms

    def test_staff_principal_covers_lead(self) -> None:
        terms = level_query_terms("Data Scientist", "staff_principal")
        assert "principal data scientist" in terms
        assert "lead data scientist" in terms

    def test_unknown_level_returns_the_bare_role(self) -> None:
        assert level_query_terms("Data Analyst", "unknown") == ["data analyst"]

    def test_empty_role_yields_no_terms(self) -> None:
        assert level_query_terms("", "senior") == []
        assert level_query_terms("   ", "internship") == []


class TestARoleThatIsOnlyASeniorityWord:
    """Typing just "Intern" or "Manager" must still search for something.

    Stripping the seniority out of such a role leaves an empty base. Returning
    no terms there dropped the role clause from the query altogether, so the
    search became "(Houston OR remote)" — every page in a city, no job filter
    at all. The level's own terms stand in instead.
    """

    @pytest.mark.parametrize("role", ["Intern", "Manager", "Senior", "Associate"])
    @pytest.mark.parametrize("level", REAL_LEVELS)
    def test_terms_are_never_empty(self, role: str, level: str) -> None:
        terms = level_query_terms(role, level)
        assert terms and all(term.strip() for term in terms)

    def test_the_terms_follow_the_selected_level_not_the_typed_word(self) -> None:
        # "Intern" typed with Senior selected must not search for internships.
        terms = level_query_terms("Intern", "senior")
        assert terms == ["senior", "sr"]
        assert not any("intern" in term for term in terms)

    def test_intern_with_internship_searches_internships(self) -> None:
        assert level_query_terms("Intern", "internship") == ["intern", "internship"]

    def test_the_query_still_carries_a_role_clause(self) -> None:
        from tools.firecrawl_search import build_search_query

        query = build_search_query("Manager", "Houston, TX", "Any", "all", "manager")
        assert '"manager"' in query
        # The regression: the whole role clause used to vanish.
        assert query != "(Houston OR remote)"

    @pytest.mark.parametrize("level", REAL_LEVELS)
    def test_every_level_produces_usable_terms(self, level: str) -> None:
        terms = level_query_terms("Data Scientist", level)
        assert terms and all(term.strip() for term in terms)


class TestDetectFromTitle:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Senior Data Scientist", "senior"),
            ("Sr. Data Analyst", "senior"),
            ("Data Analyst Intern", "internship"),
            ("Summer Internship - Analytics", "internship"),
            ("Junior Data Analyst", "entry"),
            ("Associate Data Scientist", "entry"),
            ("Principal Machine Learning Engineer", "staff_principal"),
            ("Staff Data Scientist", "staff_principal"),
            ("Lead Data Engineer", "staff_principal"),
            ("Data Science Manager", "manager"),
            ("Director of Analytics", "manager"),
            ("Data Analyst II", "mid"),
            ("Data Analyst I", "entry"),
        ],
    )
    def test_detects_stated_levels(self, title: str, expected: str) -> None:
        assert detect_from_title(title).level == expected

    @pytest.mark.parametrize(
        "title",
        [
            "Internal Audit Analyst",
            "Seniority Review Coordinator",
            "Leading Indicators Analyst",
            "Data Scientist",
            "Business Intelligence Analyst",
            "Marketing Interngration Specialist",
        ],
    )
    def test_substrings_do_not_register_as_levels(self, title: str) -> None:
        # "Internal" is not "intern"; "Seniority" is not "senior".
        assert detect_from_title(title).level == "unknown"

    def test_higher_level_wins_in_a_compound_title(self) -> None:
        assert detect_from_title("Senior Staff Engineer").level == "staff_principal"

    @pytest.mark.parametrize(
        "title",
        [
            "Graduate Intern, Data Analytics",
            "Summer 2026 Data Analyst Intern - Campus Program",
            "New Grad Data Analyst Internship",
            "Associate Analyst Intern",
            "Junior Data Analyst Co-op",
        ],
    )
    def test_an_internship_word_outranks_a_soft_entry_word(self, title: str) -> None:
        # "graduate", "campus", and "associate" sit on internship titles all the
        # time. Reading entry first labelled these as junior roles, so a user
        # asking for Internship had real internships filtered off the screen.
        assert detect_from_title(title).level == "internship"

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Entry Level Data Analyst", "entry"),
            ("Junior Data Analyst", "entry"),
            ("Associate Data Analyst", "entry"),
            ("New Grad Data Analyst", "entry"),
            ("Senior Data Analyst Intern", "senior"),
            ("Analytics Manager, Campus Recruiting", "manager"),
        ],
    )
    def test_entry_and_senior_titles_are_unaffected(
        self, title: str, expected: str
    ) -> None:
        assert detect_from_title(title).level == expected

    def test_associate_director_is_management_not_entry(self) -> None:
        assert detect_from_title("Associate Director, Analytics").level == "manager"

    def test_evidence_names_the_marker(self) -> None:
        detection = detect_from_title("Senior Data Scientist")
        assert detection.evidence and "senior" in detection.evidence

    def test_empty_title_is_unknown(self) -> None:
        assert detect_from_title("").level == "unknown"
        assert detect_from_title("").is_known is False


class TestDetectFromYears:
    @pytest.mark.parametrize(
        ("years", "expected"),
        [
            (0.0, "entry"),
            (1.0, "entry"),
            (3.0, "mid"),
            (6.0, "senior"),
            (10.0, "staff_principal"),
        ],
    )
    def test_maps_years_to_levels(self, years: float, expected: str) -> None:
        assert detect_from_years(years).level == expected

    def test_missing_years_is_unknown(self) -> None:
        assert detect_from_years(None).level == "unknown"

    def test_evidence_quotes_the_requirement(self) -> None:
        detection = detect_from_years(6.0)
        assert detection.evidence and "6" in detection.evidence


class TestDetectExperienceLevel:
    def test_title_wins_over_years(self) -> None:
        # The title is where employers state seniority.
        detection = detect_experience_level("Data Analyst Intern", "", 6.0)
        assert detection.level == "internship"

    def test_years_used_when_the_title_is_silent(self) -> None:
        detection = detect_experience_level("Data Scientist", "", 6.0)
        assert detection.level == "senior"
        assert detection.evidence and "years" in detection.evidence

    def test_body_used_when_title_and_years_are_silent(self) -> None:
        detection = detect_experience_level(
            "Data Scientist", "This is an entry level position for new graduates."
        )
        assert detection.level == "entry"

    def test_nothing_stated_stays_unknown(self) -> None:
        detection = detect_experience_level(
            "Data Scientist", "We build dashboards for retail partners."
        )
        assert detection.level == "unknown"
        assert detection.evidence is None
        assert detection.is_known is False

    def test_detection_is_deterministic(self) -> None:
        first = detect_experience_level("Senior Data Scientist", "some text", 6.0)
        second = detect_experience_level("Senior Data Scientist", "some text", 6.0)
        assert (first.level, first.evidence) == (second.level, second.evidence)


class TestLevelsConflict:
    def test_same_level_never_conflicts(self) -> None:
        assert levels_conflict("senior", "senior") is False

    def test_different_known_levels_conflict(self) -> None:
        assert levels_conflict("senior", "internship") is True

    @pytest.mark.parametrize("requested", REAL_LEVELS)
    def test_unknown_detection_never_conflicts(self, requested: str) -> None:
        # No evidence is not grounds to drop a real opening.
        assert levels_conflict(requested, "unknown") is False

    @pytest.mark.parametrize("detected", REAL_LEVELS)
    def test_unknown_request_filters_nothing(self, detected: str) -> None:
        assert levels_conflict("unknown", detected) is False


class TestLevelMetadata:
    def test_every_level_has_a_label(self) -> None:
        for level in (*REAL_LEVELS, "unknown"):
            assert EXPERIENCE_LEVEL_LABELS[level].strip()

    def test_every_real_level_has_expected_years(self) -> None:
        for level in REAL_LEVELS:
            assert level in EXPERIENCE_LEVEL_YEARS
            low, high = expected_years(level)
            assert low >= 0
            assert high is None or high > low

    def test_level_order_is_most_to_least_senior(self) -> None:
        assert LEVEL_ORDER[0] == "manager"
        assert LEVEL_ORDER[-1] == "internship"
        assert set(LEVEL_ORDER) == set(REAL_LEVELS)

    def test_expected_years_increase_with_seniority(self) -> None:
        ladder = ["internship", "entry", "mid", "senior", "staff_principal"]
        minimums = [expected_years(level)[0] for level in ladder]
        assert minimums == sorted(minimums)


class TestLookalikesAreNotLevels:
    """A marker's letters appearing somewhere is not a statement of seniority.

    Every case here was a real false positive: each one labelled a posting with
    a level it never stated, which then filtered it off the screen for every
    other level the user might ask for.
    """

    @pytest.mark.parametrize(
        "title",
        [
            "Principal's Office Coordinator",  # possessive noun, not a level
            "Principal’s Office Coordinator",  # the curly apostrophe spelling
            "Head Office Analyst",  # "head of" straddles two words
            "Bulkhead Office Analyst",
            "Overhead Offset Analyst",
            "New Gradient Systems Engineer",  # "new grad" straddles two words
            "Entry Levelling Technician",
            "Lead Generation Specialist",  # a marketing role, not a lead engineer
            "Lead Gen Analyst",
            "Grade 2 Technician",  # a pay grade, not a level II role
            "Analyst, Tier 3 Support",  # a support tier, not a level III role
        ],
    )
    def test_lookalike_titles_state_no_level(self, title: str) -> None:
        assert detect_from_title(title).level == "unknown"

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Head of Data", "manager"),
            ("New Grad Data Analyst", "entry"),
            ("Entry Level Data Analyst", "entry"),
            ("Lead Data Engineer", "staff_principal"),
            ("Data Analyst II", "mid"),
            ("Data Analyst III", "senior"),
            ("Principal Data Scientist", "staff_principal"),
        ],
    )
    def test_the_real_phrasings_still_register(self, title: str, expected: str) -> None:
        assert detect_from_title(title).level == expected


class TestBodyNegation:
    """A negated phrase states the opposite of the level it names."""

    @pytest.mark.parametrize(
        "description",
        [
            "This role has no direct reports and reports to the analytics manager.",
            "People management experience is not required for this position.",
            "The position carries no people management responsibility.",
        ],
    )
    def test_a_negated_phrase_states_no_level(self, description: str) -> None:
        assert detect_from_body(description).level == "unknown"
        assert detect_experience_level("Data Analyst", description).level == "unknown"

    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("You will manage a team of four analysts.", "manager"),
            ("This is an entry level position for recent graduates.", "entry"),
            ("As an intern you will shadow the analytics team.", "internship"),
            ("This is a senior-level role on the analytics team.", "senior"),
        ],
    )
    def test_a_plain_statement_still_registers(
        self, description: str, expected: str
    ) -> None:
        assert detect_from_body(description).level == expected


class TestStrippingKeepsTheRole:
    """Stripping seniority must not damage the role the user typed.

    A raw string replace once turned "Head Office Analyst" into "fice analyst",
    and that phrase went straight into the live search query.
    """

    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            ("Head Office Analyst", "head office analyst"),
            ("Bulkhead Office Analyst", "bulkhead office analyst"),
            ("Entry Levelling Technician", "entry levelling technician"),
            ("Data Architect", "data architect"),
            ("Lead Generation Specialist", "lead generation specialist"),
            ("Tier 3 Support Analyst", "tier 3 support analyst"),
        ],
    )
    def test_the_typed_role_survives(self, role: str, expected: str) -> None:
        assert strip_level_words(role) == expected

    @pytest.mark.parametrize("level", REAL_LEVELS)
    def test_query_terms_keep_the_typed_role(self, level: str) -> None:
        terms = level_query_terms("Head Office Analyst", level)
        assert terms
        assert all("head office analyst" in term for term in terms)

    def test_a_role_of_only_level_words_has_no_base(self) -> None:
        # "Chief of Staff" strips to a bare connector, which is no role at all.
        assert strip_level_words("Chief of Staff") == ""
        assert level_query_terms("Chief of Staff", "senior") == ["senior", "sr"]


class TestArchitectNamesTheRoleNotTheLevel:
    """"Architect" is a role noun: there are junior, senior, and principal ones."""

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Data Architect", "unknown"),
            ("Enterprise Architect", "unknown"),
            ("Senior Data Architect", "senior"),
            ("Junior Solutions Architect", "entry"),
            ("Principal Architect", "staff_principal"),
        ],
    )
    def test_only_a_stated_level_registers(self, title: str, expected: str) -> None:
        assert detect_from_title(title).level == expected

    def test_the_role_survives_a_level_search(self) -> None:
        assert level_query_terms("Data Architect", "senior") == [
            "senior data architect",
            "sr data architect",
        ]


class TestQueryTermsAgreeWithDetection:
    """A searched phrase must read back as the level that produced it.

    Otherwise the agent searches for phrasings its own filter then rejects, and
    a legitimate search returns nothing.
    """

    @pytest.mark.parametrize("level", REAL_LEVELS)
    @pytest.mark.parametrize(
        "role",
        [
            "Data Analyst",
            "Data Analyst Intern",
            "Senior Data Analyst",
            "Data Architect",
            "Head Office Analyst",
            "Lead Generation Specialist",
            "Business Intelligence Analyst",
            "Tier 3 Support Analyst",
            "Intern",
        ],
    )
    def test_no_produced_phrase_states_another_level(
        self, role: str, level: str
    ) -> None:
        for term in level_query_terms(role, level):
            detected = detect_from_title(term).level
            assert detected in (level, "unknown"), (term, detected)
