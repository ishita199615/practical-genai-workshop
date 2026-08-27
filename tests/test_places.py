"""Reading city, region, and country out of the strings job boards publish.

The cases here are real location strings taken from live ATS boards, including
the ones that name no place at all.
"""

from __future__ import annotations

import pytest

from tools.places import Place, parse_place, parse_places


class TestStructuredLocations:
    """City, region, and country are read from where the board put them."""

    @pytest.mark.parametrize(
        "raw,city,region,country",
        [
            ("Houston, Texas, United States", "Houston", "TX", "United States"),
            ("San Francisco, California", "San Francisco", "CA", "United States"),
            ("Austin, TX", "Austin", "TX", "United States"),
            ("Seattle, Washington", "Seattle", "WA", "United States"),
            ("London, United Kingdom", "London", "", "United Kingdom"),
            ("Toronto, Canada", "Toronto", "", "Canada"),
            ("Bengaluru, India", "Bengaluru", "", "India"),
        ],
    )
    def test_common_shapes(self, raw, city, region, country):
        place = parse_place(raw)
        assert (place.city, place.region, place.country) == (city, region, country)


class TestAmbiguousStateNames:
    """Several US states share a name with a city; position decides which."""

    def test_a_repeated_name_is_city_then_state(self):
        place = parse_place("New York, New York")
        assert (place.city, place.region) == ("New York", "NY")

    def test_a_two_letter_code_is_always_the_state(self):
        place = parse_place("New York, NY")
        assert (place.city, place.region) == ("New York", "NY")

    def test_a_state_name_followed_by_a_city_is_another_city(self):
        """"Chicago, New York, San Francisco" lists three cities, not a state."""
        places = parse_places("Chicago, New York, San Francisco")
        assert [p.city for p in places] == ["Chicago", "New York", "San Francisco"]
        # Each well-known city also implies its own state, not a shared one.
        assert [p.region for p in places] == ["IL", "NY", "CA"]

    def test_washington_dc_keeps_its_region(self):
        place = parse_place("Washington, DC")
        assert (place.city, place.region) == ("Washington", "DC")


class TestNonPlaces:
    """A work mode in the location field is not a location."""

    @pytest.mark.parametrize("raw", ["Hybrid", "In-Office", "Distributed", "N/A", ""])
    def test_these_name_no_place(self, raw):
        place = parse_place(raw)
        assert not place.is_known
        assert place.city_label == "Location not stated"
        assert place.country_label == "Country not stated"

    def test_bare_remote_is_remote_but_placeless(self):
        place = parse_place("Remote")
        assert place.is_remote and not place.is_known

    def test_remote_with_a_place_keeps_the_place(self):
        place = parse_place("Remote - Austin")
        assert place.is_remote
        assert place.city == "Austin" and place.country == "United States"


class TestAwkwardFormats:
    """Boards emit office codes, address prefixes, and region hierarchies."""

    def test_an_office_code_suffix_is_dropped(self):
        assert parse_place("San Francisco - SF9").city == "San Francisco"

    def test_a_country_code_prefixed_address(self):
        place = parse_place("SG-SINGAPORE-29 AYER MERBAU ROAD")
        assert place.country == "Singapore"

    def test_a_region_hierarchy_keeps_the_most_specific_part(self):
        place = parse_place("Asia / Hong Kong / Taiwan, Taipei")
        assert place.city == "Taipei" and place.country == "Taiwan"

    def test_a_country_written_before_its_city(self):
        place = parse_place("Taiwan, Taipei")
        assert place.city == "Taipei" and place.country == "Taiwan"

    def test_a_macro_region_is_not_a_city(self):
        place = parse_place("Asia")
        assert place.city == "" and place.region == "Asia"
        assert place.country_label == "Country not stated"

    def test_a_city_state_is_both(self):
        for raw in ("Singapore", "Hong Kong"):
            place = parse_place(raw)
            assert place.city == raw and place.country == raw


class TestCodePrefixedLists:
    """Some boards list places as "US-SF, US-NYC or US-Remote"."""

    def test_a_list_becomes_one_place_each(self):
        labels = [
            p.city_label
            for p in parse_places("US-SF, US-Seattle, US-NYC, US-Chicago")
        ]
        assert labels == ["San Francisco", "Seattle", "New York", "Chicago"]

    def test_a_remote_item_names_no_city(self):
        [place] = [p for p in parse_places("US-SF, US-Remote")][1:]
        assert place.is_remote and not place.city

    def test_a_state_only_name_is_a_region_not_a_city(self):
        [place] = [p for p in parse_places("US-SF, US-Georgia")][1:]
        assert place.city == "" and place.region == "GA"

    def test_a_name_that_is_both_city_and_state_reads_as_the_city(self):
        [place] = parse_places("US-NYC")
        assert place.city == "New York"

    @pytest.mark.parametrize(
        "raw,city",
        [("MY-KUL-KUALA LUMPUR", "Kuala Lumpur"),
         ("GB-SO-NAILSEA-2 HIGH STREET", "Nailsea"),
         ("MX-DF-MEXICO CITY-AVENIDA", "Mexico City")],
    )
    def test_addresses_yield_the_city_not_a_code(self, raw, city):
        assert parse_place(raw).city == city


class TestMultipleLocations:
    """One posting can be open in several places."""

    def test_semicolons_separate_places(self):
        labels = [
            p.city_label
            for p in parse_places("Mountain View, California; San Francisco, California")
        ]
        assert labels == ["Mountain View, CA", "San Francisco, CA"]

    def test_duplicates_collapse(self):
        assert len(parse_places("Austin, TX; Austin, TX")) == 1

    def test_a_single_place_still_returns_one(self):
        assert len(parse_places("Houston, Texas, United States")) == 1

    def test_an_empty_field_returns_one_unknown_place(self):
        [place] = parse_places("")
        assert not place.is_known


class TestCountryInference:
    """A country is filled in only where the place already implies one."""

    def test_a_us_state_implies_the_country(self):
        assert parse_place("Austin, TX").country == "United States"

    def test_a_well_known_city_implies_its_country(self):
        assert parse_place("Dublin").country == "Ireland"

    def test_an_unknown_bare_city_gets_no_country(self):
        place = parse_place("Zzyzx")
        assert place.city == "Zzyzx"
        assert place.country == ""
        assert place.country_label == "Country not stated"

    def test_the_raw_string_is_always_kept(self):
        assert parse_place("Hybrid").raw == "Hybrid"


class TestPlaceLabels:
    def test_city_and_region_combine(self):
        assert Place(city="Austin", region="TX").city_label == "Austin, TX"

    def test_country_only_says_so(self):
        assert Place(country="Canada").city_label == "Canada — city not stated"
