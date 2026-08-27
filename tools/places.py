"""Read a city, region, and country out of the location strings boards publish.

There is no standard here. The same pool of postings carries all of these:

    Houston, Texas, United States      London, United Kingdom
    San Francisco, California          Bengaluru
    Austin, TX                         Remote - Canada
    Asia / Hong Kong / Taiwan, Taipei  Mountain View, CA; San Francisco, CA
    Hybrid                             N/A

The last two are the important ones: a sizeable share of postings put a *work
mode* in the location field, or nothing at all. Those carry no place, and this
module says so rather than inventing one — the same rule the rest of the project
follows for seniority and posting dates. A caller can then group the located
postings by city and country and show the rest as "Location not stated", which
is the honest answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNKNOWN = ""

# Strings that occupy the location field but name no place.
NON_PLACES = {
    "hybrid", "in office", "in-office", "onsite", "on-site", "on site",
    "remote", "distributed", "flexible", "anywhere", "various", "multiple",
    "n a", "n/a", "na", "tbd", "unknown", "-", "",
}

# A leading remote marker: "Remote - Austin", "Remote, United States".
_REMOTE_PREFIX_RE = re.compile(r"^\s*(?:fully\s+)?remote\s*[-–—,:]\s*", re.IGNORECASE)
_REMOTE_BARE_RE = re.compile(r"^\s*(?:fully\s+)?remote\s*$", re.IGNORECASE)
# Trailing office codes: "San Francisco - SF9", "London - UK2".
_OFFICE_CODE_RE = re.compile(r"\s*[-–—]\s*[A-Z]{2,4}\d{1,3}\s*$")
# Country-code prefixed addresses: "SG-SINGAPORE-29 AYER MERBAU ROAD", and
# "GB-SO-NAILSEA-2 HIGH STREET", where a short region code sits before the city.
_CODE_PREFIX_RE = re.compile(r"^([A-Z]{2})-(.+)$", re.ASCII)

COUNTRY_ALIASES = {
    "united states": "United States", "united states of america": "United States",
    "usa": "United States", "us": "United States", "u s": "United States",
    "united kingdom": "United Kingdom", "uk": "United Kingdom",
    "great britain": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom",
    "canada": "Canada", "india": "India", "singapore": "Singapore",
    "japan": "Japan", "australia": "Australia", "germany": "Germany",
    "france": "France", "netherlands": "Netherlands", "ireland": "Ireland",
    "spain": "Spain", "italy": "Italy", "poland": "Poland", "brazil": "Brazil",
    "mexico": "Mexico", "china": "China", "hong kong": "Hong Kong",
    "taiwan": "Taiwan", "south korea": "South Korea", "korea": "South Korea",
    "israel": "Israel", "sweden": "Sweden", "denmark": "Denmark",
    "norway": "Norway", "finland": "Finland", "switzerland": "Switzerland",
    "belgium": "Belgium", "portugal": "Portugal", "austria": "Austria",
    "new zealand": "New Zealand", "south africa": "South Africa",
    "argentina": "Argentina", "colombia": "Colombia", "chile": "Chile",
    "indonesia": "Indonesia", "malaysia": "Malaysia", "thailand": "Thailand",
    "vietnam": "Vietnam", "philippines": "Philippines", "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates", "saudi arabia": "Saudi Arabia",
    "turkey": "Turkey", "egypt": "Egypt", "nigeria": "Nigeria", "kenya": "Kenya",
    "romania": "Romania", "czech republic": "Czech Republic", "czechia": "Czech Republic",
    "hungary": "Hungary", "greece": "Greece", "ukraine": "Ukraine",
}

# Two-letter prefixes used by address-style location strings.
COUNTRY_CODES = {
    "US": "United States", "GB": "United Kingdom", "UK": "United Kingdom",
    "CA": "Canada", "IN": "India", "SG": "Singapore", "JP": "Japan",
    "AU": "Australia", "DE": "Germany", "FR": "France", "NL": "Netherlands",
    "IE": "Ireland", "ES": "Spain", "IT": "Italy", "PL": "Poland",
    "BR": "Brazil", "MX": "Mexico", "CN": "China", "HK": "Hong Kong",
    "TW": "Taiwan", "KR": "South Korea", "IL": "Israel", "SE": "Sweden",
    "CH": "Switzerland", "BE": "Belgium", "PT": "Portugal", "AT": "Austria",
    "NZ": "New Zealand", "ZA": "South Africa", "AE": "United Arab Emirates",
    "SA": "Saudi Arabia", "MY": "Malaysia", "TH": "Thailand", "PH": "Philippines",
}

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC",
}
US_STATE_CODES = {code for code in US_STATES.values()}

# Continent- and market-level labels. Real information, but not a city, and not
# specific enough to name a country.
MACRO_REGIONS = {
    "asia": "Asia", "apac": "APAC", "asia pacific": "APAC",
    "south east asia": "Asia", "southeast asia": "Asia", "sea": "Asia",
    "south asia": "Asia", "east asia": "Asia", "greater china": "Asia",
    "europe": "Europe", "emea": "EMEA", "western europe": "Europe",
    "eastern europe": "Europe", "nordics": "Europe", "benelux": "Europe",
    "dach": "Europe", "uk and ireland": "Europe", "uk i": "Europe",
    "americas": "Americas", "north america": "North America",
    "south america": "South America", "latam": "LATAM",
    "latin america": "LATAM", "africa": "Africa", "mena": "Middle East",
    "middle east": "Middle East", "oceania": "Oceania", "anz": "Oceania",
    "global": "Global", "worldwide": "Global", "international": "Global",
    "emea apac": "Global",
}

# Cities common enough in board data to name their country unambiguously.
# Deliberately short: a bare city that is not listed keeps an unknown country
# rather than being assigned one on a guess.
CITY_COUNTRY = {
    "san francisco": "United States", "new york": "United States",
    "new york city": "United States", "chicago": "United States",
    "seattle": "United States", "boston": "United States",
    "austin": "United States", "houston": "United States",
    "dallas": "United States", "denver": "United States",
    "atlanta": "United States", "los angeles": "United States",
    "san jose": "United States", "mountain view": "United States",
    "palo alto": "United States", "bellevue": "United States",
    "washington": "United States", "miami": "United States",
    "philadelphia": "United States", "phoenix": "United States",
    "london": "United Kingdom", "manchester": "United Kingdom",
    "edinburgh": "United Kingdom", "dublin": "Ireland",
    "toronto": "Canada", "vancouver": "Canada", "montreal": "Canada",
    "bengaluru": "India", "bangalore": "India", "mumbai": "India",
    "hyderabad": "India", "delhi": "India", "new delhi": "India",
    "pune": "India", "chennai": "India", "gurgaon": "India",
    "singapore": "Singapore", "tokyo": "Japan", "osaka": "Japan",
    "sydney": "Australia", "melbourne": "Australia",
    "berlin": "Germany", "munich": "Germany", "hamburg": "Germany",
    "paris": "France", "amsterdam": "Netherlands", "madrid": "Spain",
    "barcelona": "Spain", "milan": "Italy", "rome": "Italy",
    "warsaw": "Poland", "krakow": "Poland", "lisbon": "Portugal",
    "stockholm": "Sweden", "copenhagen": "Denmark", "oslo": "Norway",
    "helsinki": "Finland", "zurich": "Switzerland", "geneva": "Switzerland",
    "brussels": "Belgium", "vienna": "Austria", "sao paulo": "Brazil",
    "mexico city": "Mexico", "hong kong": "Hong Kong", "taipei": "Taiwan",
    "seoul": "South Korea", "tel aviv": "Israel", "shanghai": "China",
    "beijing": "China", "dubai": "United Arab Emirates",
    "bucharest": "Romania", "prague": "Czech Republic", "budapest": "Hungary",
}


@dataclass(frozen=True)
class Place:
    """A location as far as the board actually stated it."""

    city: str = UNKNOWN
    region: str = UNKNOWN
    country: str = UNKNOWN
    is_remote: bool = False
    raw: str = ""

    @property
    def is_known(self) -> bool:
        """True when any part of a real place was read."""
        return bool(self.city or self.region or self.country)

    @property
    def city_label(self) -> str:
        """How this place should be titled when grouping by city."""
        if self.city and self.region:
            return f"{self.city}, {self.region}"
        if self.city:
            return self.city
        if self.country:
            return f"{self.country} — city not stated"
        if self.region:
            return f"{self.region} — city not stated"
        return "Location not stated"

    @property
    def country_label(self) -> str:
        """How this place should be titled when grouping by country."""
        return self.country or "Country not stated"


# Abbreviations boards use in place of a city name.
CITY_ABBREVIATIONS = {
    "sf": "San Francisco", "nyc": "New York", "la": "Los Angeles",
    "dc": "Washington", "sea": "Seattle", "chi": "Chicago",
    "kul": "Kuala Lumpur", "tor": "Toronto", "ldn": "London",
    "yyz": "Toronto", "sgp": "Singapore", "hkg": "Hong Kong",
}


def _expand_abbreviation(city: str) -> str:
    """Turn a board's shorthand into the city it stands for."""
    return CITY_ABBREVIATIONS.get(_normalize(city), city)


def _city_from_address(remainder: str) -> str:
    """Pick the city out of a hyphenated address tail.

    ``"SO-NAILSEA-2 HIGH STREET"`` puts a two-letter region code before the
    city, and a street address after it. The city is the first segment that is
    neither a short code nor the start of a street number — but when every
    segment is short, the first one is all the board gave.
    """
    segments = [segment.strip() for segment in remainder.split("-")]
    usable = [seg for seg in segments if seg and not seg[:1].isdigit()]
    for candidate in usable:
        if len(candidate) > 2:
            return _expand_abbreviation(candidate.title())
    return _expand_abbreviation(usable[0].title()) if usable else UNKNOWN


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").strip().lower()).strip()


def _classify(token: str) -> tuple[str, str]:
    """Return ``(kind, value)`` for one comma-separated token."""
    key = _normalize(token)
    if not key:
        return ("", "")
    if key in COUNTRY_ALIASES:
        return ("country", COUNTRY_ALIASES[key])
    if key in MACRO_REGIONS:
        return ("region", MACRO_REGIONS[key])
    if key in US_STATES:
        return ("region", US_STATES[key])
    upper = token.strip().upper()
    if upper in US_STATE_CODES and len(upper) == 2:
        return ("region", upper)
    return ("city", " ".join(word for word in token.strip().split()))


def parse_place(location: str) -> Place:
    """Read one location string into a :class:`Place`.

    A string that names no place — "Hybrid", "N/A", a bare "Remote" — yields a
    Place with nothing set but ``is_remote`` and ``raw``, so a caller can tell
    "we do not know" apart from "somewhere specific".
    """
    raw = (location or "").strip()
    if not raw:
        return Place(raw=raw)

    is_remote = bool(_REMOTE_PREFIX_RE.match(raw) or _REMOTE_BARE_RE.match(raw))
    text = _REMOTE_PREFIX_RE.sub("", raw).strip()
    if _REMOTE_BARE_RE.match(text) or _normalize(text) in NON_PLACES:
        return Place(is_remote=is_remote or _normalize(raw) == "remote", raw=raw)

    # "SG-SINGAPORE-29 AYER MERBAU ROAD"
    code_match = _CODE_PREFIX_RE.match(text)
    if code_match and code_match.group(1) in COUNTRY_CODES:
        country = COUNTRY_CODES[code_match.group(1)]
        name = _city_from_address(code_match.group(2))
        # What follows the country code is usually a city, but "US-Remote" is a
        # work mode and "US-Georgia" is a state. Classify it rather than assume.
        kind, value = _classify(name)
        if _normalize(name) in NON_PLACES:
            return Place(country=country, is_remote=True, raw=raw)
        # This slot holds a city, so a name that is both a city and a state
        # ("New York") is read as the city; only a state-only name is a region.
        if kind == "region" and _normalize(name) not in CITY_COUNTRY:
            return Place(region=value, country=country, is_remote=is_remote, raw=raw)
        return Place(city=name, country=country, is_remote=is_remote, raw=raw)

    text = _OFFICE_CODE_RE.sub("", text).strip()
    # "Asia / Hong Kong / Taiwan, Taipei" — the most specific part is last.
    if "/" in text:
        text = text.rsplit("/", 1)[-1].strip()

    return _parse_tokens(text, is_remote=is_remote, raw=raw)[0]


def _parse_tokens(text: str, *, is_remote: bool, raw: str) -> list[Place]:
    """Read comma-separated tokens into one or more places.

    Position decides, because several US states share a name with a city. In
    ``"New York, New York"`` the second token is the state; in ``"Chicago, New
    York, San Francisco"`` it is another city, because a city follows it. A
    state name is only read as a region when nothing but a country comes after.
    """
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        return [Place(is_remote=is_remote, raw=raw)]

    head_kind, head_value = _classify(tokens[0])
    if len(tokens) == 1:
        single = Place(
            city=head_value if head_kind == "city" else UNKNOWN,
            region=head_value if head_kind == "region" else UNKNOWN,
            country=head_value if head_kind == "country" else UNKNOWN,
            is_remote=is_remote,
            raw=raw,
        )
        return [_infer_country(single)]

    # The leading token is the most specific place named. Several US states
    # share a name with a city ("New York, New York"), so at the head the city
    # reading wins; only an outright country or macro-region overrides it.
    city = region = country = UNKNOWN
    if head_kind == "country":
        country = head_value
    elif head_kind == "region" and head_value in MACRO_REGIONS.values():
        region = head_value
    else:
        city = " ".join(tokens[0].split())

    # "Bengaluru, Karnataka, India" — when the last token is a country, whatever
    # sits between it and the city is that country's own region. No gazetteer of
    # the world's provinces is needed to know that from the shape alone.
    tail_is_country = len(tokens) > 2 and _classify(tokens[-1])[0] == "country"

    extra_cities: list[str] = []
    for index, token in enumerate(tokens[1:], start=1):
        kind, value = _classify(token)
        rest = tokens[index + 1 :]
        if kind == "country":
            country = country or value
        elif tail_is_country and kind == "city" and rest:
            region = region or " ".join(token.split())
        elif kind == "region":
            # A state name mid-list is a region only when it was written as a
            # two-letter code, or when nothing but a country follows it.
            # Otherwise the list is naming more cities.
            is_code = len(token.strip()) == 2
            trailing_is_country_only = all(
                _classify(later)[0] == "country" for later in rest
            )
            if not region and (is_code or trailing_is_country_only):
                region = value
            elif not city:
                city = " ".join(token.split())
            else:
                extra_cities.append(" ".join(token.split()))
        elif kind == "city":
            if not city:
                city = value
            else:
                extra_cities.append(value)

    primary = _infer_country(
        Place(city=city, region=region, country=country, is_remote=is_remote, raw=raw)
    )
    places = [primary]
    for name in extra_cities:
        places.append(
            _infer_country(
                Place(city=name, country=country, is_remote=is_remote, raw=raw)
            )
        )
    return places


# Places where the country and the city are the same place, so a bare country
# name is also a complete city name.
CITY_STATES = {"Singapore", "Hong Kong", "Macau", "Monaco", "Luxembourg"}

# The state a bare US city name implies, for the cities boards name most often.
# Only unambiguous ones: "Portland" and "Springfield" are deliberately absent.
CITY_REGION = {
    "san francisco": "CA", "los angeles": "CA", "san jose": "CA",
    "mountain view": "CA", "palo alto": "CA", "san diego": "CA",
    "new york": "NY", "new york city": "NY", "brooklyn": "NY",
    "chicago": "IL", "seattle": "WA", "bellevue": "WA", "redmond": "WA",
    "boston": "MA", "cambridge": "MA", "austin": "TX", "houston": "TX",
    "dallas": "TX", "san antonio": "TX", "denver": "CO", "boulder": "CO",
    "atlanta": "GA", "miami": "FL", "orlando": "FL", "philadelphia": "PA",
    "pittsburgh": "PA", "phoenix": "AZ", "las vegas": "NV",
    "salt lake city": "UT", "minneapolis": "MN", "detroit": "MI",
    "nashville": "TN", "charlotte": "NC", "raleigh": "NC",
}

# Roughly which countries a continent- or market-level label covers. Used only
# to rule a place *out*: a posting scoped to Asia is not one in the US.
MACRO_REGION_COUNTRIES: dict[str, set[str]] = {
    "Asia": {
        "India", "Singapore", "Japan", "China", "Hong Kong", "Taiwan",
        "South Korea", "Malaysia", "Thailand", "Vietnam", "Philippines",
        "Indonesia", "Israel",
    },
    "APAC": {
        "India", "Singapore", "Japan", "China", "Hong Kong", "Taiwan",
        "South Korea", "Malaysia", "Thailand", "Vietnam", "Philippines",
        "Indonesia", "Australia", "New Zealand",
    },
    "Europe": {
        "United Kingdom", "Germany", "France", "Netherlands", "Ireland",
        "Spain", "Italy", "Poland", "Portugal", "Sweden", "Denmark",
        "Norway", "Finland", "Switzerland", "Belgium", "Austria",
        "Romania", "Czech Republic", "Hungary", "Greece", "Ukraine",
    },
    "North America": {"United States", "Canada", "Mexico"},
    "Americas": {
        "United States", "Canada", "Mexico", "Brazil", "Argentina",
        "Colombia", "Chile",
    },
    "LATAM": {"Mexico", "Brazil", "Argentina", "Colombia", "Chile"},
    "Middle East": {
        "United Arab Emirates", "Saudi Arabia", "Israel", "Turkey", "Egypt",
    },
    "Africa": {"South Africa", "Nigeria", "Kenya", "Egypt"},
    "Oceania": {"Australia", "New Zealand"},
    "South America": {"Brazil", "Argentina", "Colombia", "Chile"},
    "EMEA": {
        "United Kingdom", "Germany", "France", "Netherlands", "Ireland",
        "Spain", "Italy", "Poland", "Portugal", "Sweden", "Denmark",
        "Norway", "Finland", "Switzerland", "Belgium", "Austria",
        "Romania", "Czech Republic", "Hungary", "Greece", "Ukraine",
        "United Arab Emirates", "Saudi Arabia", "Israel", "Turkey",
        "Egypt", "South Africa", "Nigeria", "Kenya",
    },
}


def macro_region_covers(region: str, country: str) -> bool | None:
    """Whether a macro region contains a country.

    ``None`` when the label is not a macro region, or is global and so covers
    everywhere — in both cases there is nothing to rule out.
    """
    if not region or not country or region == "Global":
        return None
    countries = MACRO_REGION_COUNTRIES.get(region)
    if countries is None:
        return None
    return country in countries


def _infer_country(place: Place) -> Place:
    """Fill in a country only where the place already implies one."""
    if place.country:
        if not place.city and place.country in CITY_STATES:
            return Place(
                place.country, place.region, place.country, place.is_remote, place.raw
            )
        return place
    if place.region and place.region in US_STATE_CODES:
        return Place(place.city, place.region, "United States", place.is_remote, place.raw)
    if place.city:
        key = _normalize(place.city)
        known = CITY_COUNTRY.get(key, UNKNOWN)
        if known:
            # A bare US city also implies its state, which lets a region search
            # tell Chicago from Houston without spelling either state out.
            region = place.region or CITY_REGION.get(key, UNKNOWN)
            return Place(place.city, region, known, place.is_remote, place.raw)
    return place


def parse_places(location: str) -> list[Place]:
    """Read a location field that may list several places.

    Boards separate multiple locations with ";" ("Mountain View, California;
    San Francisco, California"). Each is parsed independently.
    """
    raw = (location or "").strip()
    if not raw:
        return [Place(raw=raw)]
    parts = [part for part in re.split(r"\s*;\s*", raw) if part.strip()]
    seen: set[tuple[str, str, str]] = set()
    places: list[Place] = []
    for part in parts or [raw]:
        for place in _expand(part):
            key = (place.city, place.region, place.country)
            if key in seen:
                continue
            seen.add(key)
            places.append(place)
    return places or [Place(raw=raw)]


def _expand(part: str) -> list[Place]:
    """Parse one ";"-delimited part, which may itself name several cities."""
    text = (part or "").strip()
    if not text:
        return [Place(raw=part or "")]
    is_remote = bool(_REMOTE_PREFIX_RE.match(text) or _REMOTE_BARE_RE.match(text))
    body = _REMOTE_PREFIX_RE.sub("", text).strip()
    if _REMOTE_BARE_RE.match(body) or _normalize(body) in NON_PLACES:
        return [parse_place(text)]
    # "US-SF, US-Seattle, US-NYC or US-Remote" is a list of code-prefixed
    # places, not one address. Each item is read on its own.
    items = [item.strip() for item in re.split(r"\s*,\s*|\s+or\s+", body) if item.strip()]
    prefixed = [item for item in items if _CODE_PREFIX_RE.match(item)]
    if len(prefixed) > 1:
        return [parse_place(item) for item in prefixed]
    if _CODE_PREFIX_RE.match(body):
        return [parse_place(text)]
    body = _OFFICE_CODE_RE.sub("", body).strip()
    if "/" in body:
        body = body.rsplit("/", 1)[-1].strip()
    return _parse_tokens(body, is_remote=is_remote, raw=text)
