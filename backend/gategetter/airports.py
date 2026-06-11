"""
Airport registry: maps IATA codes to Flighty.com URL slugs.

To add an airport, find its slug at https://flighty.com/airports
and add an entry below. The slug is the path segment between
/airports/ and /departures.
"""

AIRPORT_SLUGS = {
    "OAK": "metropolitan-oakland-intl-oak",
    "SFO": "san-francisco-intl-sfo",
    "SJC": "san-jose-mineta-intl-sjc",
    "LAX": "los-angeles-intl-lax",
    "JFK": "john-f-kennedy-intl-jfk",
    "ORD": "chicago-ohare-intl-ord",
    "ATL": "hartsfield-jackson-atlanta-intl-atl",
    "DAL": "dallas-love-field-dal",
    "DFW": "dallas-fort-worth-intl-dfw",
    "DEN": "denver-intl-den",
    "MIA": "miami-intl-mia",
    "SEA": "seattle-tacoma-intl-sea",
    "IAH": "george-bush-intercontinental-iah",
    "MCO": "orlando-intl-mco",
    "CLT": "charlotte-douglas-intl-clt",
    "IAD": "washington-dulles-intl-iad",
    "PDX": "portland-intl-pdx",
    "HNL": "daniel-k-inouye-intl-hnl",
    "SAN": "san-diego-intl-san",
    "SJU": "luis-munoz-marin-intl-sju",
}


def build_url(iata_code: str) -> str:
    """Build the Flighty departures URL for a given IATA code."""
    code = iata_code.upper()
    slug = AIRPORT_SLUGS.get(code)
    if not slug:
        raise ValueError(
            f"Unknown airport '{code}'. Add it to AIRPORT_SLUGS in airports.py. "
            f"Find the slug at https://flighty.com/airports"
        )
    return f"https://flighty.com/airports/{slug}/departures"


def list_supported() -> list[str]:
    """Return sorted list of supported IATA codes."""
    return sorted(AIRPORT_SLUGS.keys())
