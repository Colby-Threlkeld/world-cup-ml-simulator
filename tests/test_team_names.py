"""Tests for team-name canonicalization."""

from worldcup.data.team_names import load_team_name_map, normalize_team_name


def test_normalize_known_alias():
    mapping = {"USA": "United States"}
    assert normalize_team_name("USA", mapping) == "United States"


def test_normalize_collapses_whitespace():
    assert normalize_team_name("  England  ", {}) == "England"
    assert normalize_team_name("Bosnia   and  Herzegovina", {}) == "Bosnia and Herzegovina"


def test_normalize_is_case_insensitive():
    mapping = {"Korea Republic": "South Korea"}
    assert normalize_team_name("korea republic", mapping) == "South Korea"


def test_normalize_unknown_passes_through():
    assert normalize_team_name("Brazil", {"USA": "United States"}) == "Brazil"


def test_load_map_from_shipped_config():
    mapping = load_team_name_map()
    assert isinstance(mapping, dict)
    assert mapping, "config should ship with at least one alias"
    # The shipped config canonicalizes USA -> United States.
    assert normalize_team_name("USA", mapping) == "United States"
