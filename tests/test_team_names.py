"""Tests for team-name canonicalization (worldcup.data.team_names)."""

import pandas as pd
import pytest

from worldcup.data.team_names import (
    UnknownTeamError,
    find_unknown_teams,
    normalize_team_columns,
    normalize_team_name,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # United States
        ("United States", "United States"),
        ("USA", "United States"),
        ("United States of America", "United States"),
        # South Korea
        ("South Korea", "South Korea"),
        ("Korea Republic", "South Korea"),
        ("Republic of Korea", "South Korea"),
        # Ivory Coast — accent + straight/curly apostrophe variants
        ("Ivory Coast", "Ivory Coast"),
        ("Cote d'Ivoire", "Ivory Coast"),
        ("Côte d'Ivoire", "Ivory Coast"),
        ("Côte d’Ivoire", "Ivory Coast"),
        # Europe / Asia / Africa
        ("Czechia", "Czech Republic"),
        ("Czech Republic", "Czech Republic"),
        ("Türkiye", "Turkey"),
        ("Turkiye", "Turkey"),
        ("IR Iran", "Iran"),
        ("Iran", "Iran"),
        ("Congo DR", "DR Congo"),
        ("DR Congo", "DR Congo"),
        ("Holland", "Netherlands"),
        ("Netherlands", "Netherlands"),
        ("Saudi", "Saudi Arabia"),
        ("Bosnia-Herzegovina", "Bosnia and Herzegovina"),
        ("Bosnia and Herzegovina", "Bosnia and Herzegovina"),
        ("Ireland", "Republic of Ireland"),
    ],
)
def test_aliases_resolve_to_canonical(raw: str, expected: str) -> None:
    assert normalize_team_name(raw) == expected


def test_matching_is_case_and_whitespace_insensitive() -> None:
    assert normalize_team_name("  korea republic  ") == "South Korea"
    assert normalize_team_name("usa") == "United States"


def test_canonical_or_unknown_passes_through_cleaned() -> None:
    assert normalize_team_name("Brazil") == "Brazil"
    assert normalize_team_name("  Brazil ") == "Brazil"
    # Unknown names are returned cleaned, never guessed into something else.
    assert normalize_team_name("Wakanda") == "Wakanda"


@pytest.mark.parametrize("team", ["England", "Scotland", "Wales", "Northern Ireland"])
def test_home_nations_stay_separate(team: str) -> None:
    assert normalize_team_name(team) == team


def test_home_nations_are_four_distinct_teams() -> None:
    names = {normalize_team_name(t) for t in ["England", "Scotland", "Wales", "Northern Ireland"]}
    assert len(names) == 4


def test_empty_and_non_string_raise_clearly() -> None:
    with pytest.raises(ValueError):
        normalize_team_name("   ")
    with pytest.raises(TypeError):
        normalize_team_name(None)  # type: ignore[arg-type]


def test_normalize_team_columns_normalizes_and_returns_copy() -> None:
    df = pd.DataFrame(
        {
            "home_team": ["USA", "Korea Republic"],
            "away_team": ["Türkiye", "Cote d'Ivoire"],
            "home_score": [1, 2],
        }
    )
    out = normalize_team_columns(df, ["home_team", "away_team"])
    assert list(out["home_team"]) == ["United States", "South Korea"]
    assert list(out["away_team"]) == ["Turkey", "Ivory Coast"]
    # Original untouched; non-team columns preserved.
    assert list(df["home_team"]) == ["USA", "Korea Republic"]
    assert list(out["home_score"]) == [1, 2]


def test_normalize_team_columns_missing_column_raises() -> None:
    df = pd.DataFrame({"home_team": ["USA"]})
    with pytest.raises(KeyError):
        normalize_team_columns(df, ["home_team", "away_team"])


def test_find_unknown_teams_flags_only_unmapped() -> None:
    df = pd.DataFrame(
        {
            "home_team": ["USA", "Narnia"],
            "away_team": ["Iran", "Wales"],
        }
    )
    known = {"United States", "Iran", "Wales", "South Korea"}
    assert find_unknown_teams(df, ["home_team", "away_team"], known) == ["Narnia"]


def test_normalize_team_columns_raises_on_unknown_with_helpful_message() -> None:
    df = pd.DataFrame({"home_team": ["USA"], "away_team": ["Narnia"]})
    known = {"United States", "Iran"}
    with pytest.raises(UnknownTeamError) as exc_info:
        normalize_team_columns(df, ["home_team", "away_team"], known_teams=known)
    message = str(exc_info.value)
    assert "Narnia" in message
    assert "team_name_map.yaml" in message


def test_known_teams_pass_validation() -> None:
    df = pd.DataFrame({"home_team": ["USA"], "away_team": ["Korea Republic"]})
    known = {"United States", "South Korea"}
    out = normalize_team_columns(df, ["home_team", "away_team"], known_teams=known)
    assert list(out["away_team"]) == ["South Korea"]
