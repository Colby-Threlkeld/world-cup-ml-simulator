"""Tests for the 2026 World Cup tournament configuration system.

Covers loading + validating the real ``configs/tournament_2026.yaml`` and the
structural validators, using a programmatically-built config dict so each failure
mode can be triggered in isolation.
"""

import string

import pytest

from worldcup.simulation.tournament import (
    DEFAULT_TOURNAMENT_CONFIG_PATH,
    TournamentConfig,
    TournamentConfigError,
    check_tournament_config,
    load_tournament_config,
    validate_tournament_config,
)

_GROUP_LETTERS = list(string.ascii_uppercase[:12])  # A..L
_ROUND_ROBIN = [((1, 2), 1), ((3, 4), 1), ((1, 3), 2), ((4, 2), 2), ((1, 4), 3), ((2, 3), 3)]


def _valid_config_dict() -> dict:
    """A fully valid 12x4 config dict mirroring the real YAML structure."""
    groups = {g: [f"{g}{i}" for i in range(1, 5)] for g in _GROUP_LETTERS}
    fixtures = [
        {"group": g, "matchday": md, "home": f"{g}{a}", "away": f"{g}{b}"}
        for g in _GROUP_LETTERS
        for (a, b), md in _ROUND_ROBIN
    ]
    return {
        "draw_status": "placeholder",
        "format": {
            "total_teams": 48,
            "num_groups": 12,
            "teams_per_group": 4,
            "advance_per_group": 2,
            "best_third_placed_advance": 8,
            "knockout_rounds": ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"],
        },
        "hosts": ["United States", "Canada", "Mexico"],
        "groups": groups,
        "fixtures": fixtures,
        "tiebreakers": ["points", "goal_difference", "goals_for"],
        "knockout_bracket": {"draw_status": "placeholder", "round_of_32": []},
    }


# --- the real shipped config ------------------------------------------------


def test_real_config_loads_and_validates() -> None:
    config = load_tournament_config()  # validates by default
    assert config.total_teams == 48
    assert config.num_groups == 12
    assert len(config.groups) == 12
    assert all(len(members) == 4 for members in config.groups.values())
    assert len(config.teams()) == 48
    assert len(set(config.teams())) == 48


def test_real_config_fixtures_three_per_team() -> None:
    config = load_tournament_config()
    assert config.has_full_fixture_list()
    assert len(config.fixtures) == 72  # 12 groups x 6 round-robin matches
    counts = {team: 0 for team in config.teams()}
    for fx in config.fixtures:
        counts[fx.home] += 1
        counts[fx.away] += 1
    assert set(counts.values()) == {3}


def test_real_config_hosts_and_tiebreakers() -> None:
    config = load_tournament_config()
    assert config.hosts == ("United States", "Canada", "Mexico")
    assert config.tiebreakers[0] == "points"  # ordered list, points first
    assert "goal_difference" in config.tiebreakers


def test_knockout_bracket_is_configurable_placeholder() -> None:
    config = load_tournament_config()
    # Honest placeholder: pairings empty, certain qualifier slots listed.
    assert config.draw_status == "placeholder"
    assert config.knockout_bracket.get("round_of_32") == []
    slots = config.knockout_bracket["qualifier_slots"]
    assert len(slots["group_winners"]) == 12
    assert len(slots["best_third_placed"]) == 8


def test_group_lookup_helpers() -> None:
    config = load_tournament_config()
    assert config.group_of("A1") == "A"
    assert config.group_of("L4") == "L"
    assert config.group_of("nope") is None


def test_default_path_points_at_shipped_yaml() -> None:
    assert DEFAULT_TOURNAMENT_CONFIG_PATH.exists()


# --- the validators (happy path) --------------------------------------------


def test_valid_dict_has_no_errors() -> None:
    config = TournamentConfig.from_dict(_valid_config_dict())
    assert check_tournament_config(config) == []
    validate_tournament_config(config)  # must not raise


# --- the validators (each failure mode) -------------------------------------


def test_wrong_group_count_is_caught() -> None:
    data = _valid_config_dict()
    del data["groups"]["L"]  # 11 groups now
    config = TournamentConfig.from_dict(data)
    errors = check_tournament_config(config)
    assert any("group" in e for e in errors)
    with pytest.raises(TournamentConfigError):
        validate_tournament_config(config)


def test_wrong_team_per_group_is_caught() -> None:
    data = _valid_config_dict()
    data["groups"]["A"] = ["A1", "A2", "A3"]  # only 3
    errors = check_tournament_config(TournamentConfig.from_dict(data))
    assert any("group A" in e and "expected 4" in e for e in errors)


def test_duplicate_team_is_caught() -> None:
    data = _valid_config_dict()
    data["groups"]["B"][0] = "A1"  # A1 now appears in groups A and B
    errors = check_tournament_config(TournamentConfig.from_dict(data))
    assert any("duplicate" in e for e in errors)


def test_wrong_total_team_count_is_caught() -> None:
    data = _valid_config_dict()
    data["groups"]["A"] = ["A1", "A2", "A3", "A4", "A5"]  # 49 unique teams
    errors = check_tournament_config(TournamentConfig.from_dict(data))
    assert any("unique team" in e for e in errors)


def test_fixture_with_unknown_team_is_caught() -> None:
    data = _valid_config_dict()
    data["fixtures"].append({"group": "A", "matchday": 1, "home": "A1", "away": "ZZ"})
    errors = check_tournament_config(TournamentConfig.from_dict(data))
    assert any("unknown team" in e for e in errors)


def test_team_without_three_fixtures_is_caught() -> None:
    data = _valid_config_dict()
    data["fixtures"] = data["fixtures"][:-1]  # drop one A fixture -> two A teams short
    errors = check_tournament_config(TournamentConfig.from_dict(data))
    assert any("group fixtures" in e for e in errors)


def test_fixture_team_in_wrong_group_is_caught() -> None:
    data = _valid_config_dict()
    data["fixtures"][0] = {"group": "A", "matchday": 1, "home": "A1", "away": "B1"}
    errors = check_tournament_config(TournamentConfig.from_dict(data))
    assert any("stated group" in e for e in errors)


def test_missing_fixtures_skips_per_team_count() -> None:
    data = _valid_config_dict()
    data["fixtures"] = []  # no full fixture list -> per-team count not enforced
    config = TournamentConfig.from_dict(data)
    assert not config.has_full_fixture_list()
    assert check_tournament_config(config) == []  # still structurally valid


def test_malformed_config_raises_on_from_dict() -> None:
    with pytest.raises(TournamentConfigError):
        TournamentConfig.from_dict({"groups": {}})  # missing 'format'


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_tournament_config("does/not/exist.yaml")
