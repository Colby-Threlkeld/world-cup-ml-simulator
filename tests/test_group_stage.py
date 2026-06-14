"""Tests for the group-stage simulator: outcomes, standings, tiebreakers, seeds."""

import numpy as np
import pytest

from worldcup.simulation.group_stage import (
    AWAY_WIN,
    DRAW,
    HOME_WIN,
    TeamStanding,
    build_group_table,
    select_best_third_place_teams,
    select_top_two,
    simulate_group_match,
)
from worldcup.simulation.tiebreakers import rank_group

# A 4-team round robin (matches the config's per-group schedule).
_TEAMS = ["A", "B", "C", "D"]
_FIXTURES = [("A", "B"), ("C", "D"), ("A", "C"), ("D", "B"), ("A", "D"), ("B", "C")]


def _fixed_predict(p_a: float, p_d: float, p_b: float):
    """A predictor that ignores the teams and returns fixed probabilities."""
    return lambda team_a, team_b: (p_a, p_d, p_b)


# --- simulate_group_match: outcome respects probabilities -------------------


def test_certain_home_win_outcome_and_scoreline() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        result = simulate_group_match("A", "B", _fixed_predict(1.0, 0.0, 0.0), rng)
        assert result.outcome == HOME_WIN
        assert result.home_score > result.away_score


def test_certain_draw_is_level() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        result = simulate_group_match("A", "B", _fixed_predict(0.0, 1.0, 0.0), rng)
        assert result.outcome == DRAW
        assert result.home_score == result.away_score


def test_certain_away_win_outcome_and_scoreline() -> None:
    rng = np.random.default_rng(2)
    result = simulate_group_match("A", "B", _fixed_predict(0.0, 0.0, 1.0), rng)
    assert result.outcome == AWAY_WIN
    assert result.away_score > result.home_score


def test_predict_must_return_valid_probabilities() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_group_match("A", "B", _fixed_predict(0.0, 0.0, 0.0), rng)  # sums to 0
    with pytest.raises(ValueError):
        simulate_group_match("A", "B", lambda a, b: (-1.0, 1.0, 1.0), rng)  # negative


# --- build_group_table ------------------------------------------------------


def test_build_group_table_each_team_plays_three() -> None:
    rng = np.random.default_rng(7)
    table = {s.team: s for s in build_group_table(_TEAMS, _FIXTURES, _fixed_predict(0.4, 0.2, 0.4), rng)}
    assert set(table) == set(_TEAMS)
    for standing in table.values():
        assert standing.played == 3
        # points == 3*won + 1*drawn, and goals balance out across the group.
        assert standing.points == 3 * standing.won + standing.drawn
    total_for = sum(s.goals_for for s in table.values())
    total_against = sum(s.goals_against for s in table.values())
    assert total_for == total_against  # every goal scored is conceded by someone


def test_home_team_always_wins_table_when_home_win_certain() -> None:
    rng = np.random.default_rng(3)
    table = {s.team: s for s in build_group_table(_TEAMS, _FIXTURES, _fixed_predict(1.0, 0.0, 0.0), rng)}
    # A is home in all 3 of its fixtures -> 9 points; D is home twice -> >= 6.
    assert table["A"].points == 9
    assert table["A"].won == 3


# --- tiebreakers ------------------------------------------------------------


def test_points_dominate_goal_difference() -> None:
    rng = np.random.default_rng(0)
    more_points = TeamStanding(team="X", won=1, drawn=1, goals_for=1, goals_against=0)  # 4 pts, GD +1
    fewer_points = TeamStanding(team="Y", won=1, goals_for=9, goals_against=0)  # 3 pts, GD +9
    ranked = rank_group([fewer_points, more_points], rng)
    assert [s.team for s in ranked] == ["X", "Y"]


def test_goal_difference_breaks_equal_points() -> None:
    rng = np.random.default_rng(0)
    low_gd = TeamStanding(team="X", won=1, goals_for=1, goals_against=0)  # GD +1
    high_gd = TeamStanding(team="Y", won=1, goals_for=3, goals_against=0)  # GD +3
    assert [s.team for s in rank_group([low_gd, high_gd], rng)] == ["Y", "X"]


def test_goals_for_breaks_equal_points_and_gd() -> None:
    rng = np.random.default_rng(0)
    fewer = TeamStanding(team="X", won=1, drawn=0, lost=1, goals_for=2, goals_against=1)  # GD+1, GF2
    more = TeamStanding(team="Y", won=1, drawn=0, lost=1, goals_for=4, goals_against=3)  # GD+1, GF4
    assert [s.team for s in rank_group([fewer, more], rng)] == ["Y", "X"]


def test_random_fallback_is_seed_reproducible() -> None:
    # Two completely identical records -> only the random drawing of lots decides.
    a = TeamStanding(team="A", won=1, goals_for=2, goals_against=1)
    b = TeamStanding(team="B", won=1, goals_for=2, goals_against=1)
    order1 = [s.team for s in rank_group([a, b], np.random.default_rng(42))]
    order2 = [s.team for s in rank_group([a, b], np.random.default_rng(42))]
    assert order1 == order2  # same seed -> same order
    # Input order must not change the result for a fixed seed.
    order3 = [s.team for s in rank_group([b, a], np.random.default_rng(42))]
    assert order1 == order3


def test_random_fallback_can_differ_across_seeds() -> None:
    a = TeamStanding(team="A", won=1, goals_for=2, goals_against=1)
    b = TeamStanding(team="B", won=1, goals_for=2, goals_against=1)
    orders = {tuple(s.team for s in rank_group([a, b], np.random.default_rng(seed))) for seed in range(20)}
    assert len(orders) == 2  # both orderings appear across seeds


# --- qualifier selection ----------------------------------------------------


def test_select_top_two() -> None:
    ranked = [TeamStanding(team=t) for t in ("W", "X", "Y", "Z")]
    assert [s.team for s in select_top_two(ranked)] == ["W", "X"]


def test_select_top_two_needs_two() -> None:
    with pytest.raises(ValueError):
        select_top_two([TeamStanding(team="W")])


def test_select_best_third_place_picks_eight_best() -> None:
    rng = np.random.default_rng(0)
    # 12 thirds with points 0..11; the eight best are points 4..11.
    thirds = [TeamStanding(team=f"T{p}", won=p, goals_for=p) for p in range(12)]
    chosen = select_best_third_place_teams(thirds, rng)
    assert len(chosen) == 8
    chosen_points = sorted(s.points for s in chosen)
    assert chosen_points == sorted(s.points for s in thirds)[-8:]


def test_select_best_third_place_requires_enough_teams() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        select_best_third_place_teams([TeamStanding(team="T")], rng, n=8)


# --- reproducibility end-to-end ---------------------------------------------


def test_full_group_simulation_is_reproducible() -> None:
    predict = _fixed_predict(0.45, 0.25, 0.30)

    def run(seed: int):
        table = build_group_table(_TEAMS, _FIXTURES, predict, np.random.default_rng(seed))
        ranked = rank_group(table, np.random.default_rng(seed + 100))
        return [(s.team, s.points, s.goals_for, s.goals_against) for s in ranked]

    assert run(2026) == run(2026)  # identical seed -> identical tournament


def test_different_seeds_can_produce_different_tables() -> None:
    predict = _fixed_predict(0.45, 0.25, 0.30)
    tables = set()
    for seed in range(10):
        table = build_group_table(_TEAMS, _FIXTURES, predict, np.random.default_rng(seed))
        tables.add(tuple((s.team, s.points) for s in table))
    assert len(tables) > 1  # randomness actually varies with the seed
