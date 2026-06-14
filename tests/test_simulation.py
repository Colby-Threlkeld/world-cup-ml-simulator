"""Tests for group standings, tiebreakers, and knockout resolution."""

import numpy as np
import pandas as pd
import pytest

from worldcup.simulation.group_stage import TeamStanding, compute_standings
from worldcup.simulation.knockout import resolve_knockout
from worldcup.simulation.tiebreakers import rank_standings


def _group_results() -> pd.DataFrame:
    # A beats B 2-0, A draws C 1-1, B beats C 3-1.
    return pd.DataFrame(
        {
            "home_team": ["A", "A", "B"],
            "away_team": ["B", "C", "C"],
            "home_score": [2, 1, 3],
            "away_score": [0, 1, 1],
        }
    )


def test_standings_points_and_goal_difference():
    standings = {s.team: s for s in compute_standings(_group_results())}
    assert standings["A"].points == 4  # one win, one draw
    assert standings["A"].goal_difference == 2
    assert standings["B"].points == 3
    assert standings["C"].points == 1


def test_rank_orders_by_points_then_gd():
    ranked = rank_standings(compute_standings(_group_results()))
    assert [s.team for s in ranked] == ["A", "B", "C"]


def test_rank_breaks_tie_on_goal_difference():
    t1 = TeamStanding(team="X", won=1, goals_for=1, goals_against=0)  # 3 pts, GD +1
    t2 = TeamStanding(team="Y", won=1, goals_for=3, goals_against=0)  # 3 pts, GD +3
    assert [s.team for s in rank_standings([t1, t2])] == ["Y", "X"]


def test_knockout_is_deterministic_at_extremes():
    rng = np.random.default_rng(42)
    assert resolve_knockout("A", "B", 1.0, rng) == "A"
    assert resolve_knockout("A", "B", 0.0, rng) == "B"


def test_knockout_rejects_out_of_range_probability():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        resolve_knockout("A", "B", 1.5, rng)
