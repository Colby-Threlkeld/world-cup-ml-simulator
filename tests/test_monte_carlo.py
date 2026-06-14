"""Tests for the Monte Carlo tournament simulator."""

import numpy as np
import pandas as pd
import pytest

from worldcup.data.validate_data import validate_probability_bounds
from worldcup.simulation.tournament import (
    PROBABILITY_COLUMNS,
    load_tournament_config,
    simulate_tournament,
    strength_predict_fn,
    summarize_simulation,
    uniform_predict_fn,
)

CONFIG = load_tournament_config()
SIMS = 200  # small but enough for shape/bounds/invariants; the exact-count tests
#             hold at any N, and a 100k-strength favourite dominates well before 200.


@pytest.fixture(scope="module")
def uniform_probs() -> pd.DataFrame:
    return simulate_tournament(CONFIG, uniform_predict_fn(), n_simulations=SIMS, seed=42)


# --- output shape -----------------------------------------------------------


def test_output_shape(uniform_probs: pd.DataFrame) -> None:
    assert list(uniform_probs.columns) == ["team", *PROBABILITY_COLUMNS]
    assert len(uniform_probs) == CONFIG.total_teams  # one row per team (48)
    assert set(uniform_probs["team"]) == set(CONFIG.teams())


# --- probability bounds -----------------------------------------------------


def test_all_probabilities_within_bounds(uniform_probs: pd.DataFrame) -> None:
    # Reuse the shared validator — raises if anything is outside [0, 1].
    validate_probability_bounds(uniform_probs, list(PROBABILITY_COLUMNS))


def test_title_probability_sums_to_one(uniform_probs: pd.DataFrame) -> None:
    assert uniform_probs["win_world_cup_probability"].sum() == pytest.approx(1.0)


def test_round_probabilities_are_monotonic(uniform_probs: pd.DataFrame) -> None:
    # A team that reaches a later round must have reached every earlier one.
    order = [
        "reach_round_32_probability",
        "reach_round_16_probability",
        "reach_quarterfinal_probability",
        "reach_semifinal_probability",
        "reach_final_probability",
        "win_world_cup_probability",
    ]
    for earlier, later in zip(order, order[1:]):  # consecutive pairs (unequal lengths)
        assert (uniform_probs[earlier] >= uniform_probs[later] - 1e-9).all()


def test_aggregate_counts_match_format(uniform_probs: pd.DataFrame) -> None:
    # Per simulation: 12 group winners, 32 qualifiers, 2 finalists, 1 champion.
    assert uniform_probs["win_group_probability"].sum() == pytest.approx(12.0)
    assert uniform_probs["reach_round_32_probability"].sum() == pytest.approx(32.0)
    assert uniform_probs["reach_final_probability"].sum() == pytest.approx(2.0)
    assert uniform_probs["win_world_cup_probability"].sum() == pytest.approx(1.0)


# --- reproducibility --------------------------------------------------------


def test_same_seed_is_reproducible() -> None:
    a = simulate_tournament(CONFIG, uniform_predict_fn(), n_simulations=SIMS, seed=7)
    b = simulate_tournament(CONFIG, uniform_predict_fn(), n_simulations=SIMS, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_can_differ() -> None:
    a = simulate_tournament(CONFIG, uniform_predict_fn(), n_simulations=SIMS, seed=1)
    b = simulate_tournament(CONFIG, uniform_predict_fn(), n_simulations=SIMS, seed=2)
    a_title = a.set_index("team")["win_world_cup_probability"]
    b_title = b.set_index("team")["win_world_cup_probability"]
    assert not np.allclose(a_title, b_title.reindex(a_title.index))


# --- the favourite dominates ------------------------------------------------


def test_overwhelming_favourite_wins_most() -> None:
    strengths = dict.fromkeys(CONFIG.teams(), 0.0)
    strengths["A1"] = 100_000.0  # absurdly strong
    probs = simulate_tournament(CONFIG, strength_predict_fn(strengths), n_simulations=SIMS, seed=3)
    top = probs.iloc[0]
    assert top["team"] == "A1"
    assert top["win_world_cup_probability"] > probs["win_world_cup_probability"].drop(0).max()


# --- guards & summary -------------------------------------------------------


def test_non_positive_simulations_raises() -> None:
    with pytest.raises(ValueError):
        simulate_tournament(CONFIG, uniform_predict_fn(), n_simulations=0)


def test_summary_structure(uniform_probs: pd.DataFrame) -> None:
    summary = summarize_simulation(
        uniform_probs,
        n_simulations=SIMS,
        seed=42,
        runtime_seconds=1.23,
        draw_status=CONFIG.draw_status,
    )
    assert summary["n_simulations"] == SIMS
    assert summary["n_teams"] == CONFIG.total_teams
    assert summary["title_probability_total"] == pytest.approx(1.0, abs=1e-6)
    assert summary["draw_status"] == "placeholder"
    assert len(summary["top_title_contenders"]) == 10
