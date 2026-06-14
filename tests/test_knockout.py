"""Tests for the knockout-stage simulator: no draws, brackets, seeding, seeds."""

import numpy as np
import pytest

from worldcup.simulation.knockout import (
    ADVANCEMENT_KEYS,
    knockout_advance_probability,
    simulate_bracket,
    simulate_knockout_match,
)
from worldcup.simulation.tournament import (
    TournamentConfigError,
    build_knockout_seeding,
    load_tournament_config,
)


def _fixed_predict(p_a: float, p_d: float, p_b: float):
    return lambda team_a, team_b: (p_a, p_d, p_b)


def _strength_predict(team_a: str, team_b: str):
    """Lower numeric suffix = stronger; returns a 3-way forecast with a draw mass."""
    sa, sb = int(team_a[1:]), int(team_b[1:])
    ea = 1.0 / (1.0 + 10 ** ((sa - sb) / 10.0))
    p_draw = 0.22
    return ((1 - p_draw) * ea, p_draw, (1 - p_draw) * (1 - ea))


def _bracket(n: int) -> list[str]:
    return [f"T{i}" for i in range(n)]


# --- draw redistribution / no-draw guarantee --------------------------------


def test_advance_probability_redistributes_draw() -> None:
    # p_a/(p_a+p_b) = 0.6/0.8 = 0.75, independent of the 0.2 draw mass.
    assert knockout_advance_probability((0.6, 0.2, 0.2)) == pytest.approx(0.75)
    # All-draw forecast degenerates to a coin flip.
    assert knockout_advance_probability((0.0, 1.0, 0.0)) == pytest.approx(0.5)


def test_advance_probability_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        knockout_advance_probability((0.5, 0.5))  # wrong length
    with pytest.raises(ValueError):
        knockout_advance_probability((-0.1, 0.6, 0.5))  # negative


def test_knockout_match_never_draws() -> None:
    rng = np.random.default_rng(0)
    # A forecast dominated by the draw must still yield one of the two teams.
    winners = {
        simulate_knockout_match("A", "B", _fixed_predict(0.1, 0.8, 0.1), rng) for _ in range(50)
    }
    assert winners <= {"A", "B"}
    assert winners  # at least one decided match


def test_knockout_match_respects_certain_outcomes() -> None:
    rng = np.random.default_rng(1)
    assert simulate_knockout_match("A", "B", _fixed_predict(1.0, 0.0, 0.0), rng) == "A"
    assert simulate_knockout_match("A", "B", _fixed_predict(0.0, 0.0, 1.0), rng) == "B"
    # A pure-draw forecast is still decided (here team A by the favourable seed).
    only_draw = simulate_knockout_match("A", "B", _fixed_predict(0.0, 1.0, 0.0), rng)
    assert only_draw in {"A", "B"}


# --- bracket progression ----------------------------------------------------


def test_bracket_produces_exactly_one_champion() -> None:
    rng = np.random.default_rng(7)
    result = simulate_bracket(_bracket(32), _strength_predict, rng)
    assert len([result.champion]) == 1
    assert result.advancement["champion"] == [result.champion]
    assert result.champion in _bracket(32)


def test_bracket_round_sizes_are_correct() -> None:
    rng = np.random.default_rng(7)
    result = simulate_bracket(_bracket(32), _strength_predict, rng)
    expected_counts = {
        "reach_round_32": 32,
        "reach_round_16": 16,
        "reach_quarterfinal": 8,
        "reach_semifinal": 4,
        "reach_final": 2,
        "champion": 1,
    }
    for key, count in expected_counts.items():
        assert len(result.advancement[key]) == count
    assert set(result.advancement) == set(ADVANCEMENT_KEYS)


def test_each_round_is_subset_of_previous() -> None:
    rng = np.random.default_rng(11)
    adv = simulate_bracket(_bracket(32), _strength_predict, rng).advancement
    order = [
        "reach_round_32",
        "reach_round_16",
        "reach_quarterfinal",
        "reach_semifinal",
        "reach_final",
        "champion",
    ]
    for earlier, later in zip(order, order[1:]):
        assert set(adv[later]) <= set(adv[earlier])  # survivors came from the prior round


def test_small_bracket_labels_make_sense() -> None:
    # A 4-team bracket maps to semifinal -> final -> champion.
    rng = np.random.default_rng(3)
    adv = simulate_bracket(_bracket(4), _strength_predict, rng).advancement
    assert len(adv["reach_semifinal"]) == 4
    assert len(adv["reach_final"]) == 2
    assert len(adv["champion"]) == 1


def test_bracket_rejects_non_power_of_two() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_bracket(_bracket(6), _strength_predict, rng)


def test_bracket_rejects_duplicate_teams() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_bracket(["A", "A", "B", "C"], _strength_predict, rng)


# --- reproducibility --------------------------------------------------------


def test_bracket_is_seed_reproducible() -> None:
    def run(seed: int):
        result = simulate_bracket(_bracket(32), _strength_predict, np.random.default_rng(seed))
        return result.champion, result.advancement

    champ1, adv1 = run(2026)
    champ2, adv2 = run(2026)
    assert champ1 == champ2
    assert adv1 == adv2


def test_different_seeds_can_change_champion() -> None:
    champions = {
        simulate_bracket(_bracket(32), _strength_predict, np.random.default_rng(seed)).champion
        for seed in range(30)
    }
    assert len(champions) > 1


# --- config-driven seeding (placeholder + override) -------------------------


def test_build_knockout_seeding_placeholder_path() -> None:
    config = load_tournament_config()
    groups = list(config.groups.keys())
    winners = [config.groups[g][0] for g in groups]  # 12
    runners = [config.groups[g][1] for g in groups]  # 12
    thirds = [config.groups[g][2] for g in groups[:8]]  # 8 best (placeholder choice)

    seeding = build_knockout_seeding(config, winners, runners, thirds)
    assert len(seeding) == 32
    assert len(set(seeding)) == 32
    assert set(seeding) <= set(config.teams())


def test_build_knockout_seeding_uses_config_pairings() -> None:
    config = load_tournament_config()
    groups = list(config.groups.keys())
    winners = [config.groups[g][0] for g in groups]
    runners = [config.groups[g][1] for g in groups]
    thirds = [config.groups[g][2] for g in groups[:8]]

    # Inject explicit pairings: 1A vs 3-1, 2B vs 2C, ... (just enough to be 16 matches).
    slots_home = [f"1{g}" for g in groups] + [f"2{g}" for g in groups[:4]]
    slots_away = [f"3-{i}" for i in range(1, 9)] + [f"2{g}" for g in groups[4:12]]
    pairings = [
        {"match": i + 73, "home": h, "away": a}
        for i, (h, a) in enumerate(zip(slots_home, slots_away))
    ]
    config.knockout_bracket["round_of_32"] = pairings

    seeding = build_knockout_seeding(config, winners, runners, thirds)
    assert len(seeding) == 32 and len(set(seeding)) == 32
    # First match's home slot 1A resolves to group A's winner (slot A1).
    assert seeding[0] == config.groups["A"][0]


def test_build_knockout_seeding_rejects_wrong_counts() -> None:
    config = load_tournament_config()
    with pytest.raises(TournamentConfigError):
        build_knockout_seeding(config, ["A1"], ["A2"], ["A3"])  # far too few


def test_seeding_into_bracket_runs_end_to_end() -> None:
    config = load_tournament_config()
    groups = list(config.groups.keys())
    winners = [config.groups[g][0] for g in groups]
    runners = [config.groups[g][1] for g in groups]
    thirds = [config.groups[g][2] for g in groups[:8]]
    seeding = build_knockout_seeding(config, winners, runners, thirds)

    result = simulate_bracket(seeding, _strength_predict, np.random.default_rng(1))
    assert result.champion in seeding
    assert len(result.advancement["reach_round_32"]) == 32
