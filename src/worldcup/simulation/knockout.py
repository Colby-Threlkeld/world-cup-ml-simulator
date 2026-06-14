"""Knockout-stage simulation -- no draws, a winner always advances.

A knockout match is resolved by a single Bernoulli draw on the favourite's
advance probability. Because the model emits a three-way ``(team_a_win, draw,
team_b_win)`` distribution but a knockout cannot be drawn, the draw mass is
**redistributed proportionally to the two win probabilities** -- which is exactly
conditioning on "the match is decided", ``p_a / (p_a + p_b)``. This is the
documented stand-in for explicit extra-time / penalty-shootout modelling; teams
that draw 90 minutes are assumed to win the shootout in proportion to their
relative strength, which is a reasonable first approximation.

:func:`simulate_bracket` plays an ordered seeding list down to a single champion
and records which teams reached each round, so the Monte Carlo can aggregate
per-team advancement and title probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from worldcup.simulation.group_stage import PredictFn

# Number of remaining teams -> the advancement-tracking key for that round.
ADVANCEMENT_ROUNDS: dict[int, str] = {
    32: "reach_round_32",
    16: "reach_round_16",
    8: "reach_quarterfinal",
    4: "reach_semifinal",
    2: "reach_final",
}
CHAMPION_KEY = "champion"
#: All advancement keys, outermost round first.
ADVANCEMENT_KEYS: tuple[str, ...] = (*ADVANCEMENT_ROUNDS.values(), CHAMPION_KEY)


@dataclass(frozen=True)
class KnockoutResult:
    """Outcome of one simulated bracket: the champion and per-round survivors."""

    champion: str
    #: advancement key -> teams that reached that round (e.g. 32, 16, ... 1 long).
    advancement: dict[str, list[str]]


def resolve_knockout(
    home_team: str,
    away_team: str,
    home_win_prob: float,
    rng: np.random.Generator,
) -> str:
    """Return the winner of a knockout match (never a draw).

    Args:
        home_team: First team (the one ``home_win_prob`` refers to).
        away_team: Second team.
        home_win_prob: Probability the first team advances, in ``[0, 1]``.
        rng: Seeded NumPy generator, for reproducible simulations.

    Returns:
        The name of the advancing team.

    Raises:
        ValueError: If ``home_win_prob`` is outside ``[0, 1]``.
    """
    if not 0.0 <= home_win_prob <= 1.0:
        raise ValueError(f"home_win_prob must be in [0, 1], got {home_win_prob}")
    return home_team if rng.random() < home_win_prob else away_team


def knockout_advance_probability(probs: Sequence[float]) -> float:
    """Convert a 3-way ``(p_a_win, p_draw, p_b_win)`` forecast into P(team A advances).

    Redistributes the draw probability proportionally to the two win
    probabilities, i.e. ``p_a / (p_a + p_b)``. If both win probabilities are zero
    (a degenerate all-draw forecast) the match is a coin flip (``0.5``).

    Raises:
        ValueError: If ``probs`` is not three non-negative numbers.
    """
    arr = np.asarray(probs, dtype=float)
    if arr.shape != (3,) or (arr < 0).any():
        raise ValueError(f"expected 3 non-negative probabilities, got {probs!r}")
    p_a, _, p_b = arr
    decisive = p_a + p_b
    if decisive <= 0:
        return 0.5
    return float(p_a / decisive)


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    predict: PredictFn,
    rng: np.random.Generator,
) -> str:
    """Simulate one knockout match and return the advancing team (never a draw).

    Args:
        team_a: First team (treated as the home/reference side).
        team_b: Second team.
        predict: Returns ``(p_a_win, p_draw, p_b_win)`` for the pairing.
        rng: Seeded numpy ``Generator`` (no global random state).

    Returns:
        The advancing team's name.
    """
    advance_prob = knockout_advance_probability(predict(team_a, team_b))
    return resolve_knockout(team_a, team_b, advance_prob, rng)


def simulate_bracket(
    seeding: Sequence[str],
    predict: PredictFn,
    rng: np.random.Generator,
) -> KnockoutResult:
    """Play a single-elimination bracket from an ordered seeding to one champion.

    The seeding list defines the bracket: adjacent pairs ``(0,1), (2,3), …`` meet
    in the first round, their winners meet next, and so on. Its length must be a
    power of two (32 for a full World Cup round of 32, but smaller brackets work
    for testing). Every team in ``seeding`` is recorded as having reached the
    opening round; each round's winners are recorded under the matching key.

    Args:
        seeding: Teams in bracket order. Length must be a power of two >= 2.
        predict: Match predictor.
        rng: Seeded numpy ``Generator``.

    Returns:
        A :class:`KnockoutResult` with the champion and per-round survivors.

    Raises:
        ValueError: If ``seeding`` is not a power-of-two length >= 2, or has
            duplicate teams.
    """
    current = list(seeding)
    if len(current) < 2 or not _is_power_of_two(len(current)):
        raise ValueError(f"seeding length must be a power of two >= 2, got {len(current)}")
    if len(set(current)) != len(current):
        raise ValueError("seeding contains duplicate teams")

    advancement: dict[str, list[str]] = {_round_key(len(current)): list(current)}
    while len(current) > 1:
        current = [
            simulate_knockout_match(current[i], current[i + 1], predict, rng)
            for i in range(0, len(current), 2)
        ]
        advancement[_round_key(len(current))] = list(current)
    return KnockoutResult(champion=current[0], advancement=advancement)


# --- internal helpers -------------------------------------------------------


def _round_key(n_teams: int) -> str:
    """Advancement key for ``n_teams`` remaining (1 == champion)."""
    if n_teams == 1:
        return CHAMPION_KEY
    return ADVANCEMENT_ROUNDS[n_teams]


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0
