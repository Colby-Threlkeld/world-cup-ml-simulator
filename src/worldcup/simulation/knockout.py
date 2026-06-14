"""Knockout-match resolution -- no draws, a winner always advances.

For the baseline simulation a knockout is resolved by a single Bernoulli draw on
the favorite's advance probability. Slice 4's Poisson model will replace this
with simulated scorelines plus explicit extra-time and penalty-shootout logic.
"""

from __future__ import annotations

import numpy as np


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
