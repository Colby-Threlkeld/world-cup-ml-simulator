"""Elo-rating baseline for match outcome probabilities.

Elo is the slice-2 baseline: simple, inherently leakage-safe (a rating only ever
reflects past results), and the number every later model must beat. These are
the pure rating primitives; the full backtest harness is built in slice 2.
"""

from __future__ import annotations


def expected_score(
    rating_home: float,
    rating_away: float,
    home_advantage: float = 65.0,
) -> float:
    """Compute the Elo win expectancy for the home team.

    Args:
        rating_home: Home team Elo rating.
        rating_away: Away team Elo rating.
        home_advantage: Rating points added to the home side. Use ``0`` at a
            neutral venue (most World Cup matches).

    Returns:
        The home team's expected score in ``(0, 1)`` -- a draw-inclusive win
        expectancy, not yet a calibrated win/draw/loss distribution.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_away - rating_home - home_advantage) / 400.0))


def update_rating(
    rating: float,
    expected: float,
    actual: float,
    k_factor: float = 32.0,
) -> float:
    """Return the post-match Elo rating.

    Args:
        rating: The team's current rating.
        expected: Pre-match expected score from :func:`expected_score`.
        actual: Realized score: ``1.0`` win, ``0.5`` draw, ``0.0`` loss.
        k_factor: Update step size (larger = ratings move faster).

    Returns:
        The updated rating.
    """
    return rating + k_factor * (actual - expected)
