"""FIFA group-stage tiebreakers.

Applies the primary FIFA criteria in order: points, then goal difference, then
goals scored. Remaining criteria (head-to-head, fair-play, drawing of lots) are
deferred -- the Monte Carlo rarely needs them and they add real complexity.

Two rankers share the same primary key:
    * :func:`rank_standings` breaks any final tie by team name (deterministic,
      no RNG) -- handy for fixtures and display.
    * :func:`rank_group` breaks it with a **drawing of lots** drawn from a passed
      numpy ``Generator`` -- the Monte-Carlo-correct fallback, reproducible from a
      seed and never touching global random state.
"""

from __future__ import annotations

import numpy as np

from worldcup.simulation.group_stage import TeamStanding


def _primary_key(standing: TeamStanding) -> tuple[int, int, int]:
    """Sort key for the primary FIFA criteria (all descending via negation)."""
    return (-standing.points, -standing.goal_difference, -standing.goals_for)


def rank_standings(standings: list[TeamStanding]) -> list[TeamStanding]:
    """Order standings best-to-worst, breaking final ties by team name.

    Order: points (desc), goal difference (desc), goals for (desc), then team
    name (asc) as a deterministic, documented stand-in for the remaining
    tiebreakers.

    Args:
        standings: Unordered standings for a single group.

    Returns:
        A new list ordered best-to-worst.

    TODO(slice 6): add head-to-head results and the FIFA fair-play fallbacks for
    full rule fidelity.
    """
    return sorted(standings, key=lambda s: (*_primary_key(s), s.team))


def rank_group(standings: list[TeamStanding], rng: np.random.Generator) -> list[TeamStanding]:
    """Order standings best-to-worst, breaking final ties by a random draw.

    Tiebreaker order: points → goal difference → goals for → **random drawing of
    lots** from ``rng``. The random keys are assigned in team-name order, so the
    result is reproducible for a given generator seed regardless of the input
    order.

    Args:
        standings: Unordered standings (one group, or the third-placed teams
            across groups).
        rng: A seeded numpy ``Generator`` (never the global random state).

    Returns:
        A new list ordered best-to-worst.
    """
    # Draw one lot per team in a canonical order so reproducibility does not
    # depend on how the caller ordered ``standings``.
    lots = {team: rng.random() for team in sorted(s.team for s in standings)}
    return sorted(standings, key=lambda s: (*_primary_key(s), lots[s.team]))
