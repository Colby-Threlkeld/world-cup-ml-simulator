"""FIFA group-stage tiebreakers.

Applies the primary FIFA criteria in order: points, then goal difference, then
goals scored. Remaining criteria (head-to-head, fair-play, drawing of lots) are
deferred -- the Monte Carlo rarely needs them and they add real complexity.
"""

from __future__ import annotations

from worldcup.simulation.group_stage import TeamStanding


def rank_standings(standings: list[TeamStanding]) -> list[TeamStanding]:
    """Order standings from first place to last by the primary FIFA criteria.

    Order: points (desc), goal difference (desc), goals for (desc), then team
    name (asc) as a deterministic, documented stand-in for the remaining
    tiebreakers.

    Args:
        standings: Unordered standings for a single group.

    Returns:
        A new list ordered best-to-worst.

    TODO(slice 6): add head-to-head results and the FIFA fair-play / drawing-of-
    lots fallbacks for full rule fidelity.
    """
    return sorted(
        standings,
        key=lambda s: (-s.points, -s.goal_difference, -s.goals_for, s.team),
    )
