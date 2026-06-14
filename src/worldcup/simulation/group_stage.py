"""Group-stage standings: points, goal difference, and goals for."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WIN_POINTS = 3
DRAW_POINTS = 1


@dataclass
class TeamStanding:
    """A team's running group-stage record."""

    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        """Competition points (3 per win, 1 per draw, 0 per loss)."""
        return WIN_POINTS * self.won + DRAW_POINTS * self.drawn

    @property
    def goal_difference(self) -> int:
        """Goals scored minus goals conceded."""
        return self.goals_for - self.goals_against


def compute_standings(results: pd.DataFrame) -> list[TeamStanding]:
    """Compute group standings from played matches.

    Args:
        results: Matches with columns ``home_team``, ``away_team``,
            ``home_score``, ``away_score``.

    Returns:
        One :class:`TeamStanding` per team that appears in the matches.
    """
    table: dict[str, TeamStanding] = {}

    def standing_for(team: str) -> TeamStanding:
        if team not in table:
            table[team] = TeamStanding(team=team)
        return table[team]

    for row in results.itertuples(index=False):
        home = standing_for(row.home_team)
        away = standing_for(row.away_team)
        home_score, away_score = int(row.home_score), int(row.away_score)

        home.played += 1
        away.played += 1
        home.goals_for += home_score
        home.goals_against += away_score
        away.goals_for += away_score
        away.goals_against += home_score

        if home_score > away_score:
            home.won += 1
            away.lost += 1
        elif home_score < away_score:
            away.won += 1
            home.lost += 1
        else:
            home.drawn += 1
            away.drawn += 1

    return list(table.values())
