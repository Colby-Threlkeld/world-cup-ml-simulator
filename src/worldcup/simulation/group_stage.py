"""Group-stage simulation: simulate matches, build standings, pick qualifiers.

Each match outcome is drawn from the model's predicted ``(team_a_win, draw,
team_b_win)`` probabilities; a plausible scoreline is then sampled *conditioned on
that outcome* (Poisson goals), since standings need goals for/against. All
randomness flows through a caller-supplied numpy ``Generator`` — never the global
random state — so a seed makes a whole tournament reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

WIN_POINTS = 3
DRAW_POINTS = 1

# Outcome indices, matching the model's class order (team_a == home side).
HOME_WIN, DRAW, AWAY_WIN = 0, 1, 2

# Poisson means for the conditional scoreline sampler (cosmetic but plausible).
_DRAW_GOALS_MEAN = 1.1
_LOSER_GOALS_MEAN = 0.8
_EXTRA_MARGIN_MEAN = 0.6

#: A match predictor: ``predict(team_a, team_b) -> (p_a_win, p_draw, p_b_win)``.
PredictFn = Callable[[str, str], Sequence[float]]


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


@dataclass(frozen=True)
class MatchResult:
    """A simulated group match: who played, the scoreline, and the outcome index."""

    home_team: str
    away_team: str
    home_score: int
    away_score: int
    outcome: int  # HOME_WIN / DRAW / AWAY_WIN


def simulate_group_match(
    team_a: str,
    team_b: str,
    predict: PredictFn,
    rng: np.random.Generator,
) -> MatchResult:
    """Simulate one match: draw the outcome from model probabilities, then a score.

    ``team_a`` is the home side, so the predictor's classes map to
    ``HOME_WIN / DRAW / AWAY_WIN``. The outcome is sampled from those
    probabilities; a scoreline consistent with it is then sampled from Poisson
    goal counts (the winner always outscores the loser; a draw is level).

    Args:
        team_a: Home team.
        team_b: Away team.
        predict: Returns ``(p_a_win, p_draw, p_b_win)`` for the pairing.
        rng: Seeded numpy ``Generator`` (no global random state).

    Returns:
        The :class:`MatchResult`.

    Raises:
        ValueError: If the predicted probabilities are negative or sum to ~0.
    """
    probs = np.asarray(predict(team_a, team_b), dtype=float)
    if probs.shape != (3,) or (probs < 0).any() or probs.sum() <= 0:
        raise ValueError(f"predict must return 3 non-negative probabilities, got {probs!r}")
    probs = probs / probs.sum()  # normalize away float drift

    outcome = int(rng.choice(3, p=probs))
    home_score, away_score = _sample_scoreline(outcome, rng)
    return MatchResult(team_a, team_b, home_score, away_score, outcome)


def build_group_table(
    teams: Sequence[str],
    fixtures: Sequence[object],
    predict: PredictFn,
    rng: np.random.Generator,
) -> list[TeamStanding]:
    """Simulate every fixture in a group and return the resulting standings.

    Args:
        teams: The group's teams (each gets a standing, even with no fixtures).
        fixtures: The group's matches. Each item is either a ``(home, away)``
            pair or an object with ``.home`` / ``.away`` attributes (e.g.
            :class:`worldcup.simulation.tournament.GroupFixture`).
        predict: Match predictor.
        rng: Seeded numpy ``Generator``.

    Returns:
        One :class:`TeamStanding` per team (unranked — call
        :func:`worldcup.simulation.tiebreakers.rank_group`).
    """
    table: dict[str, TeamStanding] = {team: TeamStanding(team=team) for team in teams}
    for fixture in fixtures:
        home, away = _fixture_sides(fixture)
        result = simulate_group_match(home, away, predict, rng)
        _apply_result(table[home], table[away], result)
    return list(table.values())


def select_top_two(ranked: Sequence[TeamStanding]) -> list[TeamStanding]:
    """Return the group winner and runner-up from an already-ranked group.

    Args:
        ranked: Standings ordered best-to-worst (e.g. from ``rank_group``).

    Returns:
        The first two standings.

    Raises:
        ValueError: If fewer than two teams are provided.
    """
    if len(ranked) < 2:
        raise ValueError(f"need at least 2 teams to take the top two, got {len(ranked)}")
    return list(ranked[:2])


def select_best_third_place_teams(
    third_placed: Sequence[TeamStanding],
    rng: np.random.Generator,
    *,
    n: int = 8,
) -> list[TeamStanding]:
    """Pick the ``n`` best third-placed teams across groups.

    Ranks the third-placed teams against each other by the same criteria as a
    group (points → goal difference → goals for → random drawing of lots) and
    returns the best ``n``.

    Args:
        third_placed: The third-placed standing from each group.
        rng: Seeded numpy ``Generator`` (used only for the random fallback).
        n: How many advance (8 in the 2026 format).

    Returns:
        The ``n`` best third-placed standings, best first.

    Raises:
        ValueError: If fewer than ``n`` teams are provided.
    """
    # Imported here to avoid a circular import (tiebreakers imports this module).
    from worldcup.simulation.tiebreakers import rank_group

    if len(third_placed) < n:
        raise ValueError(f"need at least {n} third-placed teams, got {len(third_placed)}")
    return rank_group(list(third_placed), rng)[:n]


# --- internal helpers -------------------------------------------------------


def _sample_scoreline(outcome: int, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a plausible scoreline consistent with ``outcome`` (home perspective)."""
    if outcome == DRAW:
        goals = int(rng.poisson(_DRAW_GOALS_MEAN))
        return goals, goals
    loser = int(rng.poisson(_LOSER_GOALS_MEAN))
    winner = loser + 1 + int(rng.poisson(_EXTRA_MARGIN_MEAN))
    return (winner, loser) if outcome == HOME_WIN else (loser, winner)


def _fixture_sides(fixture: object) -> tuple[str, str]:
    """Extract ``(home, away)`` from a fixture object or a 2-tuple."""
    if hasattr(fixture, "home") and hasattr(fixture, "away"):
        return str(fixture.home), str(fixture.away)  # type: ignore[attr-defined]
    home, away = fixture  # type: ignore[misc]
    return str(home), str(away)


def _apply_result(home: TeamStanding, away: TeamStanding, result: MatchResult) -> None:
    """Update both teams' standings in place from a simulated result."""
    home.played += 1
    away.played += 1
    home.goals_for += result.home_score
    home.goals_against += result.away_score
    away.goals_for += result.away_score
    away.goals_against += result.home_score
    if result.outcome == HOME_WIN:
        home.won += 1
        away.lost += 1
    elif result.outcome == AWAY_WIN:
        away.won += 1
        home.lost += 1
    else:
        home.drawn += 1
        away.drawn += 1
