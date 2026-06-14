"""Tournament configuration + Monte Carlo simulation of the 2026 World Cup.

The 2026 format (48 teams, 12 groups of 4, top-two plus eight best thirds into a
round of 32) lives in ``configs/tournament_2026.yaml`` as *data*, not hardcoded
control flow. This module loads that YAML into typed :class:`TournamentConfig`
objects and validates the structural invariants up front, so the simulator (slice
6) can trust its input.

Honesty note: the official group draw and the exact Round-of-32 pairings are
treated as **configurable placeholders** (``draw_status: placeholder``). The
round-robin group fixtures are structurally certain regardless of the draw and so
are fully populated; the knockout pairings are left for a configurable mapping
rather than asserting an unverified bracket.
"""

from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from worldcup.config import CONFIGS_DIR, RANDOM_SEED, load_yaml
from worldcup.simulation.group_stage import (
    PredictFn,
    build_group_table,
    select_best_third_place_teams,
)
from worldcup.simulation.knockout import CHAMPION_KEY, simulate_bracket
from worldcup.simulation.tiebreakers import rank_group

logger = logging.getLogger(__name__)

DEFAULT_TOURNAMENT_CONFIG_PATH = CONFIGS_DIR / "tournament_2026.yaml"


class TournamentConfigError(ValueError):
    """Raised when the tournament configuration is structurally invalid."""


@dataclass(frozen=True)
class GroupFixture:
    """A single group-stage match between two slots in the same group."""

    group: str
    matchday: int
    home: str
    away: str


@dataclass(frozen=True)
class TournamentConfig:
    """Typed, validated view of the 2026 World Cup format configuration.

    Team identifiers may be real nations or placeholder slots (e.g. ``A1``) when
    ``draw_status == "placeholder"``; the structure is identical either way.
    """

    total_teams: int
    num_groups: int
    teams_per_group: int
    advance_per_group: int
    best_third_placed_advance: int
    knockout_rounds: tuple[str, ...]
    hosts: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]
    fixtures: tuple[GroupFixture, ...]
    tiebreakers: tuple[str, ...]
    knockout_bracket: dict[str, Any] = field(default_factory=dict)
    draw_status: str = "placeholder"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TournamentConfig":
        """Build a config from a parsed YAML mapping (no validation performed).

        Raises:
            TournamentConfigError: If a required section is missing or malformed.
        """
        try:
            fmt = data["format"]
            groups = {
                str(name): tuple(str(t) for t in teams)
                for name, teams in (data.get("groups") or {}).items()
            }
            fixtures = tuple(_parse_fixture(item) for item in (data.get("fixtures") or []))
            return cls(
                total_teams=int(fmt["total_teams"]),
                num_groups=int(fmt["num_groups"]),
                teams_per_group=int(fmt["teams_per_group"]),
                advance_per_group=int(fmt["advance_per_group"]),
                best_third_placed_advance=int(fmt["best_third_placed_advance"]),
                knockout_rounds=tuple(str(r) for r in fmt.get("knockout_rounds", [])),
                hosts=tuple(str(h) for h in (data.get("hosts") or [])),
                groups=groups,
                fixtures=fixtures,
                tiebreakers=tuple(str(t) for t in (data.get("tiebreakers") or [])),
                knockout_bracket=dict(data.get("knockout_bracket") or {}),
                draw_status=str(data.get("draw_status", "placeholder")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TournamentConfigError(f"malformed tournament config: {exc}") from exc

    def teams(self) -> list[str]:
        """Return every team slot across all groups, in group order."""
        return [team for group in self.groups.values() for team in group]

    def group_of(self, team: str) -> str | None:
        """Return the group name containing ``team``, or ``None`` if not found."""
        for name, members in self.groups.items():
            if team in members:
                return name
        return None

    def has_full_fixture_list(self) -> bool:
        """True iff fixtures are present (so per-team fixture counts can be checked)."""
        return len(self.fixtures) > 0


def load_tournament_config(
    path: Path | str | None = None, *, validate: bool = True
) -> TournamentConfig:
    """Load and (by default) validate the tournament configuration.

    Args:
        path: YAML path. Defaults to ``configs/tournament_2026.yaml``.
        validate: If ``True``, run :func:`validate_tournament_config` and raise on
            any structural problem.

    Returns:
        The parsed :class:`TournamentConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TournamentConfigError: If parsing fails, or validation fails when enabled.
    """
    config_path = Path(path) if path is not None else DEFAULT_TOURNAMENT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"tournament config not found: {config_path}")
    config = TournamentConfig.from_dict(load_yaml(config_path))
    if validate:
        validate_tournament_config(config)
    return config


def check_tournament_config(config: TournamentConfig) -> list[str]:
    """Return a list of structural problems (empty if the config is valid).

    Non-raising counterpart to :func:`validate_tournament_config`; accumulates
    every issue so they can be fixed in one pass. Validates the certain structural
    facts only — the placeholder knockout pairings are intentionally not asserted.
    """
    errors: list[str] = []

    # 1. Exactly 12 groups (declared count and actual count agree).
    if config.num_groups != 12:
        errors.append(f"num_groups is {config.num_groups}, expected 12")
    if len(config.groups) != config.num_groups:
        errors.append(
            f"{len(config.groups)} group(s) defined but num_groups is {config.num_groups}"
        )

    # 2. Each group has exactly teams_per_group teams.
    for name, members in config.groups.items():
        if len(members) != config.teams_per_group:
            errors.append(
                f"group {name} has {len(members)} team(s), expected {config.teams_per_group}"
            )

    # 3 & 4. Correct total of unique teams, with no duplicates anywhere.
    all_teams = config.teams()
    duplicates = sorted({t for t, n in Counter(all_teams).items() if n > 1})
    if duplicates:
        errors.append(f"duplicate team(s) across groups: {duplicates}")
    unique_teams = set(all_teams)
    if len(unique_teams) != config.total_teams:
        errors.append(
            f"{len(unique_teams)} unique team(s), expected total_teams={config.total_teams}"
        )

    # Sanity: the advancement arithmetic must yield a 32-team knockout.
    qualifiers = config.advance_per_group * config.num_groups + config.best_third_placed_advance
    if qualifiers != 32:
        errors.append(
            f"advancement yields {qualifiers} qualifiers, expected 32 "
            f"({config.advance_per_group}x{config.num_groups} + {config.best_third_placed_advance})"
        )

    # 5 & 6. Fixtures reference valid teams; each team plays exactly 3 (if listed).
    errors.extend(_check_fixtures(config, unique_teams))

    return errors


def validate_tournament_config(config: TournamentConfig) -> None:
    """Validate the tournament configuration, raising on any failure.

    Raises:
        TournamentConfigError: If any structural check fails (message lists all).
    """
    errors = check_tournament_config(config)
    if errors:
        header = f"{len(errors)} tournament-config error(s):"
        raise TournamentConfigError("\n".join([header, *(f"  - {e}" for e in errors)]))


# --- internal helpers -------------------------------------------------------


def _parse_fixture(item: dict[str, Any]) -> GroupFixture:
    """Parse one fixture mapping into a :class:`GroupFixture`."""
    return GroupFixture(
        group=str(item["group"]),
        matchday=int(item["matchday"]),
        home=str(item["home"]),
        away=str(item["away"]),
    )


def _check_fixtures(config: TournamentConfig, valid_teams: set[str]) -> list[str]:
    """Validate fixture team references and per-team fixture counts."""
    errors: list[str] = []
    if not config.has_full_fixture_list():
        return errors  # nothing to check until a fixture list is provided

    appearances: Counter[str] = Counter()
    for fx in config.fixtures:
        for side in (fx.home, fx.away):
            if side not in valid_teams:
                errors.append(f"fixture references unknown team {side!r} (group {fx.group})")
            else:
                appearances[side] += 1
        if fx.home == fx.away:
            errors.append(f"fixture has identical home/away team {fx.home!r}")
        # Both sides should belong to the fixture's stated group.
        group_members = config.groups.get(fx.group, ())
        for side in (fx.home, fx.away):
            if side in valid_teams and side not in group_members:
                errors.append(f"fixture team {side!r} is not in its stated group {fx.group}")

    expected = config.teams_per_group - 1  # round robin: each team plays the other 3
    wrong = {t: n for t, n in appearances.items() if n != expected}
    if wrong:
        sample = dict(sorted(wrong.items())[:5])
        errors.append(
            f"{len(wrong)} team(s) do not have exactly {expected} group fixtures, e.g. {sample}"
        )
    return errors


def build_knockout_seeding(
    config: TournamentConfig,
    group_winners: Sequence[str],
    runners_up: Sequence[str],
    best_thirds: Sequence[str],
) -> list[str]:
    """Build the ordered Round-of-32 seeding from the group qualifiers.

    Two paths, in priority order:

    1. **Config-driven (preferred):** if ``knockout_bracket.round_of_32`` lists
       pairings, each match's ``home``/``away`` slot tokens (``1A`` = winner of
       group A, ``2A`` = runner-up, ``3-n`` = the n-th best third) are resolved to
       team names and concatenated in match order. This is how the *official*
       bracket should be encoded once known.
    2. **Documented placeholder:** the shipped config leaves ``round_of_32`` empty
       because the official third-place→slot lookup table is not implemented. As a
       fallback we seed by relative strength — winners, then runners-up, then
       thirds — and pair strongest-vs-weakest (``seed i`` vs ``seed 31-i``). This
       produces a *valid* 32-team bracket for the simulation but is **not** the
       official pairing; replace it by filling in ``round_of_32``.

    Args:
        config: The tournament config (provides the optional pairings).
        group_winners: The 12 group winners (1A..1L), in group order.
        runners_up: The 12 runners-up (2A..2L), in group order.
        best_thirds: The 8 best third-placed teams, best first.

    Returns:
        32 team names in bracket order (adjacent pairs are first-round matches).

    Raises:
        TournamentConfigError: If the qualifier counts are wrong, a configured
            slot token is unknown, or the result is not 32 unique teams.
    """
    expected = (config.num_groups, config.num_groups, config.best_third_placed_advance)
    actual = (len(group_winners), len(runners_up), len(best_thirds))
    if actual != expected:
        raise TournamentConfigError(
            f"qualifier counts {actual} do not match expected {expected} "
            "(group_winners, runners_up, best_thirds)"
        )

    pairings = config.knockout_bracket.get("round_of_32") or []
    if pairings:
        seeding = _seed_from_config_pairings(config, pairings, group_winners, runners_up, best_thirds)
    else:
        seeding = _seed_placeholder(group_winners, runners_up, best_thirds)

    if len(set(seeding)) != len(seeding) or len(seeding) != 32:
        raise TournamentConfigError(
            f"knockout seeding must be 32 unique teams, got {len(seeding)} "
            f"({len(set(seeding))} unique)"
        )
    return seeding


def _slot_map(
    config: TournamentConfig,
    group_winners: Sequence[str],
    runners_up: Sequence[str],
    best_thirds: Sequence[str],
) -> dict[str, str]:
    """Map slot tokens (1A, 2A, 3-n) to team names."""
    groups = list(config.groups.keys())
    slots: dict[str, str] = {}
    for group, winner, runner in zip(groups, group_winners, runners_up, strict=True):
        slots[f"1{group}"] = winner
        slots[f"2{group}"] = runner
    for i, team in enumerate(best_thirds, start=1):
        slots[f"3-{i}"] = team
    return slots


def _seed_from_config_pairings(
    config: TournamentConfig,
    pairings: list[dict[str, Any]],
    group_winners: Sequence[str],
    runners_up: Sequence[str],
    best_thirds: Sequence[str],
) -> list[str]:
    """Resolve configured round-of-32 slot pairings into an ordered team list."""
    slots = _slot_map(config, group_winners, runners_up, best_thirds)
    seeding: list[str] = []
    for match in pairings:
        for key in ("home", "away"):
            token = str(match[key])
            if token not in slots:
                raise TournamentConfigError(f"unknown knockout slot token {token!r}")
            seeding.append(slots[token])
    return seeding


def _seed_placeholder(
    group_winners: Sequence[str],
    runners_up: Sequence[str],
    best_thirds: Sequence[str],
) -> list[str]:
    """Strength-ordered placeholder seeding (winners > runners > thirds)."""
    by_strength = [*group_winners, *runners_up, *best_thirds]
    # Standard seeding: pair strongest with weakest, working inwards.
    ordered: list[str] = []
    lo, hi = 0, len(by_strength) - 1
    while lo < hi:
        ordered.extend([by_strength[lo], by_strength[hi]])
        lo += 1
        hi -= 1
    return ordered


# --- Monte Carlo simulation -------------------------------------------------

WIN_GROUP_KEY = "win_group"

# Tally key -> output probability column. The knockout keys match
# worldcup.simulation.knockout.ADVANCEMENT_ROUNDS; ``win_group`` is tallied here.
_TALLY_TO_COLUMN: dict[str, str] = {
    WIN_GROUP_KEY: "win_group_probability",
    "reach_round_32": "reach_round_32_probability",
    "reach_round_16": "reach_round_16_probability",
    "reach_quarterfinal": "reach_quarterfinal_probability",
    "reach_semifinal": "reach_semifinal_probability",
    "reach_final": "reach_final_probability",
    CHAMPION_KEY: "win_world_cup_probability",
}

#: Output probability columns, outermost stage first.
PROBABILITY_COLUMNS: tuple[str, ...] = tuple(_TALLY_TO_COLUMN.values())


def uniform_predict_fn() -> PredictFn:
    """A predictor that knows nothing: every match is 1/3 win / 1/3 draw / 1/3 win.

    The honest default for the placeholder draw — all slots are interchangeable, so
    the result reflects bracket structure only, not a team-specific forecast.
    """
    third = 1.0 / 3.0
    return lambda team_a, team_b: (third, third, third)


def strength_predict_fn(
    strengths: Mapping[str, float],
    *,
    draw_rate: float = 0.22,
    scale: float = 100.0,
) -> PredictFn:
    """Build a :data:`PredictFn` from per-team strength ratings (e.g. Elo).

    The decisive-result split is a logistic on the strength difference; the draw
    probability is a fixed ``draw_rate``. Teams absent from ``strengths`` get a
    neutral 0, so unknown/placeholder slots are treated as average.

    Args:
        strengths: Team name -> strength rating (higher = stronger).
        draw_rate: Fixed draw probability (international base rate ~0.22).
        scale: Rating points for a 10x odds swing (Elo-like; larger = flatter).

    Returns:
        ``predict(team_a, team_b) -> (p_a_win, p_draw, p_b_win)``.

    Raises:
        ValueError: If ``draw_rate`` is not in ``[0, 1)``.
    """
    if not 0.0 <= draw_rate < 1.0:
        raise ValueError(f"draw_rate must be in [0, 1), got {draw_rate}")
    decisive = 1.0 - draw_rate

    def predict(team_a: str, team_b: str) -> tuple[float, float, float]:
        diff = strengths.get(team_a, 0.0) - strengths.get(team_b, 0.0)
        # Clamp the exponent so an extreme strength gap saturates to ~0/1 instead
        # of overflowing 10 ** huge.
        exponent = min(50.0, max(-50.0, -diff / scale))
        p_a_decisive = 1.0 / (1.0 + 10.0**exponent)
        return (decisive * p_a_decisive, draw_rate, decisive * (1.0 - p_a_decisive))

    return predict


def simulate_tournament(
    config: TournamentConfig,
    predict: PredictFn,
    *,
    n_simulations: int = 10_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run the full 2026 World Cup Monte Carlo and estimate per-team probabilities.

    Each simulation plays all 12 groups (round robin -> standings -> top two plus
    the eight best thirds), seeds the round of 32 (see
    :func:`build_knockout_seeding`), and plays the bracket to a champion. Per-team
    advancement is accumulated as **counts** — no per-run state is stored — so
    memory stays tiny on a small CPU VM. All randomness flows through one seeded
    numpy generator, so a seed fully reproduces a run.

    The model is **not** retrained here: ``predict`` is a prediction function
    (e.g. wrapping a cached model or rating table) and is memoized internally, so a
    costly model is queried at most once per unique pairing.

    Args:
        config: Validated tournament configuration.
        predict: ``(team_a, team_b) -> (p_a_win, p_draw, p_b_win)``.
        n_simulations: Independent tournaments to run (1,000 quick / 10,000+ full).
        seed: Seed for the single numpy ``Generator``.

    Returns:
        One row per team with columns ``team`` + :data:`PROBABILITY_COLUMNS`,
        sorted by title probability (descending).

    Raises:
        ValueError: If ``n_simulations`` is not positive.
    """
    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}")

    tally: dict[str, dict[str, int]] = {
        team: dict.fromkeys(_TALLY_TO_COLUMN, 0) for team in config.teams()
    }
    cached_predict = _memoize_predict(predict)
    rng = np.random.default_rng(seed)

    fixtures_by_group: dict[str, list[GroupFixture]] = {}
    for fixture in config.fixtures:
        fixtures_by_group.setdefault(fixture.group, []).append(fixture)

    start = time.perf_counter()
    for _ in range(n_simulations):
        _simulate_one_tournament(config, fixtures_by_group, cached_predict, rng, tally)
    logger.info("Simulated %d tournaments in %.2fs", n_simulations, time.perf_counter() - start)

    return _tally_to_probabilities(tally, n_simulations)


def summarize_simulation(
    probabilities: pd.DataFrame,
    *,
    n_simulations: int,
    seed: int,
    runtime_seconds: float,
    draw_status: str,
    top_n: int = 10,
) -> dict[str, Any]:
    """Build a JSON-serializable summary of a simulation run.

    Args:
        probabilities: Output of :func:`simulate_tournament`.
        n_simulations: Number of simulations run.
        seed: Seed used.
        runtime_seconds: Wall-clock runtime.
        draw_status: ``config.draw_status`` — recorded so a placeholder run is never
            mistaken for a real forecast.
        top_n: How many title contenders to list.

    Returns:
        A dictionary suitable for ``json.dump``.
    """
    top = probabilities.nlargest(top_n, "win_world_cup_probability")
    return {
        "n_simulations": int(n_simulations),
        "seed": int(seed),
        "runtime_seconds": round(float(runtime_seconds), 3),
        "n_teams": int(len(probabilities)),
        "draw_status": draw_status,
        "title_probability_total": round(float(probabilities["win_world_cup_probability"].sum()), 6),
        "top_title_contenders": [
            {"team": str(team), "win_world_cup_probability": round(float(prob), 4)}
            for team, prob in zip(top["team"], top["win_world_cup_probability"], strict=True)
        ],
    }


# --- internal simulation helpers --------------------------------------------


def _simulate_one_tournament(
    config: TournamentConfig,
    fixtures_by_group: Mapping[str, Sequence[GroupFixture]],
    predict: PredictFn,
    rng: np.random.Generator,
    tally: dict[str, dict[str, int]],
) -> None:
    """Play one full tournament and update ``tally`` in place."""
    group_winners: list[str] = []
    runners_up: list[str] = []
    third_placed = []
    for name, group_teams in config.groups.items():
        standings = build_group_table(group_teams, fixtures_by_group.get(name, ()), predict, rng)
        ranked = rank_group(standings, rng)
        group_winners.append(ranked[0].team)
        runners_up.append(ranked[1].team)
        third_placed.append(ranked[2])
        tally[ranked[0].team][WIN_GROUP_KEY] += 1

    best_thirds = [
        standing.team
        for standing in select_best_third_place_teams(
            third_placed, rng, n=config.best_third_placed_advance
        )
    ]
    seeding = build_knockout_seeding(config, group_winners, runners_up, best_thirds)
    result = simulate_bracket(seeding, predict, rng)

    for key, reached in result.advancement.items():
        if key == CHAMPION_KEY:
            continue
        for team in reached:
            tally[team][key] += 1
    tally[result.champion][CHAMPION_KEY] += 1


def _memoize_predict(predict: PredictFn) -> PredictFn:
    """Wrap a predictor in an unbounded cache (probabilities are fixed per pairing)."""

    @functools.lru_cache(maxsize=None)
    def cached(team_a: str, team_b: str) -> tuple[float, ...]:
        return tuple(float(p) for p in predict(team_a, team_b))

    return cached


def _tally_to_probabilities(tally: Mapping[str, Mapping[str, int]], n: int) -> pd.DataFrame:
    """Convert per-team counts to probabilities, sorted by title probability."""
    rows = [
        {"team": team, **{col: counts[key] / n for key, col in _TALLY_TO_COLUMN.items()}}
        for team, counts in tally.items()
    ]
    return (
        pd.DataFrame(rows, columns=["team", *PROBABILITY_COLUMNS])
        .sort_values(["win_world_cup_probability", "team"], ascending=[False, True])
        .reset_index(drop=True)
    )
