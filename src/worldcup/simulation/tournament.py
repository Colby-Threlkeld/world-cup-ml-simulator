"""Monte Carlo tournament simulation -> winner probabilities (slice 6)."""

from __future__ import annotations

import pandas as pd


def simulate_tournament(n_simulations: int = 10_000) -> pd.DataFrame:
    """Run the full 2026 World Cup Monte Carlo.

    TODO(slice 6): for each of ``n_simulations`` runs -- draw group results from
    the match model, compute standings and qualifiers (top two per group plus
    the eight best third-placed teams), then play the round of 32 through the
    final via :func:`worldcup.simulation.knockout.resolve_knockout`. Aggregate
    per-team advancement and title probabilities across all runs.

    Args:
        n_simulations: Number of independent tournaments to simulate.

    Returns:
        One row per team with advancement and title probabilities.
    """
    raise NotImplementedError("simulate_tournament is implemented in slice 6.")
