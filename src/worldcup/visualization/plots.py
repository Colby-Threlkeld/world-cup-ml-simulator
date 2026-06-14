"""Plotting helpers for reports and the Streamlit front-end."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure


def plot_title_probabilities(probs: pd.DataFrame) -> Figure:
    """Render a horizontal bar chart of title probabilities by team.

    TODO(slice 7): implement once :func:`worldcup.simulation.tournament.
    simulate_tournament` produces real output. Sort teams by probability and
    annotate with Monte Carlo confidence intervals.

    Args:
        probs: One row per team with a title-probability column.

    Returns:
        The Matplotlib figure.
    """
    raise NotImplementedError("plot_title_probabilities is implemented in slice 7.")


def plot_reliability_curve(y_true: pd.Series, y_prob: pd.Series) -> Figure:
    """Render a reliability (calibration) diagram.

    TODO(slice 5): bin predicted probabilities and plot observed vs predicted
    frequency against the diagonal, with the Brier score in the title.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities aligned to ``y_true``.

    Returns:
        The Matplotlib figure.
    """
    raise NotImplementedError("plot_reliability_curve is implemented in slice 5.")
