"""Plotting helpers for reports and the Streamlit front-end.

Figures are built on bare :class:`matplotlib.figure.Figure` objects with an
explicit Agg canvas (no global ``pyplot`` state), so they render headless on a CPU
VM and in tests without a display.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from worldcup.models.evaluate import CLASS_ORDER, reliability_curve


def plot_calibration_curve(
    y_true: Sequence[str] | pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    classes: Sequence[str] = CLASS_ORDER,
    n_bins: int = 10,
) -> Figure:
    """Render a one-vs-rest reliability diagram for the 3 outcome classes.

    Each class gets a curve of observed frequency vs mean predicted probability;
    the dashed diagonal is perfect calibration. Curves above the diagonal are
    under-confident, below are over-confident.

    Args:
        y_true: True class labels.
        proba: Predicted probabilities, shape ``(n, len(classes))``.
        classes: Class order matching ``proba`` columns.
        n_bins: Number of probability bins.

    Returns:
        The Matplotlib figure.
    """
    curves = reliability_curve(y_true, proba, classes=classes, n_bins=n_bins)
    fig = Figure(figsize=(6, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    for cls in classes:
        points = curves[cls]
        if not points.empty:
            ax.plot(
                points["mean_predicted"],
                points["observed_frequency"],
                marker="o",
                label=cls,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve (one-vs-rest per class)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion: pd.DataFrame) -> Figure:
    """Render a labeled confusion-matrix heatmap (rows = actual, cols = predicted).

    Args:
        confusion: Square frame from
            :func:`worldcup.models.evaluate.confusion_matrix_frame`.

    Returns:
        The Matplotlib figure.
    """
    labels = list(confusion.columns)
    values = confusion.to_numpy()
    fig = Figure(figsize=(6, 5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()

    im = ax.imshow(values, cmap="Blues")
    fig.colorbar(im, ax=ax, label="match count")

    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")

    # Annotate each cell, switching text color for contrast on dark cells.
    threshold = values.max() / 2.0 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:d}",
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
            )
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path | str, *, dpi: int = 110) -> Path:
    """Save a figure to ``path`` (creating parent dirs); returns the path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    return out_path


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
