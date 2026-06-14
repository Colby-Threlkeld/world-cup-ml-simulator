"""Calibrate model probabilities so they mean what they say.

A model can rank matches well yet still be over- or under-confident. Calibration
(isotonic or Platt scaling), checked with reliability diagrams, makes a stated
"70%" actually happen ~70% of the time -- essential before the probabilities
feed the Monte Carlo simulation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def calibrate_probabilities(
    raw_probs: pd.DataFrame,
    labels: pd.Series,
    method: str = "isotonic",
) -> Any:
    """Fit a calibration map from raw model probabilities to calibrated ones.

    TODO(slice 5): fit isotonic or sigmoid (Platt) calibration on a held-out
    window and return a transformer. Report reliability curves and expected
    calibration error before/after.

    Args:
        raw_probs: Uncalibrated per-class probabilities from the model.
        labels: True outcomes aligned to ``raw_probs``.
        method: ``"isotonic"`` or ``"sigmoid"``.

    Returns:
        A fitted calibrator that maps raw probabilities to calibrated ones.
    """
    raise NotImplementedError("calibrate_probabilities is implemented in slice 5.")
