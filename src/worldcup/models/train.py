"""Train the match outcome model (Poisson goals model -- slice 4)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def train_model(features: pd.DataFrame, config: dict[str, Any]) -> Any:
    """Fit the match model on a leakage-safe feature matrix.

    TODO(slice 4): fit the bivariate/independent Poisson goals model (predict
    expected goals for each side) using only a temporal training window, and
    return the fitted estimator. Must be benchmarked against the slice-2 Elo
    baseline before it is accepted.

    Args:
        features: Feature matrix from :mod:`worldcup.features.build_features`.
        config: Model hyperparameters (see ``configs/model_config.yaml``).

    Returns:
        The fitted model object.
    """
    raise NotImplementedError("train_model is implemented in slice 4.")
