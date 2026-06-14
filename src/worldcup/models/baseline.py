"""Baseline match-outcome models and the Elo rating primitives.

Two things live here:

1. **Elo primitives** (:func:`expected_score`, :func:`update_rating`) — the pure
   rating math the slice-2 Elo baseline is built from.
2. **Baseline forecasters** — a family of dead-simple 3-class probability models
   (``team_a_win`` / ``draw`` / ``team_b_win``) that every "real" model must beat.
   They share a tiny interface (:meth:`BaselineModel.fit` /
   :meth:`BaselineModel.predict_proba`) so the evaluation harness can score them
   interchangeably. Each returns rows that sum to 1 with no exact zeros, so log
   loss stays finite.

The baselines, weakest to strongest expected:
    * :class:`UniformBaseline` — always ``1/3`` each (a forecaster that knows
      nothing).
    * :class:`ClassFrequencyBaseline` — the training-set class priors (Laplace
      smoothed), repeated for every row.
    * :class:`EloLogisticBaseline` — multinomial logistic on ``elo_diff``.
    * :class:`FifaPointsLogisticBaseline` — multinomial logistic on
      ``fifa_points_diff``.
    * :class:`RecentFormLogisticBaseline` — multinomial logistic on the rolling
      form difference features.
    * :class:`WeightedEnsembleBaseline` — weighted average of the others.

Feature-based baselines need their columns present; if a column is missing they
raise :class:`FeatureUnavailableError`, and :func:`available_baselines` skips
them with a clear warning rather than crashing the run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from worldcup.config import RANDOM_SEED
from worldcup.data.clean_data import TARGET_CLASSES
from worldcup.features.rolling_features import DIFFERENCE_FEATURES

logger = logging.getLogger(__name__)

# Probability column order shared by every baseline (and the evaluator).
CLASS_ORDER: tuple[str, ...] = TARGET_CLASSES
_N_CLASSES = len(CLASS_ORDER)
_PROBA_EPS = 1e-12


def expected_score(
    rating_home: float,
    rating_away: float,
    home_advantage: float = 65.0,
) -> float:
    """Compute the Elo win expectancy for the home team.

    Args:
        rating_home: Home team Elo rating.
        rating_away: Away team Elo rating.
        home_advantage: Rating points added to the home side. Use ``0`` at a
            neutral venue (most World Cup matches).

    Returns:
        The home team's expected score in ``(0, 1)`` -- a draw-inclusive win
        expectancy, not yet a calibrated win/draw/loss distribution.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_away - rating_home - home_advantage) / 400.0))


def update_rating(
    rating: float,
    expected: float,
    actual: float,
    k_factor: float = 32.0,
) -> float:
    """Return the post-match Elo rating.

    Args:
        rating: The team's current rating.
        expected: Pre-match expected score from :func:`expected_score`.
        actual: Realized score: ``1.0`` win, ``0.5`` draw, ``0.0`` loss.
        k_factor: Update step size (larger = ratings move faster).

    Returns:
        The updated rating.
    """
    return rating + k_factor * (actual - expected)


# --- baseline forecasters ---------------------------------------------------


class FeatureUnavailableError(ValueError):
    """Raised when a baseline's required feature column(s) are absent from ``X``."""


class BaselineModel:
    """Common interface for the 3-class baseline forecasters.

    Subclasses implement :meth:`fit` and :meth:`predict_proba`. ``predict_proba``
    always returns an ``(n, 3)`` array whose columns are ordered as
    :data:`CLASS_ORDER` and whose rows sum to 1.
    """

    #: Human-readable model name (used as the metrics JSON key).
    name: str = "baseline"
    #: Feature columns this model needs (empty for the prior-only baselines).
    features: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "BaselineModel":
        """Fit the model. Returns ``self`` for chaining."""
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, shape ``(len(X), 3)``, rows summing to 1."""
        raise NotImplementedError

    def missing_features(self, X: pd.DataFrame) -> list[str]:
        """Return this model's required feature columns absent from ``X``."""
        return [f for f in self.features if f not in X.columns]


class UniformBaseline(BaselineModel):
    """Predicts a flat ``1/3`` for every class — a forecaster that knows nothing."""

    name = "uniform_random"

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "UniformBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), _N_CLASSES), 1.0 / _N_CLASSES)


class ClassFrequencyBaseline(BaselineModel):
    """Predicts the training-set class frequencies (Laplace smoothed) for all rows.

    Smoothing keeps every probability strictly positive so a class unseen in
    training cannot send log loss to infinity on the held-out set.
    """

    name = "class_frequency"

    def __init__(self, alpha: float = 1.0) -> None:
        """Args: alpha: Laplace smoothing added to each class count."""
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "ClassFrequencyBaseline":
        counts = pd.Series(np.asarray(y)).value_counts()
        smoothed = np.array([counts.get(c, 0) + self.alpha for c in CLASS_ORDER], dtype=float)
        self.prior_ = smoothed / smoothed.sum()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(self.prior_, (len(X), 1))


class _LogisticBaseline(BaselineModel):
    """Multinomial logistic regression on a fixed list of difference features.

    Wraps a small, deterministic sklearn pipeline: constant-impute missing values
    (a missing *difference* feature means "no signal" → 0), standardize, then fit
    multinomial logistic regression seeded from :data:`RANDOM_SEED`.
    """

    def __init__(self, features: Sequence[str], name: str, *, C: float = 1.0) -> None:
        self.features = tuple(features)
        self.name = name
        self.C = C

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "_LogisticBaseline":
        missing = self.missing_features(X)
        if missing:
            raise FeatureUnavailableError(
                f"{self.name} baseline needs missing feature(s): {missing}"
            )
        self.pipeline_ = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=1000, random_state=RANDOM_SEED),
        )
        self.pipeline_.fit(X[list(self.features)], np.asarray(y))
        self.classes_ = list(self.pipeline_.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        missing = self.missing_features(X)
        if missing:
            raise FeatureUnavailableError(
                f"{self.name} baseline needs missing feature(s): {missing}"
            )
        proba = self.pipeline_.predict_proba(X[list(self.features)])
        return _reorder_to_class_order(proba, self.classes_)


class EloLogisticBaseline(_LogisticBaseline):
    """Logistic baseline on the single ``elo_diff`` feature."""

    def __init__(self, *, C: float = 1.0) -> None:
        super().__init__(("elo_diff",), "elo_logistic", C=C)


class FifaPointsLogisticBaseline(_LogisticBaseline):
    """Logistic baseline on the single ``fifa_points_diff`` feature."""

    def __init__(self, *, C: float = 1.0) -> None:
        super().__init__(("fifa_points_diff",), "fifa_points_logistic", C=C)


class RecentFormLogisticBaseline(_LogisticBaseline):
    """Logistic baseline on the rolling-form difference features."""

    def __init__(self, *, C: float = 1.0) -> None:
        super().__init__(DIFFERENCE_FEATURES, "recent_form_logistic", C=C)


class WeightedEnsembleBaseline(BaselineModel):
    """Weighted average of several baselines' probabilities (renormalized)."""

    name = "weighted_ensemble"

    def __init__(
        self,
        models: Iterable[BaselineModel],
        weights: Sequence[float] | None = None,
    ) -> None:
        """Args:
            models: The member baselines to average. Fitted by :meth:`fit`.
            weights: Per-model weights (default: equal). Normalized to sum to 1.
        """
        self.models = list(models)
        if not self.models:
            raise ValueError("WeightedEnsembleBaseline needs at least one member model")
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "WeightedEnsembleBaseline":
        for model in self.models:
            model.fit(X, y)
        raw = (
            np.ones(len(self.models))
            if self.weights is None
            else np.asarray(self.weights, dtype=float)
        )
        if raw.shape != (len(self.models),):
            raise ValueError("weights length must match number of models")
        self.weights_ = raw / raw.sum()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        stacked = np.stack([m.predict_proba(X) for m in self.models])  # (k, n, 3)
        blended = np.tensordot(self.weights_, stacked, axes=([0], [0]))  # (n, 3)
        return _clip_normalize(blended)


def available_baselines(X: pd.DataFrame) -> dict[str, BaselineModel]:
    """Build every baseline whose required features are present in ``X``.

    The two prior-only baselines are always included. Each feature baseline is
    added only if its columns exist, otherwise a clear warning is logged and it
    is skipped (the run never crashes on a missing feature). The weighted ensemble
    averages the class-frequency prior with whichever feature baselines survived.

    Args:
        X: A representative feature frame (e.g. the training split) used to decide
            which features are available.

    Returns:
        Ordered mapping of model name to a fresh, unfitted model instance.
    """
    models: dict[str, BaselineModel] = {
        UniformBaseline.name: UniformBaseline(),
        ClassFrequencyBaseline.name: ClassFrequencyBaseline(),
    }

    feature_models: tuple[BaselineModel, ...] = (
        EloLogisticBaseline(),
        FifaPointsLogisticBaseline(),
        RecentFormLogisticBaseline(),
    )
    survivors: list[BaselineModel] = []
    for model in feature_models:
        missing = model.missing_features(X)
        if missing:
            logger.warning(
                "Skipping %s baseline: missing feature(s) %s", model.name, missing
            )
            continue
        models[model.name] = model
        survivors.append(model)

    if survivors:
        members: list[BaselineModel] = [ClassFrequencyBaseline(), *_fresh_like(survivors)]
        models[WeightedEnsembleBaseline.name] = WeightedEnsembleBaseline(members)
    else:
        logger.warning(
            "Skipping %s baseline: no feature baselines available",
            WeightedEnsembleBaseline.name,
        )
    return models


# --- internal helpers -------------------------------------------------------


def _fresh_like(models: Iterable[BaselineModel]) -> list[BaselineModel]:
    """Return new, unfitted instances of the given baselines (for the ensemble)."""
    return [type(model)() for model in models]


def _reorder_to_class_order(proba: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Map an estimator's class-ordered probabilities to :data:`CLASS_ORDER`.

    Columns for classes the estimator never saw are filled with 0 before the
    final clip/normalize, so the output always has all three classes in order.
    """
    index = {c: i for i, c in enumerate(classes)}
    out = np.zeros((proba.shape[0], _N_CLASSES))
    for j, cls in enumerate(CLASS_ORDER):
        if cls in index:
            out[:, j] = proba[:, index[cls]]
    return _clip_normalize(out)


def _clip_normalize(proba: np.ndarray, eps: float = _PROBA_EPS) -> np.ndarray:
    """Clip to ``[eps, ∞)`` and renormalize rows to sum to 1 (keeps log loss finite)."""
    clipped = np.clip(proba, eps, None)
    return clipped / clipped.sum(axis=1, keepdims=True)
