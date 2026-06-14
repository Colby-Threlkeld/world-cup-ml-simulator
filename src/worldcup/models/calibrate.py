"""Calibrate model probabilities so they mean what they say.

A model can rank matches well yet still be over- or under-confident. Calibration
(isotonic or Platt/sigmoid scaling) makes a stated "70%" actually happen ~70% of
the time -- essential before the probabilities feed the Monte Carlo simulation.

Leakage rule: the base model is fit on the **train** window, the calibrator is fit
on a **separate validation** window, and both are scored on the untouched **test**
window. Calibrating on the rows the model trained on would understate
over-confidence.

We deliberately avoid :class:`sklearn.calibration.CalibratedClassifierCV` here: on
a prefit estimator it still cross-validates the (small) calibration set, which is
fragile on tiny or class-imbalanced validation windows. :class:`ProbabilityCalibrator`
fits a per-class 1-D map (isotonic or sigmoid) directly on the held-out validation
predictions instead — simple, deterministic, and robust on little data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from worldcup.data.clean_data import TARGET_CLASSES

# Calibration methods supported, plus our "none" pass-through.
CALIBRATION_METHODS: tuple[str, ...] = ("isotonic", "sigmoid", "none")
_PROBA_EPS = 1e-12


def fit_calibrated_model(
    estimator: ClassifierMixin,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    *,
    method: str = "isotonic",
    classes: tuple[str, ...] = TARGET_CLASSES,
) -> ClassifierMixin:
    """Wrap a **prefit** estimator in a calibrator fit on the validation window.

    Args:
        estimator: An already-fitted classifier (``predict_proba`` + ``classes_``).
        X_val: Validation features (disjoint from training).
        y_val: Validation labels.
        method: ``"isotonic"`` (flexible), ``"sigmoid"`` (Platt, data-light), or
            ``"none"`` to return ``estimator`` unchanged.
        classes: Output class order.

    Returns:
        A fitted :class:`ProbabilityCalibrator` (or the original estimator if
        ``method="none"``).

    Raises:
        ValueError: If ``method`` is not one of :data:`CALIBRATION_METHODS`.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"method must be one of {CALIBRATION_METHODS}, got {method!r}")
    if method == "none":
        return estimator
    return ProbabilityCalibrator(estimator, method=method, classes=list(classes)).fit(
        X_val, y_val
    )


class ProbabilityCalibrator(ClassifierMixin, BaseEstimator):
    """Per-class prefit probability calibrator (isotonic or sigmoid/Platt).

    Reorders the wrapped estimator's probabilities to ``classes``, maps each
    class column through a fitted 1-D calibrator, then renormalizes the rows to a
    valid distribution. Exposes ``classes_`` / ``predict_proba`` / ``predict`` so
    it is a drop-in classifier for the evaluator and is joblib-picklable.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        *,
        method: str = "isotonic",
        classes: list[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.classes = classes

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "ProbabilityCalibrator":
        order = list(self.classes) if self.classes is not None else list(TARGET_CLASSES)
        self.classes_ = np.asarray(order)
        proba = _proba_in_order(self.estimator, X, order)
        y_arr = np.asarray(y)
        self.calibrators_ = [
            _Calibrator1D(self.method).fit(proba[:, k], (y_arr == cls).astype(float))
            for k, cls in enumerate(order)
        ]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = _proba_in_order(self.estimator, X, list(self.classes_))
        out = np.column_stack(
            [cal.predict(proba[:, k]) for k, cal in enumerate(self.calibrators_)]
        )
        out = np.clip(out, _PROBA_EPS, None)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    classes: list[str] | tuple[str, ...],
    *,
    n_bins: int = 10,
) -> float:
    """Top-label expected calibration error (ECE); lower is better, 0 is perfect.

    Bins predictions by their winning-class confidence and measures the average
    gap between confidence and empirical accuracy, weighted by bin population.

    Args:
        y_true: True labels.
        proba: Predicted probabilities, shape ``(n, len(classes))``.
        classes: Class order matching ``proba`` columns.
        n_bins: Number of equal-width confidence bins in ``[0, 1]``.

    Returns:
        The ECE as a float in ``[0, 1]``.
    """
    labels = np.asarray(classes)
    y = np.asarray(y_true)
    confidence = proba.max(axis=1)
    predicted = labels[proba.argmax(axis=1)]
    correct = (predicted == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Last bin is closed on the right so confidence == 1.0 is counted.
        in_bin = (confidence > lo) & (confidence <= hi) if hi < 1.0 else (confidence > lo)
        count = int(in_bin.sum())
        if count:
            ece += (count / n) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


# --- internal helpers -------------------------------------------------------


class _Calibrator1D:
    """A 1-D probability→calibrated-probability map (isotonic or sigmoid/Platt).

    Falls back to a constant (the observed positive rate) when the target has a
    single class in the calibration window, so degenerate columns never crash.
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method

    def fit(self, scores: np.ndarray, target: np.ndarray) -> "_Calibrator1D":
        scores = np.asarray(scores, dtype=float)
        target = np.asarray(target, dtype=float)
        if np.unique(target).size < 2:
            self.constant_: float | None = float(target.mean())
            self.model_ = None
            return self
        self.constant_ = None
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(scores, target)
        else:  # sigmoid / Platt scaling on the single probability score
            self.model_ = LogisticRegression().fit(scores.reshape(-1, 1), target)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.constant_ is not None:
            return np.full(scores.shape[0], self.constant_)
        if self.method == "isotonic":
            return np.asarray(self.model_.predict(scores))
        return self.model_.predict_proba(scores.reshape(-1, 1))[:, 1]


def _proba_in_order(estimator: ClassifierMixin, X: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Return ``estimator.predict_proba(X)`` with columns reordered to ``classes``."""
    proba = np.asarray(estimator.predict_proba(X))
    est_classes = list(estimator.classes_)
    if est_classes == classes:
        return proba
    index = {c: i for i, c in enumerate(est_classes)}
    out = np.zeros((proba.shape[0], len(classes)))
    for j, cls in enumerate(classes):
        if cls in index:
            out[:, j] = proba[:, index[cls]]
    return out
