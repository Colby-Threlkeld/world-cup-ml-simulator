"""Evaluate match-probability models with proper scoring rules.

Probabilistic forecasts are judged with proper scoring rules (Brier score, log
loss) on a temporally held-out set -- never random K-fold, which would leak
future information into the past.

Provides the temporal train/validation/test split, the three headline metrics
(log loss, multiclass Brier, accuracy), and a small JSON writer, all keyed to the
``team_a_win`` / ``draw`` / ``team_b_win`` class order.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from worldcup.data.clean_data import TARGET_CLASSES

_LOG_LOSS_EPS = 1e-15

#: Class column order shared with the baseline models.
CLASS_ORDER: tuple[str, ...] = TARGET_CLASSES


def temporal_train_val_test_split(
    features: pd.DataFrame,
    *,
    date_col: str = "date",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a feature frame into train/validation/test windows *by date*.

    Rows are ordered by ``date_col``; the earliest ``1 - val - test`` fraction is
    training, the next ``val_fraction`` is validation, and the most recent
    ``test_fraction`` is test. This is a strict temporal holdout — never a random
    split — so a model is always scored on matches later than those it trained on.

    Args:
        features: Feature matrix (must contain ``date_col``).
        date_col: The kickoff-date column to order by.
        val_fraction: Fraction of rows (most recent-but-one block) for validation.
        test_fraction: Fraction of rows (most recent block) for test.

    Returns:
        ``(train, val, test)`` frames, each a copy with a reset index.

    Raises:
        ValueError: If ``date_col`` is absent or the fractions are not in
            ``(0, 1)`` with a positive training remainder.
    """
    if date_col not in features.columns:
        raise ValueError(f"features missing date column '{date_col}'")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must leave room for training")

    ordered = features.sort_values(date_col, kind="stable").reset_index(drop=True)
    n = len(ordered)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"no training rows left after split (n={n})")

    train = ordered.iloc[:n_train].reset_index(drop=True)
    val = ordered.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = ordered.iloc[n_train + n_val :].reset_index(drop=True)
    return train, val, test


def evaluate_proba(
    y_true: Sequence[str] | pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    classes: Sequence[str] = CLASS_ORDER,
) -> dict[str, float]:
    """Score 3-class probability forecasts with log loss, Brier, and accuracy.

    Args:
        y_true: True class labels (each one of ``classes``).
        proba: Predicted probabilities, shape ``(len(y_true), len(classes))``,
            columns ordered as ``classes`` and rows summing to 1.
        classes: The class order matching ``proba``'s columns.

    Returns:
        ``{"log_loss", "brier", "accuracy", "n"}``. Brier is the multiclass form
        (mean over rows of the summed squared error across classes, range 0-2);
        lower log loss / Brier and higher accuracy are better.

    Raises:
        ValueError: If shapes disagree or an unexpected label appears.
    """
    labels = list(classes)
    y = np.asarray(y_true)
    if proba.shape != (len(y), len(labels)):
        raise ValueError(
            f"proba shape {proba.shape} does not match (n={len(y)}, k={len(labels)})"
        )
    unexpected = set(np.unique(y)) - set(labels)
    if unexpected:
        raise ValueError(f"y_true contains labels outside {labels}: {sorted(unexpected)}")

    index = {c: i for i, c in enumerate(labels)}
    true_idx = np.array([index[v] for v in y])
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), true_idx] = 1.0

    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    # Compute log loss against our own column order (sklearn.log_loss assumes
    # lexicographically-sorted columns, which would silently mis-map our classes).
    true_proba = proba[np.arange(len(y)), true_idx]
    ll = float(-np.mean(np.log(np.clip(true_proba, _LOG_LOSS_EPS, 1.0))))
    preds = np.asarray(labels)[proba.argmax(axis=1)]
    accuracy = float(accuracy_score(y, preds))
    return {"log_loss": ll, "brier": brier, "accuracy": accuracy, "n": int(len(y))}


def evaluate_estimator(
    estimator: object,
    X: pd.DataFrame,
    y_true: Sequence[str] | pd.Series | np.ndarray,
    *,
    classes: Sequence[str] = CLASS_ORDER,
) -> dict[str, float]:
    """Score any fitted estimator with a ``predict_proba`` against ``y_true``.

    Reorders the estimator's probability columns to ``classes`` before scoring,
    so a model whose ``classes_`` are in a different order is handled correctly.

    Args:
        estimator: A fitted classifier exposing ``predict_proba`` and ``classes_``.
        X: Feature frame to predict on.
        y_true: True labels aligned to ``X``.
        classes: Desired class order for scoring.

    Returns:
        The metrics dict from :func:`evaluate_proba`.
    """
    proba = predict_proba_in_order(estimator, X, classes=classes)
    return evaluate_proba(y_true, proba, classes=classes)


def predict_proba_in_order(
    estimator: object, X: pd.DataFrame, *, classes: Sequence[str] = CLASS_ORDER
) -> np.ndarray:
    """Return ``estimator.predict_proba(X)`` with columns ordered as ``classes``."""
    proba = np.asarray(estimator.predict_proba(X))  # type: ignore[attr-defined]
    est_classes = list(estimator.classes_)  # type: ignore[attr-defined]
    if est_classes == list(classes):
        return proba
    index = {c: i for i, c in enumerate(est_classes)}
    ordered = np.zeros((proba.shape[0], len(classes)))
    for j, cls in enumerate(classes):
        if cls in index:
            ordered[:, j] = proba[:, index[cls]]
    return ordered


def probabilities_to_frame(
    meta: pd.DataFrame,
    proba: np.ndarray,
    y_true: Sequence[str] | pd.Series | np.ndarray | None = None,
    *,
    classes: Sequence[str] = CLASS_ORDER,
    id_columns: Sequence[str] = ("match_id", "date", "team_a", "team_b"),
) -> pd.DataFrame:
    """Assemble a predictions frame: id columns + per-class ``p_<class>`` columns.

    Args:
        meta: Source frame holding the identifier columns (e.g. the test split).
        proba: Probabilities, shape ``(len(meta), len(classes))``.
        y_true: Optional true labels to attach as ``target_class``.
        classes: Class order matching ``proba`` columns.
        id_columns: Identifier columns to carry through (those that exist in
            ``meta`` are kept, in order).

    Returns:
        A new predictions DataFrame.
    """
    present_ids = [c for c in id_columns if c in meta.columns]
    out = meta[present_ids].reset_index(drop=True).copy()
    if y_true is not None:
        out["target_class"] = np.asarray(y_true)
    for j, cls in enumerate(classes):
        out[f"p_{cls}"] = proba[:, j]
    return out


def save_metrics(metrics: dict, path: Path | str) -> Path:
    """Write a metrics dict to JSON (sorted keys, 2-space indent → deterministic).

    Args:
        metrics: Any JSON-serializable metrics mapping.
        path: Output path; parent directories are created.

    Returns:
        The path written to.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path


def temporal_backtest(features: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    """Backtest a model by training before ``cutoff`` and scoring after it.

    TODO(slice 2/5): walk forward through time, fit on matches before each
    cutoff, predict the next window, and aggregate Brier score, log loss, and
    accuracy. This is the single source of truth for "is model B better than
    the Elo baseline?".

    Args:
        features: Feature matrix with labels.
        cutoff: The train/test split date.

    Returns:
        Mapping of metric name to value (e.g. ``{"brier": ..., "log_loss": ...}``).
    """
    raise NotImplementedError("temporal_backtest is implemented in slice 2.")
