"""Train the calibrated main match-outcome model and compare it to the baselines.

The main model predicts the 3-class outcome (``team_a_win`` / ``draw`` /
``team_b_win``). The default is **multinomial logistic regression** in a small,
deterministic, CPU-friendly sklearn pipeline (constant-impute → standardize →
logistic). With ~10-20 leakage-safe difference features and tens of thousands of
rows, a regularized linear model is the honest first choice: fast on a 2-vCPU VM,
hard to overfit, and a clean thing to beat. ``HistGradientBoostingClassifier`` is
available behind a config switch for when the data justifies it -- not by default.

Discipline encoded here:
    * Time-aware split (train < validation < test); never a random K-fold.
    * The base model is fit on **train** only, the calibrator on **validation**
      (see :mod:`worldcup.models.calibrate`), and **test** is scored once at the
      end -- never tuned on.
    * Every baseline is fit on the same train split and scored on the same test
      split, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from worldcup.config import RANDOM_SEED
from worldcup.data.clean_data import TARGET_CLASSES
from worldcup.data.validate_data import DataValidationError
from worldcup.features.rolling_features import DIFFERENCE_FEATURES
from worldcup.models.baseline import available_baselines
from worldcup.models.calibrate import expected_calibration_error, fit_calibrated_model
from worldcup.models.evaluate import (
    CLASS_ORDER,
    evaluate_estimator,
    evaluate_proba,
    predict_proba_in_order,
    probabilities_to_frame,
    temporal_train_val_test_split,
)

logger = logging.getLogger(__name__)

TARGET_COLUMN = "target_class"

# Candidate input features, weakest assumptions first. Only those actually present
# in the feature table are used (rating features arrive in later slices), so the
# pipeline degrades gracefully instead of demanding columns it may not have.
# NB: every candidate is computed as-of kickoff; no label/score columns here.
DEFAULT_FEATURE_CANDIDATES: tuple[str, ...] = (
    *DIFFERENCE_FEATURES,  # form_5_diff, ..., rest_days_diff
    "elo_diff",
    "fifa_points_diff",
    "fifa_rank_diff",
    "is_team_a_home",
    "is_neutral",
)

# Columns that must never be used as inputs (labels / identifiers / leakage).
_FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "match_id",
        "date",
        "team_a",
        "team_b",
        "team_a_score",
        "team_b_score",
        "team_a_result",
        "target_class",
        "tournament",
        "host_country",
    }
)


@dataclass
class TrainingResult:
    """Everything a training run produces, ready for the CLI to persist."""

    model: ClassifierMixin  # calibrated (or base, if calibration="none")
    base_model: ClassifierMixin
    features: list[str]
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)


def select_features(features: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return the candidate columns present in ``features`` (order preserved).

    Drops anything forbidden (labels/identifiers) defensively and warns about
    candidates that are simply absent, so a missing rating feature is a logged
    skip rather than a crash.

    Raises:
        DataValidationError: If no usable feature columns remain.
    """
    forbidden = [c for c in candidates if c in _FORBIDDEN_FEATURES]
    if forbidden:
        raise DataValidationError(f"refusing to train on label/id columns: {forbidden}")

    present = [c for c in candidates if c in features.columns]
    missing = [c for c in candidates if c not in features.columns]
    if missing:
        logger.warning("Skipping %d unavailable feature(s): %s", len(missing), missing)
    if not present:
        raise DataValidationError("no usable feature columns found in the feature table")
    return present


def build_estimator(config: dict[str, Any]) -> Pipeline:
    """Build the (uncalibrated) main-model pipeline from config.

    ``type: logistic`` (default) → impute+scale+multinomial logistic.
    ``type: gradient_boosting`` → a shallow, capped HistGradientBoosting (still
    CPU-only and seeded), used only when explicitly requested.
    """
    model_type = str(config.get("type", "logistic")).lower()
    if model_type == "logistic":
        clf: ClassifierMixin = LogisticRegression(
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 1000)),
            random_state=RANDOM_SEED,
        )
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scale", StandardScaler()),
                ("clf", clf),
            ]
        )
    if model_type == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            learning_rate=float(config.get("learning_rate", 0.05)),
            max_depth=int(config.get("max_depth", 3)),
            max_iter=int(config.get("n_estimators", 300)),
            l2_regularization=float(config.get("l2_regularization", 1.0)),
            early_stopping=True,
            random_state=RANDOM_SEED,
        )
        # Tree model handles NaN natively; no imputation/scaling needed.
        return Pipeline([("clf", clf)])
    raise DataValidationError(
        f"unknown main_model type {model_type!r}; expected 'logistic' or 'gradient_boosting'"
    )


def run_training(features: pd.DataFrame, config: dict[str, Any] | None = None) -> TrainingResult:
    """Train, calibrate, and evaluate the main model against the baselines.

    Args:
        features: Leakage-safe feature matrix with a ``date`` column and the
            ``target_class`` label (played matches only — unplayed fixtures must
            already be excluded upstream).
        config: ``main_model`` config block (see ``configs/model_config.yaml``).
            Missing keys fall back to sensible defaults.

    Returns:
        A :class:`TrainingResult` with the calibrated model, feature list, the
        metrics comparison, and the test-set predictions.

    Raises:
        DataValidationError: If the label column is missing or unusable.
    """
    cfg = dict(config or {})
    if TARGET_COLUMN not in features.columns:
        raise DataValidationError(f"features missing label column '{TARGET_COLUMN}'")

    labeled = features[features[TARGET_COLUMN].isin(TARGET_CLASSES)].copy()
    dropped = len(features) - len(labeled)
    if dropped:
        logger.warning("Dropped %d unlabeled (unplayed) row(s) before training", dropped)
    if labeled.empty:
        raise DataValidationError("no labeled matches available to train the model")

    feature_list = select_features(
        labeled, list(cfg.get("feature_candidates", DEFAULT_FEATURE_CANDIDATES))
    )
    cal_method = str(cfg.get("calibration", "isotonic")).lower()

    train, val, test = temporal_train_val_test_split(
        labeled,
        val_fraction=float(cfg.get("val_fraction", 0.15)),
        test_fraction=float(cfg.get("test_fraction", 0.15)),
    )
    logger.info(
        "Split: train=%d (%s..%s) | val=%d (..%s) | test=%d (..%s) | %d feature(s)",
        len(train),
        train["date"].min().date(),
        train["date"].max().date(),
        len(val),
        val["date"].max().date(),
        len(test),
        test["date"].max().date(),
        len(feature_list),
    )

    X_train, y_train = _xy(train, feature_list)

    base = build_estimator(cfg)
    base.fit(X_train, y_train)
    calibrated = fit_calibrated_model(
        base, val[feature_list], val[TARGET_COLUMN].to_numpy(), method=cal_method
    )

    metrics = _build_metrics(base, calibrated, cal_method, train, val, test, feature_list)

    test_proba = predict_proba_in_order(calibrated, test[feature_list])
    predictions = probabilities_to_frame(test, test_proba, test[TARGET_COLUMN].to_numpy())

    return TrainingResult(
        model=calibrated,
        base_model=base,
        features=feature_list,
        metrics=metrics,
        predictions=predictions,
        config=cfg,
    )


# --- internal helpers -------------------------------------------------------


def _xy(split: pd.DataFrame, feature_list: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    return split[feature_list], split[TARGET_COLUMN].to_numpy()


def _build_metrics(
    base: ClassifierMixin,
    calibrated: ClassifierMixin,
    cal_method: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_list: list[str],
) -> dict[str, Any]:
    """Assemble the full metrics comparison (main model vs baselines)."""
    X_val, y_val = val[feature_list], val[TARGET_COLUMN].to_numpy()
    X_test, y_test = test[feature_list], test[TARGET_COLUMN].to_numpy()

    base_test = evaluate_estimator(base, X_test, y_test)
    cal_test = evaluate_estimator(calibrated, X_test, y_test)

    baselines = _baseline_metrics(train, test)
    best_baseline_ll = min((m["log_loss"] for m in baselines.values()), default=float("inf"))

    return {
        "classes": list(CLASS_ORDER),
        "features": feature_list,
        "calibration_method": cal_method,
        "main_model": {
            "validation": {
                "base": evaluate_estimator(base, X_val, y_val),
                "calibrated": evaluate_estimator(calibrated, X_val, y_val),
            },
            "test": {"base": base_test, "calibrated": cal_test},
            "test_calibration_error": {
                "base": expected_calibration_error(
                    y_test, predict_proba_in_order(base, X_test), CLASS_ORDER
                ),
                "calibrated": expected_calibration_error(
                    y_test, predict_proba_in_order(calibrated, X_test), CLASS_ORDER
                ),
            },
        },
        "baselines": {name: {"test": m} for name, m in baselines.items()},
        "comparison": {
            "main_calibrated_test_log_loss": cal_test["log_loss"],
            "best_baseline_test_log_loss": best_baseline_ll,
            "beats_best_baseline": bool(cal_test["log_loss"] < best_baseline_ll),
        },
    }


def _baseline_metrics(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit each available baseline on train and score it on the same test rows.

    Baselines select their own feature columns from the frame and already emit
    probabilities in :data:`CLASS_ORDER`, so they are scored with
    :func:`evaluate_proba` directly (they have no sklearn ``classes_``).
    """
    y_train = train[TARGET_COLUMN]
    out: dict[str, dict[str, float]] = {}
    for name, model in available_baselines(train).items():
        model.fit(train, y_train)
        out[name] = evaluate_proba(test[TARGET_COLUMN], model.predict_proba(test))
    return out
