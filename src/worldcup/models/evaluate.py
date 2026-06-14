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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from worldcup.data.clean_data import TARGET_CLASSES

_LOG_LOSS_EPS = 1e-15

# Plain-English explanations of the scoring rules, reused in the markdown report
# so the metrics are understandable without a stats background.
LOG_LOSS_EXPLANATION = (
    "**Log loss** measures how good the predicted *probabilities* are, not just the "
    "yes/no call. It rewards being confident and right, and punishes being confident "
    "and wrong very harshly. Lower is better. A model that learns nothing and always "
    "predicts the three outcomes as equally likely scores about **1.099** (`ln 3`); "
    "a perfect, fully-confident model scores **0**."
)
BRIER_EXPLANATION = (
    "**Brier score** is the average squared distance between the predicted "
    "probabilities and what actually happened. Put 100% on the correct result and you "
    "score 0; spread your bets and you score somewhere in between. Lower is better "
    "(0 is perfect, 2 is the worst possible in a 3-way race)."
)
CALIBRATION_EXPLANATION = (
    "**Calibration** asks whether the probabilities mean what they say: of all the "
    "matches the model calls *60% likely*, do roughly 60% actually happen? The "
    "calibration curve plots predicted probability against the observed rate — the "
    "closer it hugs the diagonal, the more honest the numbers. **Expected Calibration "
    "Error (ECE)** is the average gap from that diagonal (lower is better), which "
    "matters because these probabilities feed a Monte Carlo simulation downstream."
)

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


def predicted_labels(proba: np.ndarray, *, classes: Sequence[str] = CLASS_ORDER) -> np.ndarray:
    """Return the argmax class label per row."""
    return np.asarray(list(classes))[proba.argmax(axis=1)]


def confusion_matrix_frame(
    y_true: Sequence[str] | pd.Series | np.ndarray,
    y_pred: Sequence[str] | pd.Series | np.ndarray,
    *,
    classes: Sequence[str] = CLASS_ORDER,
) -> pd.DataFrame:
    """Confusion matrix as a labeled frame (rows = actual, columns = predicted)."""
    labels = list(classes)
    matrix = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=labels)
    return pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )


def class_performance_frame(
    y_true: Sequence[str] | pd.Series | np.ndarray,
    y_pred: Sequence[str] | pd.Series | np.ndarray,
    *,
    classes: Sequence[str] = CLASS_ORDER,
) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support (one row per class).

    Precision = of matches predicted as this class, how many were right. Recall =
    of matches that truly were this class, how many we caught. F1 is their harmonic
    mean. ``zero_division=0`` keeps a never-predicted class at 0 rather than NaN.
    """
    labels = list(classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        np.asarray(y_true), np.asarray(y_pred), labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )


def reliability_curve(
    y_true: Sequence[str] | pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    classes: Sequence[str] = CLASS_ORDER,
    n_bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """One-vs-rest reliability points per class for a calibration diagram.

    For each class, predictions are bucketed into ``n_bins`` equal-width
    probability bins; each non-empty bin yields its mean predicted probability and
    the observed frequency of that class. Plotting observed vs predicted against
    the diagonal shows over-/under-confidence.

    Returns:
        ``{class_name: DataFrame[mean_predicted, observed_frequency, count]}``.
    """
    labels = list(classes)
    y = np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: dict[str, pd.DataFrame] = {}
    for k, cls in enumerate(labels):
        p = proba[:, k]
        hit = (y == cls).astype(float)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            in_bin = (p > lo) & (p <= hi) if hi < 1.0 else (p > lo)
            count = int(in_bin.sum())
            if count:
                rows.append(
                    {
                        "mean_predicted": float(p[in_bin].mean()),
                        "observed_frequency": float(hit[in_bin].mean()),
                        "count": count,
                    }
                )
        out[cls] = pd.DataFrame(rows, columns=["mean_predicted", "observed_frequency", "count"])
    return out


def metrics_summary_frame(metrics: dict) -> pd.DataFrame:
    """Recruiter-friendly test-set table: main model vs every baseline.

    Reads the metrics dict produced by :func:`worldcup.models.train.run_training`
    and returns one row per model with test log loss / Brier / accuracy, sorted by
    log loss (best first), with a ``model`` column flagging the main model.
    """
    rows: list[dict] = []
    main_test = metrics["main_model"]["test"]["calibrated"]
    rows.append({"model": "main_model (calibrated)", **_metric_cols(main_test)})
    if "base" in metrics["main_model"]["test"]:
        rows.append(
            {"model": "main_model (uncalibrated)", **_metric_cols(metrics["main_model"]["test"]["base"])}
        )
    for name, entry in metrics.get("baselines", {}).items():
        rows.append({"model": name, **_metric_cols(entry["test"])})

    frame = pd.DataFrame(rows)
    return frame.sort_values("log_loss", kind="stable").reset_index(drop=True)


def render_evaluation_markdown(
    metrics: dict,
    predictions: pd.DataFrame,
    *,
    calibration_figure: str,
    confusion_figure: str,
    classes: Sequence[str] = CLASS_ORDER,
) -> str:
    """Render the full, honest evaluation report as a Markdown string.

    Args:
        metrics: The training metrics dict (see ``run_training``).
        predictions: Test-set predictions frame with ``target_class`` and
            ``p_<class>`` columns.
        calibration_figure: Report-relative path to the calibration PNG.
        confusion_figure: Report-relative path to the confusion-matrix PNG.
        classes: Class order.

    Returns:
        Markdown text (no figures are written here — only referenced).
    """
    labels = list(classes)
    proba = predictions[[f"p_{c}" for c in labels]].to_numpy()
    y_true = predictions["target_class"].to_numpy()
    y_pred = predicted_labels(proba, classes=labels)

    cmp = metrics.get("comparison", {})
    cal_err = metrics["main_model"].get("test_calibration_error", {})
    n_test = metrics["main_model"]["test"]["calibrated"].get("n", len(predictions))
    verdict = (
        "edges ahead of" if cmp.get("beats_best_baseline") else "does not beat"
    )

    lines: list[str] = []
    lines.append("# Model Evaluation Report")
    lines.append("")
    lines.append(
        "Predicting the 3-way outcome of an international football match — "
        "**team A win**, **draw**, **team B win** — from leakage-safe, as-of-kickoff "
        "features. Every number below is measured on a **temporally held-out test set** "
        f"of **{n_test:,} matches** the model never saw during training or calibration."
    )
    lines.append("")
    lines.append(
        "> **Honest framing:** this is a deliberately simple, well-regularized model "
        "judged against honest baselines. On the test set it "
        f"**{verdict}** the strongest baseline (see below) — a small, real edge, not a "
        "dramatic one. It is not state of the art and is not presented as such."
    )
    lines.append("")

    lines.append("## How to read the metrics")
    lines.append("")
    lines.append(f"- {LOG_LOSS_EXPLANATION}")
    lines.append(f"- {BRIER_EXPLANATION}")
    lines.append(f"- {CALIBRATION_EXPLANATION}")
    lines.append("")

    lines.append("## Metrics summary (test set)")
    lines.append("")
    lines.append(_frame_to_markdown(metrics_summary_frame(metrics), floatfmt=4))
    lines.append("")

    lines.append("## Baseline comparison")
    lines.append("")
    lines.append(
        f"- Main model (calibrated) test log loss: **{cmp.get('main_calibrated_test_log_loss', float('nan')):.4f}**"
    )
    lines.append(
        f"- Best baseline test log loss: **{cmp.get('best_baseline_test_log_loss', float('nan')):.4f}**"
    )
    lines.append(
        "- A baseline that beats your model is a useful result too: it tells you the "
        "fancy features are not pulling their weight yet."
    )
    lines.append("")

    lines.append("## Calibration")
    lines.append("")
    if cal_err:
        lines.append(
            f"Expected Calibration Error on the test set: **{cal_err.get('base', float('nan')):.4f}** "
            f"before calibration → **{cal_err.get('calibrated', float('nan')):.4f}** after "
            f"({metrics.get('calibration_method', 'n/a')} scaling). Lower is better."
        )
        lines.append("")
    lines.append(f"![Calibration curve]({calibration_figure})")
    lines.append("")
    lines.append(
        "*Each line is one outcome class (one-vs-rest). Points on the diagonal mean the "
        "stated probabilities match reality.*"
    )
    lines.append("")

    lines.append("## Confusion matrix")
    lines.append("")
    lines.append(f"![Confusion matrix]({confusion_figure})")
    lines.append("")
    lines.append(_frame_to_markdown(confusion_matrix_frame(y_true, y_pred, classes=labels).reset_index()))
    lines.append("")
    lines.append(
        "*Rows are what actually happened, columns are what the model predicted. Draws "
        "are notoriously hard to call in football — expect them to be the weakest row.*"
    )
    lines.append("")

    lines.append("## Class-level performance")
    lines.append("")
    lines.append(_frame_to_markdown(class_performance_frame(y_true, y_pred, classes=labels), floatfmt=3))
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Scores reflect the current feature set only; stronger signals (self-computed "
        "Elo, FIFA ranking) are planned and not yet included."
    )
    lines.append(
        "- International football is high-variance; even a good model will look only "
        "modestly better than the base rates."
    )
    lines.append("- No hyperparameters were tuned on the test set; it was scored once.")
    lines.append("")
    return "\n".join(lines)


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


def _metric_cols(entry: dict) -> dict:
    """Pull the headline metrics out of an evaluate_proba dict (rounded)."""
    return {
        "log_loss": round(float(entry["log_loss"]), 4),
        "brier": round(float(entry["brier"]), 4),
        "accuracy": round(float(entry["accuracy"]), 4),
        "n": int(entry.get("n", 0)),
    }


def _frame_to_markdown(df: pd.DataFrame, *, floatfmt: int | None = None) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table (no extra deps)."""

    def fmt(value: object) -> str:
        if floatfmt is not None and isinstance(value, float):
            return f"{value:.{floatfmt}f}"
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(fmt(v) for v in record) + " |"
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


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
