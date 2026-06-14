"""Tests for the evaluation-report metrics, plots, and CLI."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_evaluation_report as cli  # noqa: E402

from worldcup.data.clean_data import TARGET_CLASSES  # noqa: E402
from worldcup.models.evaluate import (  # noqa: E402
    BRIER_EXPLANATION,
    CALIBRATION_EXPLANATION,
    LOG_LOSS_EXPLANATION,
    class_performance_frame,
    confusion_matrix_frame,
    metrics_summary_frame,
    predicted_labels,
    reliability_curve,
    render_evaluation_markdown,
)
from worldcup.visualization.plots import (  # noqa: E402
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_title_probabilities,
    save_figure,
)


def _predictions(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A predictions frame where the argmax usually matches the label."""
    rng = np.random.default_rng(seed)
    classes = np.array(TARGET_CLASSES)
    y = rng.choice(classes, n)
    proba = rng.dirichlet(np.ones(3), size=n)
    # Nudge probability mass toward the true class so the model is informative.
    for i, label in enumerate(y):
        k = list(TARGET_CLASSES).index(label)
        proba[i, k] += 1.0
    proba /= proba.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        {
            "match_id": np.arange(n),
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target_class": y,
        }
    )
    for k, c in enumerate(TARGET_CLASSES):
        frame[f"p_{c}"] = proba[:, k]
    return frame


def _metrics() -> dict:
    """A metrics dict shaped like worldcup.models.train.run_training output."""

    def m(ll, br, ac):
        return {"log_loss": ll, "brier": br, "accuracy": ac, "n": 200}

    return {
        "classes": list(TARGET_CLASSES),
        "features": ["form_5_diff"],
        "calibration_method": "isotonic",
        "main_model": {
            "validation": {"base": m(0.98, 0.58, 0.53), "calibrated": m(0.97, 0.57, 0.54)},
            "test": {"base": m(0.96, 0.57, 0.55), "calibrated": m(0.95, 0.56, 0.55)},
            "test_calibration_error": {"base": 0.04, "calibrated": 0.02},
        },
        "baselines": {
            "uniform_random": {"test": m(1.0986, 0.6667, 0.33)},
            "recent_form_logistic": {"test": m(0.97, 0.57, 0.54)},
        },
        "comparison": {
            "main_calibrated_test_log_loss": 0.95,
            "best_baseline_test_log_loss": 0.97,
            "beats_best_baseline": True,
        },
    }


# --- metrics functions ------------------------------------------------------


def test_confusion_matrix_is_square_and_total_matches() -> None:
    preds = _predictions()
    proba = preds[[f"p_{c}" for c in TARGET_CLASSES]].to_numpy()
    cm = confusion_matrix_frame(preds["target_class"], predicted_labels(proba))
    assert cm.shape == (3, 3)
    assert list(cm.columns) == list(TARGET_CLASSES)
    assert int(cm.to_numpy().sum()) == len(preds)


def test_class_performance_bounds_and_support() -> None:
    preds = _predictions()
    proba = preds[[f"p_{c}" for c in TARGET_CLASSES]].to_numpy()
    perf = class_performance_frame(preds["target_class"], predicted_labels(proba))
    assert list(perf["class"]) == list(TARGET_CLASSES)
    for col in ("precision", "recall", "f1"):
        assert ((perf[col] >= 0) & (perf[col] <= 1)).all()
    assert perf["support"].sum() == len(preds)


def test_reliability_curve_points_are_in_unit_square() -> None:
    preds = _predictions()
    proba = preds[[f"p_{c}" for c in TARGET_CLASSES]].to_numpy()
    curves = reliability_curve(preds["target_class"], proba, n_bins=5)
    assert set(curves) == set(TARGET_CLASSES)
    for frame in curves.values():
        if not frame.empty:
            assert frame["mean_predicted"].between(0, 1).all()
            assert frame["observed_frequency"].between(0, 1).all()
            assert (frame["count"] > 0).all()


def test_metrics_summary_frame_sorted_and_complete() -> None:
    summary = metrics_summary_frame(_metrics())
    assert {"model", "log_loss", "brier", "accuracy"}.issubset(summary.columns)
    # Sorted ascending by log loss (best first).
    assert list(summary["log_loss"]) == sorted(summary["log_loss"])
    # Main model + both baselines + uncalibrated variant are all present.
    assert any(name == "main_model (calibrated)" for name in summary["model"])
    assert {"uniform_random", "recent_form_logistic"} <= set(summary["model"])


def test_explanations_are_plain_english_and_nonempty() -> None:
    for text in (LOG_LOSS_EXPLANATION, BRIER_EXPLANATION, CALIBRATION_EXPLANATION):
        assert len(text) > 40
    assert "log loss" in LOG_LOSS_EXPLANATION.lower()
    assert "brier" in BRIER_EXPLANATION.lower()
    assert "calibrat" in CALIBRATION_EXPLANATION.lower()


def test_render_markdown_has_sections_and_does_not_overclaim() -> None:
    md = render_evaluation_markdown(
        _metrics(),
        _predictions(),
        calibration_figure="figures/c.png",
        confusion_figure="figures/cm.png",
    )
    for heading in (
        "# Model Evaluation Report",
        "## Metrics summary",
        "## Baseline comparison",
        "## Calibration",
        "## Confusion matrix",
        "## Class-level performance",
        "## Caveats",
    ):
        assert heading in md
    assert "figures/c.png" in md and "figures/cm.png" in md
    # Honesty guardrails: an explicit honest-framing note and caveats, and no
    # breathless positive superlatives.
    lowered = md.lower()
    assert "honest framing" in lowered
    for banned in ("best in class", "highly accurate", "world-class", "groundbreaking"):
        assert banned not in lowered


# --- plots ------------------------------------------------------------------


def test_plots_return_figures_and_save(tmp_path: Path) -> None:
    preds = _predictions()
    proba = preds[[f"p_{c}" for c in TARGET_CLASSES]].to_numpy()
    cal_fig = plot_calibration_curve(preds["target_class"], proba)
    cm_fig = plot_confusion_matrix(
        confusion_matrix_frame(preds["target_class"], predicted_labels(proba))
    )
    assert isinstance(cal_fig, Figure) and isinstance(cm_fig, Figure)

    cal_path = save_figure(cal_fig, tmp_path / "figures" / "cal.png")
    cm_path = save_figure(cm_fig, tmp_path / "figures" / "cm.png")
    assert cal_path.exists() and cal_path.stat().st_size > 0
    assert cm_path.exists() and cm_path.stat().st_size > 0


def test_plot_title_probabilities(tmp_path: Path) -> None:
    probs = pd.DataFrame(
        {
            "team": [f"T{i}" for i in range(20)],
            "win_world_cup_probability": np.linspace(0.2, 0.01, 20),
        }
    )
    fig = plot_title_probabilities(probs, top_n=10)
    assert isinstance(fig, Figure)
    out = save_figure(fig, tmp_path / "title.png")
    assert out.exists() and out.stat().st_size > 0
    with pytest.raises(KeyError):
        plot_title_probabilities(pd.DataFrame({"team": ["A"]}))  # missing prob column


# --- CLI end-to-end ---------------------------------------------------------


def test_cli_generates_all_outputs(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "metrics.json").write_text(json.dumps(_metrics()), encoding="utf-8")
    _predictions().to_csv(model_dir / "predictions.csv", index=False)
    out_dir = tmp_path / "reports"

    rc = cli.main(["--model-dir", str(model_dir), "--output-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "evaluation_report.md").exists()
    assert (out_dir / "metrics" / "model_metrics.json").exists()
    assert (out_dir / "figures" / "calibration_curve.png").exists()
    assert (out_dir / "figures" / "confusion_matrix.png").exists()


def test_cli_missing_artifacts_returns_error(tmp_path: Path) -> None:
    assert (
        cli.main(["--model-dir", str(tmp_path / "absent"), "--output-dir", str(tmp_path / "r")])
        == 1
    )
