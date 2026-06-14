"""CLI: turn a trained model's metrics + predictions into a recruiter-friendly report.

Reads the artifacts written by ``scripts/train_model.py`` (``metrics.json`` and
``predictions.csv``) and produces:

    * ``reports/metrics/model_metrics.json``      — the metrics, canonicalized here
    * ``reports/figures/calibration_curve.png``   — one-vs-rest reliability diagram
    * ``reports/figures/confusion_matrix.png``    — test-set confusion heatmap
    * ``reports/evaluation_report.md``            — the written report, with the
      metrics explained in plain English and performance stated honestly.

Usage::

    python scripts/generate_evaluation_report.py
    python scripts/generate_evaluation_report.py --model-dir data/processed/model \
        --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from worldcup.models.evaluate import (  # noqa: E402
    CLASS_ORDER,
    confusion_matrix_frame,
    predicted_labels,
    render_evaluation_markdown,
    save_metrics,
)
from worldcup.visualization.plots import (  # noqa: E402
    plot_calibration_curve,
    plot_confusion_matrix,
    save_figure,
)

logger = logging.getLogger("generate_evaluation_report")

DEFAULT_MODEL_DIR = PROCESSED_DIR / "model"
# Report-relative figure paths (so the Markdown links work from reports/).
_CAL_FIG_REL = "figures/calibration_curve.png"
_CM_FIG_REL = "figures/confusion_matrix.png"


def main(argv: list[str] | None = None) -> int:
    """Generate the evaluation report. Returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        metrics, predictions = _load_inputs(args.model_dir)
        out_dir: Path = args.output_dir
        proba = predictions[[f"p_{c}" for c in CLASS_ORDER]].to_numpy()
        y_true = predictions["target_class"].to_numpy()
        y_pred = predicted_labels(proba)

        # Figures.
        save_figure(plot_calibration_curve(y_true, proba), out_dir / _CAL_FIG_REL)
        confusion = confusion_matrix_frame(y_true, y_pred)
        save_figure(plot_confusion_matrix(confusion), out_dir / _CM_FIG_REL)

        # Metrics JSON (canonical copy for the report).
        save_metrics(metrics, out_dir / "metrics" / "model_metrics.json")

        # Markdown report.
        report = render_evaluation_markdown(
            metrics,
            predictions,
            calibration_figure=_CAL_FIG_REL,
            confusion_figure=_CM_FIG_REL,
        )
        report_path = out_dir / "evaluation_report.md"
        report_path.write_text(report, encoding="utf-8")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Report generation failed: %s", exc)
        return 1

    logger.info("Wrote report -> %s", report_path)
    logger.info("Wrote figures -> %s, %s", out_dir / _CAL_FIG_REL, out_dir / _CM_FIG_REL)
    logger.info("Wrote metrics -> %s", out_dir / "metrics" / "model_metrics.json")
    return 0


def _load_inputs(model_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Load ``metrics.json`` and ``predictions.csv`` from a training output dir."""
    metrics_path = model_dir / "metrics.json"
    predictions_path = model_dir / "predictions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found: {metrics_path} (run train_model.py first)")
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions not found: {predictions_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    expected = {"target_class", *(f"p_{c}" for c in CLASS_ORDER)}
    missing = expected - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions.csv missing columns: {sorted(missing)}")
    return metrics, predictions


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the model evaluation report.")
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help=f"training artifacts (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPORTS_DIR, help=f"report output dir (default: {REPORTS_DIR})"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
