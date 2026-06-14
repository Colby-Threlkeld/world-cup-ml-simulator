"""CLI: train, calibrate, and persist the main match-outcome model.

Pipeline: load the feature table → time-aware train/val/test split → fit the main
model (default multinomial logistic) on train → calibrate on validation → score on
the held-out test window and against every baseline → save four artifacts:

    * ``model.joblib``      — the fitted, calibrated estimator
    * ``feature_list.json`` — the exact input features + run config (provenance)
    * ``metrics.json``      — main-model vs baseline log loss / Brier / accuracy
    * ``predictions.csv``   — per-test-match calibrated class probabilities

Honesty notes: the test window is scored once and never tuned on; the script
reports whether the model beat the best baseline but makes no quality claim.

Usage::

    python scripts/train_model.py
    python scripts/train_model.py --features data/processed/features.csv \
        --output-dir data/processed/model
    python scripts/train_model.py --model-type gradient_boosting --sample 5000 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup.config import PROCESSED_DIR, model_config  # noqa: E402
from worldcup.data.clean_data import TARGET_CLASSES  # noqa: E402
from worldcup.data.validate_data import DataValidationError  # noqa: E402
from worldcup.models.evaluate import save_metrics  # noqa: E402
from worldcup.models.train import TrainingResult, run_training  # noqa: E402

logger = logging.getLogger("train_model")

DEFAULT_FEATURES = PROCESSED_DIR / "features.csv"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "model"
TARGET_COLUMN = "target_class"

# Artifact filenames (relative to --output-dir).
MODEL_FILE = "model.joblib"
FEATURES_FILE = "feature_list.json"
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.csv"


def main(argv: list[str] | None = None) -> int:
    """Run the main-model training pipeline. Returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        features = _load_features(args.features, sample=args.sample)
        config = _resolve_config(args)
        result = run_training(features, config)
        out_dir = _save_artifacts(result, args.output_dir)
    except (FileNotFoundError, DataValidationError, ValueError) as exc:
        logger.error("Model training failed: %s", exc)
        return 1

    _log_summary(result)
    logger.info("Artifacts written to %s", out_dir)
    return 0


def _resolve_config(args: argparse.Namespace) -> dict:
    """Load the ``main_model`` config block and apply CLI overrides."""
    cfg = dict(model_config().get("main_model", {}))
    if args.model_type is not None:
        cfg["type"] = args.model_type
    if args.calibration is not None:
        cfg["calibration"] = args.calibration
    return cfg


def _load_features(path: Path, *, sample: int | None) -> pd.DataFrame:
    """Load the feature CSV (parsing dates); optionally take the earliest N rows."""
    if not path.exists():
        raise FileNotFoundError(f"features file not found: {path}")
    features = pd.read_csv(path, parse_dates=["date"])
    if sample is not None:
        labeled = features[features[TARGET_COLUMN].isin(TARGET_CLASSES)]
        features = labeled.sort_values("date", kind="stable").head(sample)
        logger.info("Sample mode: using the first %d labeled match(es)", len(features))
    return features


def _save_artifacts(result: TrainingResult, out_dir: Path) -> Path:
    """Persist the model, feature list, metrics, and predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, out_dir / MODEL_FILE)
    (out_dir / FEATURES_FILE).write_text(
        json.dumps(
            {"features": result.features, "config": result.config, "classes": list(TARGET_CLASSES)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_metrics(result.metrics, out_dir / METRICS_FILE)
    result.predictions.to_csv(out_dir / PREDICTIONS_FILE, index=False)
    return out_dir


def _log_summary(result: TrainingResult) -> None:
    """Log the headline test-set comparison without overclaiming."""
    cal = result.metrics["main_model"]["test"]["calibrated"]
    cmp = result.metrics["comparison"]
    logger.info(
        "Main model (calibrated) test: log_loss=%.4f brier=%.4f acc=%.3f",
        cal["log_loss"], cal["brier"], cal["accuracy"],
    )
    verdict = "BEATS" if cmp["beats_best_baseline"] else "does NOT beat"
    logger.info(
        "Main model %s best baseline on test log loss (%.4f vs %.4f).",
        verdict, cmp["main_calibrated_test_log_loss"], cmp["best_baseline_test_log_loss"],
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the calibrated main model.")
    parser.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES, help=f"feature CSV (default: {DEFAULT_FEATURES})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"artifact dir (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--model-type", choices=["logistic", "gradient_boosting"], default=None, help="override main_model.type"
    )
    parser.add_argument(
        "--calibration", choices=["isotonic", "sigmoid", "none"], default=None, help="override calibration method"
    )
    parser.add_argument("--sample", type=int, default=None, help="use only the first N matches (quick smoke)")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
