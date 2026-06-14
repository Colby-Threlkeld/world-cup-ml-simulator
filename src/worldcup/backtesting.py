"""Backtest the match model against past World Cups (2014 / 2018 / 2022).

The question a backtest answers: *if we had stood before a past World Cup with only
the data available then, how well would this system have predicted it?* That makes
leakage the whole ballgame, so every step here is strictly as-of the tournament:

* **Training window** — the model is fit only on matches played *before* the
  tournament's start date.
* **Ratings** — a walk-forward Elo gives every match an ``elo_pre`` computed from
  prior matches only (:func:`add_elo_features`); the model never sees a rating that
  postdates the match it describes.
* **Evaluation** — predictions for the tournament's 64 matches are scored with log
  loss, Brier score, accuracy, and calibration error (ECE).

For the "did we fancy the eventual winner?" question we report the champion's rank
in the pre-tournament Elo favourite ordering, plus whether they were a top-3 / top-5
/ top-10 pick. A full Monte-Carlo *winner-probability* ranking would need each year's
official group draw and bracket encoded as a :class:`TournamentConfig`; those
historical configs are **not shipped** (only the 2026 placeholder exists), so that
path is wired but optional and documented as a TODO -- we do not fabricate a draw.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from worldcup.models.baseline import expected_score, update_rating
from worldcup.models.calibrate import expected_calibration_error, fit_calibrated_model
from worldcup.models.evaluate import (
    CLASS_ORDER,
    evaluate_proba,
    predict_proba_in_order,
    probabilities_to_frame,
)
from worldcup.models.train import (
    DEFAULT_FEATURE_CANDIDATES,
    TARGET_COLUMN,
    build_estimator,
    select_features,
)

logger = logging.getLogger(__name__)

# Elo hyperparameters for the leakage-safe walk-forward.
ELO_BASE = 1500.0
ELO_K = 32.0
ELO_HOME_ADVANTAGE = 65.0

# Only calibrate when there is enough pre-tournament data to spare a holdout tail.
_MIN_TRAIN_FOR_CALIBRATION = 500
_CALIBRATION_TAIL_FRACTION = 0.15


class BacktestError(ValueError):
    """Raised when a backtest cannot be run (missing data, columns, etc.)."""


@dataclass(frozen=True)
class BacktestTournament:
    """A past tournament to backtest: its window and confirmed champion."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    champion: str
    tournament_name: str = "FIFA World Cup"


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


# The three target tournaments. Champions are historical fact, not predictions.
BACKTEST_TOURNAMENTS: tuple[BacktestTournament, ...] = (
    BacktestTournament("2014 World Cup", _ts("2014-06-12"), _ts("2014-07-13"), "Germany"),
    BacktestTournament("2018 World Cup", _ts("2018-06-14"), _ts("2018-07-15"), "France"),
    BacktestTournament("2022 World Cup", _ts("2022-11-20"), _ts("2022-12-18"), "Argentina"),
)


@dataclass(frozen=True)
class BacktestResult:
    """Per-tournament backtest outcome."""

    name: str
    n_train: int
    n_test: int
    feature_list: list[str]
    metrics: dict[str, float]
    winner_rank: dict[str, Any]
    ranking: list[str]
    predictions: pd.DataFrame = field(repr=False)
    simulated: bool = False

    def metrics_row(self) -> dict[str, Any]:
        """Flat row for the summary table / metrics JSON (no DataFrames)."""
        return {
            "tournament": self.name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "log_loss": round(self.metrics["log_loss"], 4),
            "brier": round(self.metrics["brier"], 4),
            "accuracy": round(self.metrics["accuracy"], 4),
            "calibration_error": round(self.metrics["calibration_error"], 4),
            "champion": self.winner_rank["champion"],
            "champion_predicted_rank": self.winner_rank["predicted_rank"],
            "n_participants": self.winner_rank["n_participants"],
            "champion_in_top_3": self.winner_rank["in_top_3"],
            "champion_in_top_5": self.winner_rank["in_top_5"],
            "champion_in_top_10": self.winner_rank["in_top_10"],
            "winner_rank_basis": self.winner_rank["basis"],
        }


# --- leakage-safe Elo -------------------------------------------------------


def add_elo_features(
    model_df: pd.DataFrame,
    *,
    base: float = ELO_BASE,
    k_factor: float = ELO_K,
    home_advantage: float = ELO_HOME_ADVANTAGE,
) -> pd.DataFrame:
    """Add leakage-safe ``team_a_elo`` / ``team_b_elo`` / ``elo_diff`` columns.

    Walks matches in date order, recording each team's rating **before** the match
    (``elo_pre``) and only then applying the result — so a match never sees its own
    outcome reflected in its rating. Home advantage is applied when ``is_team_a_home``
    is present and true.

    Args:
        model_df: A model dataset with ``date``, ``team_a``, ``team_b``,
            ``team_a_score``, ``team_b_score`` (e.g. from ``build_feature_matrix``).
        base: Starting rating for an unseen team.
        k_factor: Elo update step.
        home_advantage: Rating points added to a true home side.

    Returns:
        A copy of ``model_df`` (original row order) with the three Elo columns.

    Raises:
        BacktestError: If required columns are missing.
    """
    required = {"date", "team_a", "team_b", "team_a_score", "team_b_score"}
    missing = required - set(model_df.columns)
    if missing:
        raise BacktestError(f"add_elo_features missing columns: {sorted(missing)}")

    sort_cols = ["date", "match_id"] if "match_id" in model_df.columns else ["date"]
    ordered = model_df.sort_values(sort_cols, kind="stable")
    has_home = "is_team_a_home" in ordered.columns

    ratings: dict[str, float] = {}
    n = len(ordered)
    a_pre = np.empty(n)
    b_pre = np.empty(n)
    for i, row in enumerate(ordered.itertuples(index=False)):
        ra = ratings.get(row.team_a, base)
        rb = ratings.get(row.team_b, base)
        a_pre[i] = ra
        b_pre[i] = rb
        adv = home_advantage if (has_home and getattr(row, "is_team_a_home")) else 0.0
        expected_a = expected_score(ra, rb, adv)
        if row.team_a_score > row.team_b_score:
            actual_a = 1.0
        elif row.team_a_score < row.team_b_score:
            actual_a = 0.0
        else:
            actual_a = 0.5
        ratings[row.team_a] = update_rating(ra, expected_a, actual_a, k_factor)
        ratings[row.team_b] = update_rating(rb, 1.0 - expected_a, 1.0 - actual_a, k_factor)

    out = ordered.copy()
    out["team_a_elo"] = a_pre
    out["team_b_elo"] = b_pre
    out["elo_diff"] = a_pre - b_pre
    return out.sort_index()


def entering_strengths(test_rows: pd.DataFrame) -> dict[str, float]:
    """Each participant's Elo *entering* the tournament (its first-match ``elo_pre``).

    Args:
        test_rows: The tournament's matches, with ``team_a_elo`` / ``team_b_elo``
            from :func:`add_elo_features`.

    Returns:
        Mapping of team -> entering Elo rating.
    """
    sort_cols = ["date", "match_id"] if "match_id" in test_rows.columns else ["date"]
    strengths: dict[str, float] = {}
    for row in test_rows.sort_values(sort_cols, kind="stable").itertuples(index=False):
        strengths.setdefault(row.team_a, float(row.team_a_elo))
        strengths.setdefault(row.team_b, float(row.team_b_elo))
    return strengths


def strength_ranking(strengths: Mapping[str, float], participants: Sequence[str]) -> list[str]:
    """Rank ``participants`` strongest-first by Elo (ties broken by name)."""
    return sorted(participants, key=lambda team: (-strengths.get(team, ELO_BASE), team))


def winner_rank(
    ranking: Sequence[str], champion: str, *, basis: str = "elo_favourite"
) -> dict[str, Any]:
    """Report the champion's position in a favourite ``ranking`` and top-k flags."""
    n = len(ranking)
    if champion not in ranking:
        return {
            "champion": champion,
            "predicted_rank": None,
            "n_participants": n,
            "in_top_3": False,
            "in_top_5": False,
            "in_top_10": False,
            "basis": basis,
        }
    rank = list(ranking).index(champion) + 1
    return {
        "champion": champion,
        "predicted_rank": rank,
        "n_participants": n,
        "in_top_3": rank <= 3,
        "in_top_5": rank <= 5,
        "in_top_10": rank <= 10,
        "basis": basis,
    }


# --- match prediction + evaluation ------------------------------------------


def predict_tournament_matches(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_list: Sequence[str],
    *,
    model_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Fit a calibrated model on the pre-tournament window and predict the test set.

    With enough training data, the model is fit on the earlier part of the window
    and calibrated on a held-out *tail* (the most recent pre-tournament matches);
    otherwise it is fit on everything uncalibrated. Returns probabilities in
    :data:`CLASS_ORDER`.
    """
    cfg = dict(model_config or {})
    cal_method = str(cfg.get("calibration", "sigmoid")).lower()
    features = list(feature_list)
    ordered = train.sort_values("date", kind="stable")
    estimator = build_estimator(cfg)

    can_calibrate = cal_method != "none" and len(ordered) >= _MIN_TRAIN_FOR_CALIBRATION
    if can_calibrate:
        cut = ordered["date"].quantile(1.0 - _CALIBRATION_TAIL_FRACTION)
        base_rows = ordered[ordered["date"] <= cut]
        calib_rows = ordered[ordered["date"] > cut]
        if len(base_rows) > 0 and calib_rows[TARGET_COLUMN].nunique() >= 2:
            estimator.fit(base_rows[features], base_rows[TARGET_COLUMN].to_numpy())
            model = fit_calibrated_model(
                estimator, calib_rows[features], calib_rows[TARGET_COLUMN].to_numpy(),
                method=cal_method,
            )
            return predict_proba_in_order(model, test[features])

    estimator.fit(ordered[features], ordered[TARGET_COLUMN].to_numpy())
    return predict_proba_in_order(estimator, test[features])


def evaluate_match_predictions(
    y_true: Sequence[str] | np.ndarray, proba: np.ndarray
) -> dict[str, float]:
    """Score tournament predictions: log loss, Brier, accuracy, calibration error."""
    metrics = evaluate_proba(y_true, proba)
    metrics["calibration_error"] = expected_calibration_error(y_true, proba, CLASS_ORDER)
    return metrics


# --- orchestration ----------------------------------------------------------


def backtest_tournament(
    features: pd.DataFrame,
    tournament: BacktestTournament,
    *,
    feature_candidates: Sequence[str] | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> BacktestResult:
    """Backtest one tournament end-to-end (train pre-start, score the matches).

    Args:
        features: Feature matrix that already includes the Elo columns (call
            :func:`add_elo_features` first) plus ``date``, ``tournament``,
            ``target_class``, and the candidate feature columns.
        tournament: The tournament window + champion.
        feature_candidates: Candidate feature columns (defaults to the model's).
        model_config: Estimator/calibration config.

    Returns:
        The :class:`BacktestResult`.

    Raises:
        BacktestError: If there are no training or no tournament matches.
    """
    train = features[features["date"] < tournament.start]
    in_window = (
        (features["date"] >= tournament.start)
        & (features["date"] <= tournament.end)
        & (features["tournament"].astype(str) == tournament.tournament_name)
    )
    test = features[in_window]
    if train.empty:
        raise BacktestError(f"{tournament.name}: no matches before {tournament.start.date()}")
    if test.empty:
        raise BacktestError(f"{tournament.name}: no '{tournament.tournament_name}' matches in window")

    feature_list = select_features(train, list(feature_candidates or DEFAULT_FEATURE_CANDIDATES))
    proba = predict_tournament_matches(train, test, feature_list, model_config=model_config)

    y_true = test[TARGET_COLUMN].to_numpy()
    metrics = evaluate_match_predictions(y_true, proba)
    predictions = probabilities_to_frame(test, proba, y_true)

    strengths = entering_strengths(test)
    participants = sorted(set(test["team_a"]) | set(test["team_b"]))
    ranking = strength_ranking(strengths, participants)
    rank_report = winner_rank(ranking, tournament.champion)

    logger.info(
        "%s: train=%d test=%d | log_loss=%.4f brier=%.4f acc=%.3f ece=%.4f | "
        "champion %s ranked %s/%d",
        tournament.name, len(train), len(test), metrics["log_loss"], metrics["brier"],
        metrics["accuracy"], metrics["calibration_error"], tournament.champion,
        rank_report["predicted_rank"], rank_report["n_participants"],
    )
    return BacktestResult(
        name=tournament.name,
        n_train=int(len(train)),
        n_test=int(len(test)),
        feature_list=feature_list,
        metrics=metrics,
        winner_rank=rank_report,
        ranking=ranking,
        predictions=predictions,
    )


def run_backtests(
    features: pd.DataFrame,
    *,
    tournaments: Sequence[BacktestTournament] = BACKTEST_TOURNAMENTS,
    feature_candidates: Sequence[str] | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> list[BacktestResult]:
    """Backtest several tournaments, adding Elo features once if absent."""
    if "elo_diff" not in features.columns:
        features = add_elo_features(features)
    return [
        backtest_tournament(
            features, tournament,
            feature_candidates=feature_candidates, model_config=model_config,
        )
        for tournament in tournaments
    ]


def backtest_report(results: Sequence[BacktestResult]) -> dict[str, Any]:
    """Assemble a JSON-serializable report (per-tournament rows + aggregate means)."""
    rows = [r.metrics_row() for r in results]
    metric_keys = ("log_loss", "brier", "accuracy", "calibration_error")
    aggregate = {
        f"mean_{key}": round(float(np.mean([r.metrics[key] for r in results])), 4)
        for key in metric_keys
    }
    aggregate["n_tournaments"] = len(results)
    aggregate["champion_top_5_rate"] = round(
        float(np.mean([r.winner_rank["in_top_5"] for r in results])), 4
    )
    return {"tournaments": rows, "aggregate": aggregate}


def render_backtest_markdown(results: Sequence[BacktestResult]) -> str:
    """Render an honest Markdown backtest report (no overclaiming)."""
    report = backtest_report(results)
    agg = report["aggregate"]
    lines: list[str] = []
    lines.append("# Backtesting Report")
    lines.append("")
    lines.append(
        "How this system *would have* predicted the 2014, 2018 and 2022 World Cups, "
        "trained only on matches played before each tournament and using only ratings "
        "available before each match (a leakage-safe walk-forward Elo). Predictions "
        "are scored on the 64 matches of each tournament."
    )
    lines.append("")
    lines.append("## Match-prediction accuracy")
    lines.append("")
    lines.append(_metrics_table(report["tournaments"]))
    lines.append("")
    lines.append(
        f"Across the three tournaments: mean log loss **{agg['mean_log_loss']}**, "
        f"mean Brier **{agg['mean_brier']}**, mean accuracy **{agg['mean_accuracy']}**, "
        f"mean calibration error **{agg['mean_calibration_error']}**. For reference an "
        "uninformed 1/3-each model scores log loss ~1.099."
    )
    lines.append("")
    lines.append("## Did we fancy the eventual champion?")
    lines.append("")
    lines.append(_winner_table(report["tournaments"]))
    lines.append("")
    lines.append(
        "*Rank is the champion's position in the pre-tournament Elo favourite ordering "
        "among the 32 participants — a proxy for a full winner-probability simulation.*"
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The champion **rank** uses the pre-tournament Elo favourite ordering, not a "
        "Monte-Carlo winner-probability simulation: that would require each year's "
        "official group draw and bracket encoded as a tournament config, which are "
        "**not implemented** for historical years (TODO). The simulation hook exists "
        "but is unused here — we do not fabricate a draw."
    )
    lines.append(
        "- Football is high-variance and knockouts are short series; a strong model can "
        "still rank the eventual winner outside the top few. These are small samples "
        "(3 tournaments, 64 matches each)."
    )
    lines.append("- No hyperparameters were tuned on the tournament being scored.")
    lines.append("")
    return "\n".join(lines)


# --- internal helpers -------------------------------------------------------


def _metrics_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = "| tournament | matches | log loss | Brier | accuracy | calibration error |"
    divider = "| --- | --- | --- | --- | --- | --- |"
    body = [
        f"| {r['tournament']} | {r['n_test']} | {r['log_loss']} | {r['brier']} | "
        f"{r['accuracy']} | {r['calibration_error']} |"
        for r in rows
    ]
    return "\n".join([header, divider, *body])


def _winner_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = "| tournament | champion | predicted rank | top 3 | top 5 | top 10 |"
    divider = "| --- | --- | --- | --- | --- | --- |"
    body = [
        f"| {r['tournament']} | {r['champion']} | {r['champion_predicted_rank']}/{r['n_participants']} "
        f"| {'yes' if r['champion_in_top_3'] else 'no'} | {'yes' if r['champion_in_top_5'] else 'no'} "
        f"| {'yes' if r['champion_in_top_10'] else 'no'} |"
        for r in rows
    ]
    return "\n".join([header, divider, *body])
