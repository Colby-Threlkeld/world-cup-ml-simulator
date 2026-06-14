"""Tests for the backtesting framework, driven by a tiny fake tournament."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_backtest as cli  # noqa: E402
from worldcup.backtesting import (  # noqa: E402
    BACKTEST_TOURNAMENTS,
    BacktestError,
    BacktestTournament,
    add_elo_features,
    backtest_report,
    backtest_tournament,
    entering_strengths,
    render_backtest_markdown,
    run_backtests,
    strength_ranking,
    winner_rank,
)

_STRENGTH = {"S0": 3, "S1": 2, "S2": 1, "S3": 0}  # S0 strongest, S3 weakest
_TOURNEY = BacktestTournament(
    "Mini Cup", pd.Timestamp("2014-01-01"), pd.Timestamp("2014-02-01"), "S0", "Mini Cup"
)


def _row(mid: int, date: str, a: str, b: str, strength: dict[str, int]) -> dict:
    """A match where the stronger side wins (equal strength draws)."""
    sa, sb = strength[a], strength[b]
    ha, hb = (1, 1) if sa == sb else ((2, 0) if sa > sb else (0, 2))
    target = "team_a_win" if ha > hb else ("team_b_win" if ha < hb else "draw")
    tourn = "Mini Cup" if pd.Timestamp(date).year == 2014 else "Friendly"
    return {
        "match_id": mid, "date": pd.Timestamp(date), "team_a": a, "team_b": b,
        "team_a_score": ha, "team_b_score": hb, "target_class": target,
        "tournament": tourn, "is_team_a_home": False, "is_neutral": True,
    }


def _features(strength: dict[str, int] | None = None) -> pd.DataFrame:
    """History (2010-2013) of round-robins, then a fake 'Mini Cup' in Jan 2014.

    Each pairing is played in both orientations across years, so both
    ``team_a_win`` and ``team_b_win`` appear in training (the model needs >= 2
    classes). ``strength`` controls who is good (default: S0 strongest).
    """
    strength = strength or _STRENGTH
    teams = ["S0", "S1", "S2", "S3"]
    pairs = [(x, y) for i, x in enumerate(teams) for y in teams[i + 1 :]]
    rows: list[dict] = []
    mid = 0
    for year in range(2010, 2014):
        for month, (a, b) in enumerate(pairs, start=1):
            # Alternate which team is listed first so both win-classes occur.
            home, away = (a, b) if year % 2 == 0 else (b, a)
            rows.append(_row(mid, f"{year}-{month:02d}-15", home, away, strength))
            mid += 1
    for week, (a, b) in enumerate(pairs, start=1):
        home, away = (a, b) if week % 2 == 0 else (b, a)
        rows.append(_row(mid, f"2014-01-{week + 5:02d}", home, away, strength))
        mid += 1
    return pd.DataFrame(rows)


_WEAK_CHAMPION_STRENGTH = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}  # S0 (champion) is weakest


# --- registry ---------------------------------------------------------------


def test_registry_has_three_real_tournaments() -> None:
    names = [t.name for t in BACKTEST_TOURNAMENTS]
    champs = [t.champion for t in BACKTEST_TOURNAMENTS]
    assert names == ["2014 World Cup", "2018 World Cup", "2022 World Cup"]
    assert champs == ["Germany", "France", "Argentina"]
    for t in BACKTEST_TOURNAMENTS:
        assert t.start < t.end


# --- leakage-safe Elo -------------------------------------------------------


def test_add_elo_features_is_leakage_safe_and_orders_strength() -> None:
    feats = add_elo_features(_features())
    # A team's very first match must use the base rating (no self-knowledge).
    first = feats.sort_values("match_id").iloc[0]
    assert first["team_a_elo"] == 1500.0 and first["team_b_elo"] == 1500.0
    # After a season of the strong team winning, entering strengths are ordered.
    test_rows = feats[feats["tournament"] == "Mini Cup"]
    strengths = entering_strengths(test_rows)
    assert strengths["S0"] > strengths["S1"] > strengths["S2"] > strengths["S3"]


def test_add_elo_features_requires_columns() -> None:
    with pytest.raises(BacktestError):
        add_elo_features(pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]}))


# --- winner-rank helper -----------------------------------------------------


def test_winner_rank_top_k_boundaries() -> None:
    ranking = [f"T{i}" for i in range(12)]  # T0 best ... T11 worst
    assert winner_rank(ranking, "T0")["predicted_rank"] == 1
    third = winner_rank(ranking, "T2")
    assert third["in_top_3"] and third["in_top_5"] and third["in_top_10"]
    fourth = winner_rank(ranking, "T3")
    assert not fourth["in_top_3"] and fourth["in_top_5"]
    eleventh = winner_rank(ranking, "T10")
    assert not eleventh["in_top_10"]


def test_winner_rank_champion_absent() -> None:
    report = winner_rank(["A", "B"], "Z")
    assert report["predicted_rank"] is None
    assert not report["in_top_3"]


def test_strength_ranking_orders_desc() -> None:
    assert strength_ranking(_STRENGTH, ["S2", "S0", "S3", "S1"]) == ["S0", "S1", "S2", "S3"]


# --- backtest_tournament end-to-end -----------------------------------------


def test_backtest_produces_valid_probabilities_and_metrics() -> None:
    result = backtest_tournament(add_elo_features(_features()), _TOURNEY)
    proba = result.predictions[[f"p_{c}" for c in ("team_a_win", "draw", "team_b_win")]].to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert ((proba >= 0) & (proba <= 1)).all()
    for key in ("log_loss", "brier", "accuracy", "calibration_error", "n"):
        assert key in result.metrics
        assert np.isfinite(result.metrics[key])


def test_backtest_trains_only_on_pre_tournament_matches() -> None:
    feats = add_elo_features(_features())
    result = backtest_tournament(feats, _TOURNEY)
    n_pre = int((feats["date"] < _TOURNEY.start).sum())
    n_window = int((feats["tournament"] == "Mini Cup").sum())
    assert result.n_train == n_pre
    assert result.n_test == n_window
    # Predictions cover exactly the tournament matches, all dated within the window.
    assert len(result.predictions) == n_window
    assert (result.predictions["date"] >= _TOURNEY.start).all()


def test_strong_champion_ranks_first() -> None:
    result = backtest_tournament(add_elo_features(_features()), _TOURNEY)
    assert result.winner_rank["predicted_rank"] == 1
    assert result.winner_rank["in_top_3"]


def test_weak_champion_ranks_last() -> None:
    feats = add_elo_features(_features(_WEAK_CHAMPION_STRENGTH))
    result = backtest_tournament(feats, _TOURNEY)
    assert result.winner_rank["predicted_rank"] == 4  # S0 is now the weakest of 4
    assert not result.winner_rank["in_top_3"]


def test_backtest_is_deterministic() -> None:
    feats = add_elo_features(_features())
    m1 = backtest_tournament(feats, _TOURNEY).metrics
    m2 = backtest_tournament(feats, _TOURNEY).metrics
    assert m1 == m2


def test_backtest_errors_without_tournament_matches() -> None:
    feats = add_elo_features(_features())
    empty_window = BacktestTournament(
        "Nope", pd.Timestamp("1990-01-01"), pd.Timestamp("1990-02-01"), "S0", "Mini Cup"
    )
    with pytest.raises(BacktestError):
        backtest_tournament(feats, empty_window)


# --- report + run_backtests -------------------------------------------------


def test_run_backtests_and_report() -> None:
    results = run_backtests(_features(), tournaments=[_TOURNEY])
    report = backtest_report(results)
    assert report["aggregate"]["n_tournaments"] == 1
    assert set(report["tournaments"][0]) >= {
        "tournament", "log_loss", "champion_predicted_rank", "champion_in_top_5"
    }


def test_render_markdown_is_honest_and_complete() -> None:
    md = render_backtest_markdown(run_backtests(_features(), tournaments=[_TOURNEY]))
    for heading in ("# Backtesting Report", "## Match-prediction accuracy", "## Caveats"):
        assert heading in md
    assert "Mini Cup" in md
    # Honest framing about the missing historical bracket configs.
    assert "not implemented" in md.lower() or "todo" in md.lower()


# --- CLI --------------------------------------------------------------------


def test_cli_missing_matches_file_returns_error(tmp_path: Path) -> None:
    rc = cli.main(["--matches", str(tmp_path / "absent.parquet"), "--output-dir", str(tmp_path / "o"),
                   "--report", str(tmp_path / "r.md")])
    assert rc == 1
