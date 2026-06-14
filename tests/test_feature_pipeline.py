"""Integration tests for the end-to-end feature-building pipeline.

Drives ``scripts/build_features.py`` over small fixture CSVs written to a tmp
``data/processed`` layout and asserts the contract the modeling slice depends on:
deterministic output, optional rating files, no crash when ratings are absent,
and a clear warning instead.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_features as cli  # noqa: E402

from worldcup.features.build_features import (  # noqa: E402
    LABEL_COLUMNS,
    ROLLING_FEATURE_COLUMNS,
    load_matches,
    validate_feature_matrix,
)
from worldcup.features.rating_features import (  # noqa: E402
    ELO_FEATURES,
    FIFA_FEATURES,
    RATING_FEATURES,
)


def _write_matches(path: Path) -> int:
    """Write a tiny cleaned-matches CSV; returns the row count."""
    rows = [
        (0, "2000-01-01", "Brazil", "Argentina", 2, 1, False),
        (1, "2000-02-01", "Brazil", "United States", 3, 0, False),
        (2, "2000-03-01", "Argentina", "United States", 1, 1, True),
        (3, "2000-04-01", "Brazil", "Argentina", 0, 0, False),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "neutral",
        ],
    )
    df["tournament"] = "Friendly"
    df.to_csv(path, index=False)
    return len(df)


def _write_elo(path: Path) -> None:
    pd.DataFrame(
        {
            "team": ["Brazil", "Argentina", "USA"],  # "USA" exercises normalization
            "date": ["1999-12-01", "1999-12-01", "1999-12-01"],
            "elo": [2100, 1950, 1600],
        }
    ).to_csv(path, index=False)


def _write_fifa(path: Path) -> None:
    pd.DataFrame(
        {
            "team": ["Brazil", "Argentina", "USA"],
            "rank_date": ["1999-11-01", "1999-11-01", "1999-11-01"],
            "rank": [1, 3, 22],
            "points": [1500, 1400, 980],
        }
    ).to_csv(path, index=False)


# --- full pipeline through the CLI ------------------------------------------


def test_pipeline_with_ratings(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    matches_csv = tmp_path / "matches.csv"
    elo_csv = tmp_path / "elo_ratings.csv"
    fifa_csv = tmp_path / "fifa_rankings.csv"
    out_csv = tmp_path / "features.csv"
    n = _write_matches(matches_csv)
    _write_elo(elo_csv)
    _write_fifa(fifa_csv)

    rc = cli.main(
        [
            "--matches",
            str(matches_csv),
            "--elo",
            str(elo_csv),
            "--fifa",
            str(fifa_csv),
            "--output",
            str(out_csv),
        ]
    )
    assert rc == 0
    assert out_csv.exists()

    feats = pd.read_csv(out_csv)
    assert len(feats) == n
    for col in (*ROLLING_FEATURE_COLUMNS, *RATING_FEATURES):
        assert col in feats.columns
    # match 1 (Brazil vs USA): USA's Elo joined via name normalization.
    assert feats.set_index("match_id").loc[1, "team_b_elo"] == 1600


def test_pipeline_without_ratings_warns_and_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    matches_csv = tmp_path / "matches.csv"
    out_csv = tmp_path / "features.csv"
    n = _write_matches(matches_csv)

    # Point --elo/--fifa at files that do not exist: must warn, not crash.
    with caplog.at_level("WARNING"):
        rc = cli.main(
            [
                "--matches",
                str(matches_csv),
                "--elo",
                str(tmp_path / "nope_elo.csv"),
                "--fifa",
                str(tmp_path / "nope_fifa.csv"),
                "--output",
                str(out_csv),
            ]
        )
    assert rc == 0
    feats = pd.read_csv(out_csv)
    assert len(feats) == n
    # Rolling features present; Elo is attached from the leakage-safe walk-forward
    # even with no external snapshot (it is the primary strength signal). FIFA
    # features still require a source file, so they stay absent.
    assert "form_5_diff" in feats.columns
    for col in ELO_FEATURES:
        assert col in feats.columns
    for col in FIFA_FEATURES:
        assert col not in feats.columns
    assert any("not found" in m for m in caplog.messages)


def test_pipeline_attaches_leakage_safe_walk_forward_elo(tmp_path: Path) -> None:
    # With no external Elo file the pipeline attaches a walk-forward Elo. The very
    # first match (both teams unseen) starts level, so elo_diff == 0 — proof the
    # rating reflects only prior matches, never the match's own result.
    matches_csv = tmp_path / "matches.csv"
    out_csv = tmp_path / "features.csv"
    _write_matches(matches_csv)

    rc = cli.main(
        [
            "--matches",
            str(matches_csv),
            "--elo",
            str(tmp_path / "nope_elo.csv"),
            "--fifa",
            str(tmp_path / "nope_fifa.csv"),
            "--output",
            str(out_csv),
        ]
    )
    assert rc == 0

    feats = pd.read_csv(out_csv).set_index("match_id").sort_index()
    assert {"team_a_elo", "team_b_elo", "elo_diff"} <= set(feats.columns)
    # match 0 is the earliest fixture; both teams enter at the base rating.
    assert feats.loc[0, "elo_diff"] == 0.0
    # Later matches must carry a real Elo signal once results have accrued.
    assert float(feats["elo_diff"].abs().sum()) > 0.0
    for col in RATING_FEATURES:
        if col in ELO_FEATURES:
            assert col in feats.columns


def test_missing_matches_file_returns_error_code(tmp_path: Path) -> None:
    rc = cli.main(["--matches", str(tmp_path / "absent.csv"), "--output", str(tmp_path / "f.csv")])
    assert rc == 1


def test_sample_mode_limits_rows(tmp_path: Path) -> None:
    matches_csv = tmp_path / "matches.csv"
    out_csv = tmp_path / "features.csv"
    _write_matches(matches_csv)

    rc = cli.main(["--matches", str(matches_csv), "--sample", "2", "--output", str(out_csv)])
    assert rc == 0
    assert len(pd.read_csv(out_csv)) == 2


# --- determinism ------------------------------------------------------------


def test_output_is_deterministic(tmp_path: Path) -> None:
    matches_csv = tmp_path / "matches.csv"
    _write_matches(matches_csv)
    out1 = tmp_path / "f1.csv"
    out2 = tmp_path / "f2.csv"

    cli.main(["--matches", str(matches_csv), "--output", str(out1)])
    cli.main(["--matches", str(matches_csv), "--output", str(out2)])

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


# --- validation -------------------------------------------------------------


def test_validate_feature_matrix_rejects_row_mismatch(tmp_path: Path) -> None:
    matches_csv = tmp_path / "matches.csv"
    _write_matches(matches_csv)
    matches = load_matches(matches_csv)
    feats = cli.build_feature_matrix(matches)

    from worldcup.data.validate_data import DataValidationError

    with pytest.raises(DataValidationError):
        validate_feature_matrix(feats, expected_rows=len(matches) + 1)


def test_no_label_leakage_into_rolling_features() -> None:
    # A future blowout must not change an earlier match's features (the leakage
    # contract the whole pipeline rests on), re-checked at the table level.
    from worldcup.features.build_features import build_feature_matrix

    base = pd.DataFrame(
        {
            "match_id": [0, 1, 2],
            "date": pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01"]),
            "home_team": ["A", "A", "A"],
            "away_team": ["X", "Y", "Z"],
            "home_score": [1, 2, 3],
            "away_score": [0, 0, 0],
            "neutral": [False, False, False],
            "tournament": ["Friendly"] * 3,
        }
    )
    future = pd.DataFrame(
        {
            "match_id": [3],
            "date": pd.to_datetime(["2000-04-01"]),
            "home_team": ["A"],
            "away_team": ["W"],
            "home_score": [100],
            "away_score": [0],
            "neutral": [False],
            "tournament": ["Friendly"],
        }
    )
    with_future = build_feature_matrix(pd.concat([base, future], ignore_index=True))
    without = build_feature_matrix(base)

    cols = [c for c in without.columns if c in ROLLING_FEATURE_COLUMNS]
    earlier = with_future[with_future["match_id"] < 3].set_index("match_id")
    pd.testing.assert_frame_equal(
        earlier[cols], without.set_index("match_id")[cols], check_like=True
    )


def test_label_columns_present_but_distinct_from_features() -> None:
    matches = pd.DataFrame(
        {
            "match_id": [0],
            "date": pd.to_datetime(["2000-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [1],
            "away_score": [0],
            "neutral": [False],
            "tournament": ["Friendly"],
        }
    )
    feats = cli.build_feature_matrix(matches)
    for label in LABEL_COLUMNS:
        assert label in feats.columns
        assert label not in ROLLING_FEATURE_COLUMNS
