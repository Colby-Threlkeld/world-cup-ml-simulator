"""Tests for raw match loading and cleaning (slice 1 ingestion).

Uses a tiny fixture CSV — never the full real dataset — so the suite stays fast
and hermetic. A skipped smoke test exercises the real file when it is present.
"""

from pathlib import Path

import pandas as pd
import pytest

from worldcup.data.clean_data import (
    DEFAULT_MATCHES_PATH,
    MATCHES_COLUMNS,
    clean_matches,
    save_matches,
)
from worldcup.data.load_data import load_raw_matches
from worldcup.data.validate_data import DataValidationError

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "raw_matches_sample.csv"


# --- load -------------------------------------------------------------------


def test_load_raw_matches_reads_fixture() -> None:
    df = load_raw_matches(SAMPLE)
    assert len(df) == 6
    assert {"date", "home_team", "away_team", "home_score", "away_score"}.issubset(df.columns)


def test_load_raw_matches_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_matches(tmp_path / "nope.csv")


def test_load_raw_matches_missing_columns_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("date,home_team,away_team\n2020-01-01,A,B\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_raw_matches(bad)


# --- clean ------------------------------------------------------------------


@pytest.fixture
def cleaned() -> pd.DataFrame:
    return clean_matches(load_raw_matches(SAMPLE))


def test_clean_drops_dup_and_unplayed(cleaned: pd.DataFrame) -> None:
    # 6 raw -> drop 1 exact dup -> 5 -> drop 1 unplayed (2026 fixture) -> 4 played.
    assert len(cleaned) == 4


def test_clean_columns_match_schema(cleaned: pd.DataFrame) -> None:
    assert list(cleaned.columns) == list(MATCHES_COLUMNS)
    assert "city" not in cleaned.columns
    assert "country" not in cleaned.columns
    assert "venue_city" in cleaned.columns


def test_clean_normalizes_team_names(cleaned: pd.DataFrame) -> None:
    teams = set(cleaned["home_team"]) | set(cleaned["away_team"])
    assert {"Saudi Arabia", "Iran", "Netherlands", "Turkey"} <= teams
    assert not ({"Saudi", "IR Iran", "Holland", "Türkiye"} & teams)


def test_clean_result_labels_and_derived(cleaned: pd.DataFrame) -> None:
    rows = cleaned.set_index(["home_team", "away_team"])

    netherlands = rows.loc[("Netherlands", "Turkey")]
    assert netherlands["result"] == "D"
    assert bool(netherlands["draw"])
    assert not bool(netherlands["home_win"])
    assert not bool(netherlands["away_win"])
    assert netherlands["tournament"] == "Friendly"  # trailing space was stripped
    assert not bool(netherlands["is_competitive"])
    assert not bool(netherlands["neutral"])

    russia = rows.loc[("Russia", "Saudi Arabia")]
    assert russia["result"] == "H"
    assert bool(russia["home_win"])
    assert bool(russia["is_competitive"])
    assert russia["total_goals"] == 5
    assert russia["goal_diff"] == 5


def test_match_id_is_unique_and_date_sorted(cleaned: pd.DataFrame) -> None:
    assert cleaned["match_id"].is_unique
    assert list(cleaned["date"]) == sorted(cleaned["date"])


def test_clean_dtypes(cleaned: pd.DataFrame) -> None:
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert str(cleaned["home_score"].dtype) == "int16"
    assert cleaned["neutral"].dtype == bool
    assert str(cleaned["result"].dtype) == "category"


def test_clean_does_not_mutate_input() -> None:
    raw = load_raw_matches(SAMPLE)
    before = raw.copy(deep=True)
    clean_matches(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_negative_score_raises() -> None:
    raw = load_raw_matches(SAMPLE)
    raw.loc[0, "home_score"] = -1
    with pytest.raises(DataValidationError):
        clean_matches(raw)


def test_clean_unparseable_date_raises() -> None:
    raw = load_raw_matches(SAMPLE)
    raw.loc[0, "date"] = "not-a-date"
    with pytest.raises(DataValidationError):
        clean_matches(raw)


def test_clean_empty_raises() -> None:
    with pytest.raises(DataValidationError):
        clean_matches(pd.DataFrame(columns=list(MATCHES_COLUMNS)))


# --- save -------------------------------------------------------------------


def test_save_matches_round_trip(tmp_path: Path, cleaned: pd.DataFrame) -> None:
    out = save_matches(cleaned, tmp_path / "matches.parquet")
    assert out.exists()
    reloaded = pd.read_parquet(out)
    assert len(reloaded) == len(cleaned)
    assert list(reloaded.columns) == list(MATCHES_COLUMNS)


def test_default_matches_path_points_to_interim() -> None:
    assert DEFAULT_MATCHES_PATH.name == "matches.parquet"
    assert DEFAULT_MATCHES_PATH.parent.name == "interim"


# --- optional real-data smoke (skipped when the gitignored file is absent) ---

REAL_RESULTS = Path(__file__).resolve().parents[1] / "data" / "raw" / "results.csv"


@pytest.mark.skipif(not REAL_RESULTS.exists(), reason="real results.csv not present (gitignored)")
def test_clean_real_results_smoke() -> None:
    cleaned = clean_matches(load_raw_matches(REAL_RESULTS))
    assert len(cleaned) > 30_000
    assert cleaned["match_id"].is_unique
    assert int(cleaned["home_score"].min()) >= 0
