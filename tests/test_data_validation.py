"""Tests for match-table validation and result labeling."""

import pandas as pd
import pytest

from worldcup.data.clean_data import add_result_label, match_result
from worldcup.data.validate_data import DataValidationError, validate_matches


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-11-20", "2022-11-21"]),
            "home_team": ["Qatar", "England"],
            "away_team": ["Ecuador", "Iran"],
            "home_score": [0, 6],
            "away_score": [2, 2],
        }
    )


def test_valid_table_passes():
    validate_matches(_valid_df())  # should not raise


def test_missing_column_raises():
    df = _valid_df().drop(columns=["away_score"])
    with pytest.raises(DataValidationError):
        validate_matches(df)


def test_negative_score_raises():
    df = _valid_df()
    df.loc[0, "home_score"] = -1
    with pytest.raises(DataValidationError):
        validate_matches(df)


def test_null_in_required_column_raises():
    df = _valid_df()
    df.loc[0, "home_team"] = None
    with pytest.raises(DataValidationError):
        validate_matches(df)


def test_match_result_and_label():
    assert match_result(2, 0) == "H"
    assert match_result(1, 1) == "D"
    assert match_result(0, 3) == "A"
    labeled = add_result_label(_valid_df())
    assert list(labeled["result"]) == ["A", "H"]
