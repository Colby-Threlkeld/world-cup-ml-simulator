"""Tests for leakage-safe rolling features."""

import numpy as np
import pandas as pd
import pytest

from worldcup.features.build_features import build_feature_matrix
from worldcup.features.rolling_features import add_rolling_mean


def test_rolling_mean_excludes_current_row():
    """The current match must never contribute to its own feature (no leakage)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "team": ["A", "A", "A"],
            "goals": [1, 2, 3],
        }
    )
    out = add_rolling_mean(df, "team", "goals", window=5, out_col="form")
    out = out.sort_values("date").reset_index(drop=True)

    assert np.isnan(out.loc[0, "form"])   # no prior history
    assert out.loc[1, "form"] == 1.0       # mean of [1]
    assert out.loc[2, "form"] == 1.5       # mean of [1, 2] -- excludes the current 3


def test_rolling_mean_is_computed_per_group():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
            "team": ["A", "B", "A", "B"],
            "goals": [1, 5, 3, 7],
        }
    )
    out = add_rolling_mean(df, "team", "goals", window=5, out_col="form").sort_values("date")
    a_second = out[out["team"] == "A"].iloc[1]
    b_second = out[out["team"] == "B"].iloc[1]
    assert a_second["form"] == 1.0   # only A's first match
    assert b_second["form"] == 5.0   # only B's first match


def test_build_feature_matrix_not_yet_implemented():
    """Documents the slice-3 contract; delete when implemented."""
    with pytest.raises(NotImplementedError):
        build_feature_matrix(pd.DataFrame())
