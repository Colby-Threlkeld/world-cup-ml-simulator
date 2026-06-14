"""Assemble the leakage-safe match feature matrix -- the single join point.

This module is the *only* place features are joined to labels, so the
as-of-date contract lives here: every feature must be computed from information
available strictly before kickoff.
"""

from __future__ import annotations

import pandas as pd


def build_feature_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature matrix from cleaned matches.

    TODO(slice 3): join Elo ratings, rolling form, rest days, FIFA ranking, and
    the neutral-venue flag -- each computed as-of the match date -- to the H/D/A
    (and goals) labels. No post-kickoff information may enter this table.

    Args:
        matches: Cleaned match table from :mod:`worldcup.data.clean_data`.

    Returns:
        One row per match with features and labels, free of leakage.
    """
    raise NotImplementedError("build_feature_matrix is implemented in slice 3.")
