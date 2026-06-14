"""Canonicalize international team names across data sources.

results.csv (martj42) already uses clean English names — "United States",
"South Korea", "Ivory Coast" — and is treated as the CANONICAL spelling. Other
sources (FIFA ranking, Elo) differ ("USA", "Korea Republic", "Côte d'Ivoire").
This module maps known aliases to the canonical name so a join never silently
splits one nation into several rows.

Why two steps:
    * :func:`normalize_team_name` cleans and maps aliases but never *guesses* — a
      name that is already canonical (or simply unknown) passes through cleaned.
    * Catching a *forgotten* alias needs a reference set of valid names.
      :func:`find_unknown_teams` reports unmapped names, and
      :func:`normalize_team_columns` with ``known_teams`` raises
      :class:`UnknownTeamError` listing them — so the gap fails loudly instead of
      corrupting a downstream join.

Matching is case-, accent-, and apostrophe-insensitive, so "Côte d'Ivoire",
"Côte d’Ivoire", and "cote d'ivoire" all resolve through one map entry.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Iterable, Sequence
from functools import lru_cache

import pandas as pd
import yaml

from worldcup.config import team_name_map_path

__all__ = [
    "UnknownTeamError",
    "find_unknown_teams",
    "normalize_team_columns",
    "normalize_team_name",
]

# Curly / modifier apostrophes normalized to a straight quote before matching.
_APOSTROPHES = {"’": "'", "‘": "'", "ʼ": "'", "`": "'"}


class UnknownTeamError(ValueError):
    """Raised when team names cannot be matched to the known canonical set."""


def _replace_apostrophes(text: str) -> str:
    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)
    return text


def _fold(name: str) -> str:
    """Return an accent/case/apostrophe-insensitive key, used only for matching."""
    text = _replace_apostrophes(name)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split()).casefold()


def _clean(name: str) -> str:
    """Collapse whitespace and normalize apostrophes, preserving case and accents."""
    return " ".join(_replace_apostrophes(name).split())


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    """Load and index the alias map by folded key (cached).

    Every alias *and* every canonical name maps to the canonical name, so
    canonical inputs are stable and aliases resolve regardless of accent or case.
    """
    path = team_name_map_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases = data.get("aliases", {}) or {}
    index: dict[str, str] = {}
    for alias, canonical in aliases.items():
        canonical_str = str(canonical)
        index[_fold(str(alias))] = canonical_str
        index.setdefault(_fold(canonical_str), canonical_str)
    return index


def normalize_team_name(name: str) -> str:
    """Map a raw team name to its canonical (results.csv) spelling.

    Cleans whitespace/apostrophes and applies the alias map. A name that is
    already canonical (or unknown) passes through cleaned — this function never
    guesses a mapping. Use :func:`find_unknown_teams` or
    :func:`normalize_team_columns` with ``known_teams`` to catch unmapped names.

    Args:
        name: Raw team name.

    Returns:
        The canonical team name.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is empty after cleaning.
    """
    if not isinstance(name, str):
        raise TypeError(f"team name must be a str, got {type(name).__name__}")
    cleaned = _clean(name)
    if not cleaned:
        raise ValueError("team name is empty after cleaning")
    return _alias_index().get(_fold(cleaned), cleaned)


def normalize_team_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    known_teams: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the given team ``columns`` normalized.

    Args:
        df: Input DataFrame (not mutated).
        columns: Team-name columns to normalize.
        known_teams: If provided, every normalized name must appear in this set,
            otherwise an :class:`UnknownTeamError` is raised listing the offenders.

    Returns:
        A copy of ``df`` with the named columns canonicalized.

    Raises:
        KeyError: If any requested column is missing.
        UnknownTeamError: If a value is non-string/empty, or — when ``known_teams``
            is given — any normalized name is unrecognized.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"columns not found in DataFrame: {missing}")

    out = df.copy()
    for col in columns:
        normalized: list[str] = []
        for value in out[col].tolist():
            if not isinstance(value, str) or not value.strip():
                raise UnknownTeamError(
                    f"column '{col}' contains a non-string/empty team value: {value!r}"
                )
            normalized.append(normalize_team_name(value))
        out[col] = normalized

    if known_teams is not None:
        unknown = find_unknown_teams(out, columns, known_teams)
        if unknown:
            raise UnknownTeamError(_unknown_message(unknown, known_teams))
    return out


def find_unknown_teams(
    df: pd.DataFrame,
    columns: Sequence[str],
    known_teams: Iterable[str],
) -> list[str]:
    """Return sorted normalized names in ``columns`` not in ``known_teams``.

    Names are normalized defensively, so this works on raw *or* already-normalized
    data. Non-raising — intended for auditing before a strict load.

    Args:
        df: Input DataFrame.
        columns: Team-name columns to scan.
        known_teams: Reference set of valid canonical names.

    Returns:
        Sorted list of unrecognized canonical names (empty if all are known).

    Raises:
        KeyError: If any requested column is missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"columns not found in DataFrame: {missing}")

    known = set(known_teams)
    unknown: set[str] = set()
    for col in columns:
        for value in df[col].dropna().unique():
            if not isinstance(value, str):
                continue
            canonical = normalize_team_name(value)
            if canonical not in known:
                unknown.add(canonical)
    return sorted(unknown)


def _unknown_message(unknown: Sequence[str], known_teams: Iterable[str]) -> str:
    """Build a helpful error listing unmapped names and their closest known match."""
    known_list = list(known_teams)
    lines = [f"{len(unknown)} team name(s) not in the known canonical set:"]
    for name in unknown:
        match = difflib.get_close_matches(name, known_list, n=1, cutoff=0.6)
        hint = f"  (closest known: '{match[0]}')" if match else ""
        lines.append(f"  - {name!r}{hint}")
    lines.append(
        "Fix by adding an alias to configs/team_name_map.yaml under 'aliases:', "
        "or correct the source data."
    )
    return "\n".join(lines)
