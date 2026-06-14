"""Canonicalize international team names across data sources.

Different sources spell the same nation differently ("USA" vs "United States",
"Korea Republic" vs "South Korea"). A single canonical name is required before
any join, otherwise the same team splits into multiple rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from worldcup.config import team_name_map_path


def load_team_name_map(path: Path | None = None) -> dict[str, str]:
    """Load the alias -> canonical team-name mapping from YAML.

    Args:
        path: Path to the YAML map. Defaults to the project config.

    Returns:
        Mapping of alias names to canonical names (empty if none defined).
    """
    yaml_path = path or team_name_map_path()
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("aliases", {}))


def normalize_team_name(name: str, alias_map: Mapping[str, str]) -> str:
    """Map a raw team name to its canonical form.

    Whitespace is trimmed and collapsed, then the alias map is consulted
    (exact match first, then case-insensitive). Names not in the map pass
    through cleaned but otherwise unchanged.

    Args:
        name: Raw team name from a data source.
        alias_map: Alias -> canonical mapping from :func:`load_team_name_map`.

    Returns:
        The canonical team name.
    """
    cleaned = " ".join(name.strip().split())
    if cleaned in alias_map:
        return alias_map[cleaned]
    lowered = {alias.lower(): canonical for alias, canonical in alias_map.items()}
    return lowered.get(cleaned.lower(), cleaned)
