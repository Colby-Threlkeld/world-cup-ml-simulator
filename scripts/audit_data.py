"""Lightweight raw-data audit for world-cup-ml-simulator.

Reads every CSV in a directory (default ``data/raw/``) and prints a quick
profile of each: row count, columns and dtypes, missing-value counts, duplicate
rows, date range, top teams, and a few sample rows.

This is a *throwaway understanding tool*, deliberately decoupled from the
``worldcup`` package so it runs before anything is installed. It does not clean
or write data.

Run::

    python scripts/audit_data.py
    python scripts/audit_data.py --dir path/to/other/csvs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# scripts/ lives at the repo root, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

SAMPLE_ROWS = 5
TOP_TEAMS = 20
RULE = "=" * 72


def _columns_containing(df: pd.DataFrame, token: str) -> list[str]:
    """Return columns whose name contains ``token`` (case-insensitive)."""
    return [col for col in df.columns if token in col.lower()]


def audit_file(path: Path) -> None:
    """Print an audit profile for a single CSV file.

    Args:
        path: Path to the CSV file.
    """
    print(f"\n{RULE}\nFILE: {path.name}\n{RULE}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - an audit must never abort the run
        print(f"  !! could not read: {exc}")
        return

    n_rows, n_cols = df.shape
    print(f"rows: {n_rows:,}   columns: {n_cols}")

    print("\ncolumns & dtypes:")
    for name, dtype in df.dtypes.items():
        print(f"  {str(name):<22} {dtype}")

    print("\nmissing values (columns with > 0):")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  none")
    else:
        for name, count in missing.items():
            print(f"  {str(name):<22} {count:,} ({count / n_rows:.1%})")

    print(f"\nduplicate rows (full row): {df.duplicated().sum():,}")

    for col in _columns_containing(df, "date"):
        parsed = pd.to_datetime(df[col], errors="coerce")
        unparseable = int(parsed.isna().sum() - df[col].isna().sum())
        suffix = f"   (unparseable: {unparseable})" if unparseable > 0 else ""
        print(f"\ndate range [{col}]: {parsed.min()} -> {parsed.max()}{suffix}")

    team_cols = _columns_containing(df, "team")
    if team_cols:
        teams = pd.concat([df[col] for col in team_cols], ignore_index=True).dropna()
        print(f"\ntop {TOP_TEAMS} teams across {team_cols} (by appearances):")
        for name, count in teams.value_counts().head(TOP_TEAMS).items():
            print(f"  {str(name):<30} {count:,}")

    print(f"\nsample rows (first {SAMPLE_ROWS}):")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.head(SAMPLE_ROWS).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    """Audit every CSV in the target directory.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code: ``0`` on success, ``1`` if nothing was audited.
    """
    parser = argparse.ArgumentParser(description="Audit raw CSV files in a directory.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"directory of CSV files to audit (default: {DEFAULT_RAW_DIR})",
    )
    args = parser.parse_args(argv)
    raw_dir: Path = args.dir

    if not raw_dir.exists():
        print(f"directory not found: {raw_dir}")
        return 1

    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        print(f"no CSV files found in {raw_dir}. Download datasets first (see README).")
        return 1

    print(f"Auditing {len(csv_paths)} CSV file(s) in {raw_dir}")
    for path in csv_paths:
        audit_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
