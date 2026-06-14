"""World Cup ML Simulator — Streamlit front-end.

A read-only dashboard over artifacts produced by the pipeline scripts. It never
trains a model: it loads saved outputs (simulation probabilities, evaluation and
backtest metrics, figures) and computes only a cheap, cached Elo rating table for
the interactive Team Explorer and Match Predictor. When an artifact is missing it
shows the exact command to generate it rather than any fabricated numbers.

Run locally::

    make app                       # or: streamlit run app/streamlit_app.py

Run on a headless Azure VM (then SSH-tunnel port 8501 — see the README)::

    streamlit run app/streamlit_app.py --server.headless true \
        --server.address 127.0.0.1 --server.port 8501
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the src/ package importable when run via `streamlit run`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from worldcup.config import INTERIM_DIR, REPORTS_DIR  # noqa: E402
from worldcup.features.build_features import build_model_dataset  # noqa: E402
from worldcup.simulation.tournament import strength_predict_fn  # noqa: E402
from worldcup.visualization.plots import plot_title_probabilities  # noqa: E402

st.set_page_config(page_title="World Cup ML Simulator", page_icon="⚽", layout="wide")

MATCHES_PATH = INTERIM_DIR / "matches.parquet"
SIM_DIR = REPORTS_DIR / "simulation"
HOME_ADVANTAGE_ELO = 65.0
CLASS_LABELS = {"team_a_win": "Team A win", "draw": "Draw", "team_b_win": "Team B win"}

ADVANCEMENT_COLUMNS = {
    "win_group_probability": "Win group",
    "reach_round_16_probability": "Reach R16",
    "reach_quarterfinal_probability": "Reach QF",
    "reach_semifinal_probability": "Reach SF",
    "reach_final_probability": "Reach final",
    "win_world_cup_probability": "Win title",
}


# --- cached loaders (return None when an artifact is absent) -----------------


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_text(path_str: str) -> str | None:
    path = Path(path_str)
    return path.read_text(encoding="utf-8") if path.exists() else None


@st.cache_data(show_spinner="Computing Elo ratings from match history…")
def load_elo_table() -> pd.DataFrame | None:
    """Cheap, cached current-Elo table from the cleaned matches (no model training)."""
    if not MATCHES_PATH.exists():
        return None
    from worldcup.backtesting import current_elo_ratings  # local import: heavy deps

    model_df = build_model_dataset(pd.read_parquet(MATCHES_PATH))
    ratings = current_elo_ratings(model_df)
    table = (
        pd.DataFrame({"team": list(ratings), "elo": list(ratings.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    table.insert(0, "rank", table.index + 1)
    return table


def missing_artifact(message: str, command: str) -> None:
    """Friendly 'nothing to show yet' panel with the command that produces it."""
    st.info(f"📭 {message}")
    st.caption("Generate it with:")
    st.code(command, language="bash")


def uncertainty_note() -> None:
    st.caption(
        "ℹ️ These are **probabilities, not predictions of certainty**. Football is "
        "high-variance; a 20% title shot is genuinely uncertain. Monte-Carlo figures "
        "carry sampling noise (more simulations → tighter estimates), and the model is "
        "judged by calibration — whether stated probabilities match reality — not by "
        "calling single matches right."
    )


# --- pages ------------------------------------------------------------------


def page_overview() -> None:
    st.title("⚽ World Cup ML Simulator")
    st.markdown(
        "A portfolio ML system that forecasts international football matches from "
        "historical data and simulates the **2026 FIFA World Cup** (48 teams, 12 "
        "groups) with Monte Carlo to estimate title odds — judged against honest "
        "baselines and backtested on past tournaments."
    )
    uncertainty_note()

    st.subheader("What's available right now")
    checks = {
        "Cleaned match history": MATCHES_PATH.exists(),
        "Tournament simulation": (SIM_DIR / "tournament_probabilities.csv").exists(),
        "Model evaluation": (REPORTS_DIR / "metrics" / "model_metrics.json").exists(),
        "Backtesting results": (REPORTS_DIR / "backtesting" / "backtest_metrics.json").exists(),
    }
    cols = st.columns(len(checks))
    for col, (label, ready) in zip(cols, checks.items(), strict=True):
        col.metric(label, "Ready ✅" if ready else "Missing ⬜")

    summary = load_json(str(SIM_DIR / "tournament_summary.json"))
    probs = load_csv(str(SIM_DIR / "tournament_probabilities.csv"))
    st.subheader("Title contenders (latest simulation)")
    if summary is None or probs is None:
        missing_artifact(
            "No tournament simulation found yet.",
            "python scripts/run_simulation.py --quick",
        )
        return
    if summary.get("draw_status") == "placeholder":
        st.warning(
            "The official 2026 group draw isn't encoded yet, so teams appear as "
            "**placeholder slots** (A1…L4). These odds reflect bracket structure, not a "
            "team-specific forecast. Re-run the simulation with a real strengths table "
            "for a true forecast."
        )
    st.pyplot(plot_title_probabilities(probs, top_n=12))


def page_team_explorer() -> None:
    st.title("Team Explorer")
    st.caption("Current self-computed **Elo** strength from the full match history.")
    elo = load_elo_table()
    if elo is None:
        missing_artifact(
            "No match history found, so Elo ratings can't be computed.",
            "python scripts/build_matches.py",
        )
        return

    team = st.selectbox("Team", elo["team"], index=0)
    row = elo.loc[elo["team"] == team].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Elo rating", f"{row['elo']:.0f}")
    c2.metric("World rank", f"#{int(row['rank'])} of {len(elo)}")
    c3.metric("Percentile", f"{100 * (1 - (row['rank'] - 1) / len(elo)):.0f}th")

    st.subheader("Top 20 by Elo")
    top = elo.head(20).set_index("team")["elo"]
    st.bar_chart(top)
    with st.expander("Full rating table"):
        st.dataframe(elo, width="stretch", hide_index=True)
    st.caption(
        "Elo is updated chronologically (a rating only ever reflects prior matches), "
        "so it is leakage-safe. It is a strength signal, not the full match model."
    )


def page_match_predictor() -> None:
    st.title("Match Predictor")
    st.caption("Win / draw / loss probabilities from the Elo strength difference.")
    elo = load_elo_table()
    if elo is None:
        missing_artifact(
            "No match history found, so match probabilities can't be computed.",
            "python scripts/build_matches.py",
        )
        return

    teams = list(elo["team"])
    strengths = dict(zip(elo["team"], elo["elo"], strict=True))
    c1, c2 = st.columns(2)
    team_a = c1.selectbox("Team A", teams, index=0)
    team_b = c2.selectbox("Team B", teams, index=min(1, len(teams) - 1))
    neutral = st.checkbox("Neutral venue", value=True)

    if team_a == team_b:
        st.warning("Pick two different teams.")
        return

    adjusted = dict(strengths)
    if not neutral:
        adjusted[team_a] = adjusted.get(team_a, 1500.0) + HOME_ADVANTAGE_ELO
    predict = strength_predict_fn(adjusted)
    p_a, p_draw, p_b = predict(team_a, team_b)

    cols = st.columns(3)
    cols[0].metric(f"{team_a} win", f"{p_a:.1%}")
    cols[1].metric("Draw", f"{p_draw:.1%}")
    cols[2].metric(f"{team_b} win", f"{p_b:.1%}")
    chart = pd.DataFrame(
        {"probability": [p_a, p_draw, p_b]},
        index=[f"{team_a} win", "Draw", f"{team_b} win"],
    )
    st.bar_chart(chart)
    uncertainty_note()
    st.caption(
        f"Elo: **{team_a}** {strengths.get(team_a, 1500):.0f} vs **{team_b}** "
        f"{strengths.get(team_b, 1500):.0f}"
        + ("" if neutral else f" (+{HOME_ADVANTAGE_ELO:.0f} home advantage to {team_a})")
        + ". Draw rate is a fixed international base rate; this Elo predictor is the "
        "simulator's strength model, not the calibrated match model."
    )


def page_tournament_simulator() -> None:
    st.title("Tournament Simulator")
    st.caption("Monte-Carlo 2026 World Cup outcomes (loaded from a saved run).")
    probs = load_csv(str(SIM_DIR / "tournament_probabilities.csv"))
    summary = load_json(str(SIM_DIR / "tournament_summary.json"))
    if probs is None:
        missing_artifact(
            "No tournament simulation output found.",
            "python scripts/run_simulation.py --quick   # ~3s, 1,000 sims",
        )
        return

    if summary is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Simulations", f"{summary.get('n_simulations', 0):,}")
        c2.metric("Teams", summary.get("n_teams", len(probs)))
        c3.metric("Seed", summary.get("seed", "—"))
        if summary.get("draw_status") == "placeholder":
            st.warning(
                "Teams are **placeholder slots** (the official draw isn't encoded yet), "
                "so these odds reflect bracket structure only — not a team forecast."
            )

    st.subheader("Title probabilities")
    st.pyplot(plot_title_probabilities(probs, top_n=15))

    st.subheader("Advancement probabilities by team")
    available = {c: label for c, label in ADVANCEMENT_COLUMNS.items() if c in probs.columns}
    view = probs[["team", *available]].rename(columns=available)
    view = view.sort_values("Win title", ascending=False) if "Win title" in view else view
    st.dataframe(
        view.style.format({label: "{:.1%}" for label in available.values()}),
        width="stretch",
        hide_index=True,
    )
    uncertainty_note()


def page_backtesting() -> None:
    st.title("Backtesting")
    st.caption("How the system would have done before the 2014 / 2018 / 2022 World Cups.")
    report = load_json(str(REPORTS_DIR / "backtesting" / "backtest_metrics.json"))
    if report is None:
        missing_artifact(
            "No backtesting results found.",
            "python scripts/run_backtest.py",
        )
        return

    agg = report.get("aggregate", {})
    cols = st.columns(4)
    cols[0].metric("Mean log loss", agg.get("mean_log_loss", "—"))
    cols[1].metric("Mean Brier", agg.get("mean_brier", "—"))
    cols[2].metric("Mean accuracy", _pct(agg.get("mean_accuracy")))
    cols[3].metric("Champion top-5 rate", _pct(agg.get("champion_top_5_rate")))
    st.caption("Reference: an uninformed 1/3-each model scores log loss ≈ 1.099 (lower is better).")

    tdf = pd.DataFrame(report.get("tournaments", []))
    st.subheader("Match-prediction accuracy")
    acc_cols = ["tournament", "n_test", "log_loss", "brier", "accuracy", "calibration_error"]
    st.dataframe(tdf[[c for c in acc_cols if c in tdf]], width="stretch", hide_index=True)

    st.subheader("Did we fancy the eventual champion?")
    win_cols = [
        "tournament",
        "champion",
        "champion_predicted_rank",
        "n_participants",
        "champion_in_top_3",
        "champion_in_top_5",
        "champion_in_top_10",
    ]
    st.dataframe(tdf[[c for c in win_cols if c in tdf]], width="stretch", hide_index=True)
    st.caption(
        "Rank is the champion's place in the pre-tournament Elo favourite ordering "
        "(32 teams) — a proxy for a full winner-probability simulation, which would "
        "need each year's official bracket (a documented TODO)."
    )


def page_methodology() -> None:
    st.title("Methodology")
    st.markdown(
        "- **Data:** martj42 international results (1872–present), canonicalized team "
        "names, ~49k played matches.\n"
        "- **Features:** leakage-safe, computed *as of kickoff* — rolling form "
        "(last 5/10 points, goals for/against, goal diff), rest days, and a "
        "walk-forward **Elo** difference. Labels never leak into features.\n"
        "- **Model:** a calibrated multinomial-logistic 3-class outcome model, trained "
        "on a strictly temporal split (never random K-fold).\n"
        "- **Simulation:** Monte Carlo of the 48-team format — group round-robins with "
        "FIFA's *primary* tiebreakers (points, goal difference, goals scored; "
        "head-to-head and fair-play are deferred to a seeded random draw), then a "
        "single-elimination bracket with draws resolved by redistributing draw "
        "probability.\n"
        "- **Honesty:** no fabricated numbers; the app loads saved outputs only."
    )

    metrics = load_json(str(REPORTS_DIR / "metrics" / "model_metrics.json"))
    if metrics is not None:
        cmp = metrics.get("comparison", {})
        c1, c2 = st.columns(2)
        c1.metric("Model test log loss", round(cmp.get("main_calibrated_test_log_loss", 0), 4))
        c2.metric("Best baseline log loss", round(cmp.get("best_baseline_test_log_loss", 0), 4))
        st.write("**Model features:** " + ", ".join(metrics.get("features", [])))
    else:
        missing_artifact(
            "Model evaluation metrics not found.",
            "python scripts/train_model.py && python scripts/generate_evaluation_report.py",
        )

    fig_path = REPORTS_DIR / "figures" / "calibration_curve.png"
    if fig_path.exists():
        st.subheader("Calibration")
        st.image(str(fig_path), caption="Reliability diagram — closer to the diagonal is better.")

    report_md = load_text(str(REPORTS_DIR / "evaluation_report.md"))
    if report_md:
        with st.expander("Full evaluation report"):
            st.markdown(report_md)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


PAGES = {
    "Overview": page_overview,
    "Team Explorer": page_team_explorer,
    "Match Predictor": page_match_predictor,
    "Tournament Simulator": page_tournament_simulator,
    "Backtesting": page_backtesting,
    "Methodology": page_methodology,
}


def main() -> None:
    st.sidebar.title("⚽ Navigation")
    choice = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Read-only dashboard. Nothing is trained here — it loads saved pipeline "
        "outputs and a cached Elo table. Missing pages show the command to generate "
        "their data."
    )
    PAGES[choice]()


main()
