"""Streamlit front-end for the World Cup ML Simulator.

Run with ``make app`` (or ``streamlit run app/streamlit_app.py``).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="World Cup ML Simulator", page_icon="soccer", layout="wide")

st.title("World Cup ML Simulator")
st.caption("Probabilistic 2026 FIFA World Cup forecasting - work in progress.")

st.info(
    "No predictions yet. The match model and tournament simulation are still "
    "being built (slices 2-6). Once real data has been processed, this page will "
    "show calibrated match probabilities and Monte Carlo title odds. "
    "By design, nothing here is fabricated."
)

# TODO(slice 7): load simulation output from data/processed and render the
# title-probability chart, a match explorer, and calibration diagnostics.
