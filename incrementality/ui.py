"""Shared formatting, palette and sidebar state."""

from __future__ import annotations

import math

import streamlit as st

# Palette validated with the dataviz skill's validator (light surface):
# blue / gold / red pass lightness, chroma, CVD and normal-vision checks.
# Gold sits just under 3:1 contrast, so every gold mark carries a direct label.
BLUE = "#5b7bd6"
GOLD = "#c08a1e"
RED = "#b03a3a"
GRAY = "#9aa0a8"
INK = "#1f2430"
MUTED = "#6b7280"

SEGMENT_COLORS = {
    "Persuadable": GOLD,
    "Sure Thing": BLUE,
    "Lost Cause": GRAY,
    "Weak Responder": RED,
}
SEGMENT_HELP = {
    "Persuadable": "Top 25% by predicted uplift: ads change what they do.",
    "Sure Thing": "Middling uplift, likely to buy anyway: ads mostly take credit.",
    "Lost Cause": "Middling uplift, unlikely to buy either way.",
    "Weak Responder": "Bottom 25% by predicted uplift: the least ad-responsive.",
}

DEFAULT_AOV = 85.0
DEFAULT_COST = 0.35

DECILE_LABELS = {i: f"D{i}" for i in range(1, 11)}
DECILE_LABELS[1] = "D1 · least likely"
DECILE_LABELS[10] = "D10 · most likely"


# --------------------------------------------------------------------------- format
def money(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:,.2f}M"
    if x >= 1e4:
        return f"{sign}${x / 1e3:,.0f}K"
    return f"{sign}${x:,.{digits}f}"


def num(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:,.0f}"


def pct(x: float, digits: int = 3) -> str:
    return f"{x * 100:.{digits}f}%"


def pp(x: float, digits: int = 3) -> str:
    return f"{x * 100:+.{digits}f}pp"


def roas(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.2f}×"


# --------------------------------------------------------------------------- state
def assumptions() -> tuple[float, float]:
    """CFO inputs, set in the sidebar on every page (see app.py)."""
    return float(st.session_state.get("aov", DEFAULT_AOV)), float(st.session_state.get("cost", DEFAULT_COST))


def sidebar_assumptions(meta: dict) -> None:
    st.sidebar.markdown("### CFO assumptions")
    st.session_state.setdefault("aov", DEFAULT_AOV)
    st.session_state.setdefault("cost", DEFAULT_COST)
    st.sidebar.number_input("Average order value ($)", min_value=1.0, max_value=5000.0, step=5.0, key="aov",
                            help="Revenue from one incremental conversion. Use Vantage's contracted figure.")
    st.sidebar.number_input("Cost per treated user ($)", min_value=0.01, max_value=50.0, step=0.05,
                            format="%.2f", key="cost",
                            help="Media cost to put one user in the ad-exposed group for the quarter.")
    aov, cost = assumptions()
    if aov == DEFAULT_AOV and cost == DEFAULT_COST:
        st.sidebar.warning(
            "**Illustrative defaults** ($85 AOV, $0.35/user). These are not Vantage's numbers. "
            "Enter the contracted values before quoting any dollar figure.", icon="⚠️")
    else:
        st.sidebar.success("Using your inputs on every page.", icon="✅")
    st.sidebar.caption(
        f"Data: {meta['rows']:,} users · {meta['source']} · built {meta['built_at'][:10]}")


def data_banner(meta: dict) -> None:
    if meta.get("synthetic"):
        st.error("**Synthetic test data**: this bundle was built from a generated file, not the Criteo "
                 "experiment. Rebuild with `scripts/build_data.py` before presenting.", icon="🧪")


def style() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-size: 1.6rem; }
        .verdict { border-left: 6px solid var(--c); padding: .8rem 1rem; border-radius: 6px;
                   background: color-mix(in srgb, var(--c) 9%, transparent); margin: .4rem 0 1rem; }
        .verdict h3 { margin: 0 0 .25rem; font-size: 1.25rem; }
        .verdict p { margin: 0; color: inherit; opacity: .85; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def verdict_box(level: str, headline: str, detail: str) -> None:
    color = {"profit": "#2e7d32", "loss": RED, "straddle": GOLD, "noise": MUTED}[level]
    icon = {"profit": "✅", "loss": "⛔", "straddle": "⚖️", "noise": "❔"}[level]
    st.markdown(f'<div class="verdict" style="--c:{color}"><h3>{icon} {headline}</h3><p>{detail}</p></div>',
                unsafe_allow_html=True)
