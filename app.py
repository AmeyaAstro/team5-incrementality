"""Vantage Incrementality Audit — Team 5 (Account C: The Incrementality Lab).

Run:  uv run streamlit run app.py
"""

import streamlit as st

from incrementality.data import load_bundle
from incrementality.ui import data_banner, sidebar_assumptions, style

st.set_page_config(page_title="Vantage Incrementality Audit", page_icon="📉", layout="wide")
style()

bundle = load_bundle()
st.session_state["bundle"] = bundle

pages = {
    "Mandate": [
        st.Page("pages/verdict.py", title="The verdict: is the spend incremental?", icon="⚖️", default=True),
    ],
    "Opportunity": [
        st.Page("pages/segments.py", title="Wasted-spend finder & reallocation", icon="🎯"),
    ],
    "AI": [
        st.Page("pages/analyst.py", title="Ask the analyst (AI)", icon="💬"),
        st.Page("pages/model.py", title="Uplift model & validation", icon="🧪"),
    ],
    "Reference": [
        st.Page("pages/method.py", title="Method, glossary & visual choices", icon="📖"),
    ],
}
nav = st.navigation(pages)

st.sidebar.markdown("**Vantage Performance Group**  \nCFO incrementality audit · $6M/quarter program")
sidebar_assumptions(bundle.meta)
data_banner(bundle.meta)
nav.run()
