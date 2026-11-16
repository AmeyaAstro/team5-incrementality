"""Reference: glossary, method, visual-encoding justifications, limits."""

import pandas as pd
import streamlit as st

b = st.session_state["bundle"]

st.title("Method, glossary & visual choices")

st.subheader("Glossary for the CFO")
st.dataframe(pd.DataFrame([
    ("Treated / control", "Users randomly made eligible for ads (85%) or held out (15%). Random assignment makes the groups comparable, so their gap is caused by the ads."),
    ("Lift (pp)", "Treated conversion rate minus control conversion rate, in percentage points. +0.115pp means 1.15 extra buyers per 1,000 treated users."),
    ("Incremental conversions", "Lift × treated users: conversions that would not have happened without the ads."),
    ("Dashboard (attributed) ROAS", "All revenue from people who saw ads ÷ spend. Counts people who would have bought anyway."),
    ("Incremental ROAS", "Incremental revenue ÷ spend. Below 1.0×, the ads cost more than the revenue they cause."),
    ("95% interval", "The range the true value plausibly lies in, given sampling noise. If it includes 0 (or 1.0× for ROAS), the claim isn't proven."),
    ("Baseline likelihood", "The control model's predicted chance a user buys *without* ads. Used to filter audiences."),
    ("Predicted uplift", "Treated model's probability minus control model's probability: how much ads are predicted to change this user."),
    ("Persuadable / Sure Thing / Lost Cause / Weak Responder", "Top 25% predicted uplift / middle 50% with high baseline / middle 50% with low baseline / bottom 25% predicted uplift."),
    ("Exposure rate", "Share of treated users actually served an impression (~3.6%). Results are intent-to-treat: per user *eligible* for ads."),
], columns=["Term", "Meaning"]), hide_index=True, use_container_width=True)

st.subheader("Why each chart looks the way it does")
st.dataframe(pd.DataFrame([
    ("Revenue waterfall", "Verdict", "Decompose credited revenue into baseline + incremental; compare with spend",
     "Floating bars (length on a shared $ axis), color = role (credited / baseline / caused); spend as a reference line on the same axis",
     "A single '% lift' card hides the baseline and looks like the dashboards the CFO distrusts; a dual-axis spend line would invent a second scale"),
    ("Conversion bars + CI whiskers", "Verdict", "Is the treated-vs-control gap real or noise?",
     "Bar length + position on a common zero baseline, whiskers = 95% CI",
     "A line chart: two unordered arms have no trend to connect"),
    ("ROAS vs. AOV what-if", "Verdict", "At what order value do the ads pay for themselves?",
     "Line (position) with a CI band, red break-even rule at 1.0×, marker at the CFO's AOV",
     "A table of scenarios: hides where the curve crosses break-even"),
    ("Segment lift bars", "Opportunity", "Compare four segments' real lift",
     "Bar length, color = segment identity (kept the same on every page), CI whiskers",
     "Pie chart: angle is a weak channel and can't show negative or uncertain values"),
    ("Targeting curve (2 panels)", "Opportunity", "Where should we cut the audience?",
     "Lines over audience share; revenue vs. spend share one $ axis, ROAS gets its own panel",
     "Dual-axis revenue + ROAS chart: two scales imply false crossings"),
    ("Before / after grouped bars", "Opportunity", "What does reallocation do per segment?",
     "Paired bar lengths side by side, color = scenario",
     "Two pies (before/after): can't compare four segments shifting by different amounts"),
    ("Quartile validation bars", "Model", "Does the ranking hold on unseen users?",
     "Grouped bars per holdout, CI whiskers, zero rule",
     "A single AUUC/Qini score: unreadable for a CFO and hides where it fails"),
], columns=["Chart", "Page", "Client task", "Marks & channels", "Rejected alternative"]),
    hide_index=True, use_container_width=True)
st.caption("Palette (blue / gold / red + neutral gray) was run through a colorblind-separation and contrast "
           "validator. Gold falls just under 3:1 contrast, so every gold mark carries a direct value label.")

st.subheader("What the data can and cannot answer")
st.markdown(f"""
- **Can:** the causal effect of ad eligibility on visits and conversions for {b.meta['rows']:,} users, overall and
  for any audience defined by pre-treatment scores.
- **Cannot:** revenue, margin or price. The dataset has no order values, so **AOV and cost per user are inputs
  the client must supply.** Defaults ($85, $0.35) are illustrative.
- **Cannot:** which channel or creative did the work, or what f0–f11 mean (anonymized by Criteo).
- **Projections assume** extra users in a segment respond like that segment's average. A one-quarter holdout
  test is the right way to confirm before committing the full budget.

**Data:** Criteo Uplift Prediction dataset v2.1 (CC BY-NC-SA 4.0), via Hugging Face `criteo/criteo-uplift`.
Bundle built {b.meta['built_at']} from {b.meta['source']}.
""")

st.subheader("What changed after the AI-client critique (Progress II)")
st.markdown("""
| Critique | What we did |
|---|---|
| "Are $85 / $0.35 your numbers or slider defaults?" | Sidebar flags them as **illustrative** on every page; the verdict now says plainly when ROAS < 1.0× and shows break-even AOV and cost. |
| "I can't act on a filter called f0" | Audience filter uses the model's **baseline likelihood to buy**, a nameable trait. |
| "Would a different 800K sample rank users the same?" | Two extra models trained on independent 800K samples. Rank correlation and top-quartile overlap are shown on the Model page. |
| "'Sleeping Dog' isn't proven negative" | Renamed **Weak Responder**; the bottom-1% check is shown live. |
| "The pitch assumes I've already decided to renew" | The targeting curve answers renew vs. cut directly: which audience share clears break-even. |
""")

st.caption("Team 5 · Lavannya Patil · Amey Joshi · Meghana Thummalapally · Visual Analytics, UNC Charlotte")
