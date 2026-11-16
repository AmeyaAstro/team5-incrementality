"""Mandate: how much of the revenue would have happened with zero ad spend?"""

import streamlit as st

from incrementality import charts
from incrementality.metrics import SEGMENTS, economics, filter_cube, lift_stats, verdict
from incrementality.ui import DECILE_LABELS, assumptions, money, num, pp, roas, verdict_box

b = st.session_state["bundle"]
aov, cost = assumptions()

st.title("Would that revenue have happened with zero ad spend?")
st.caption("13.98M users were randomly split: 85% eligible for ads, 15% held out. Because the split is random, "
           "the hold-out group shows what would have happened without ads. The gap between the groups is what the ads caused.")

# ---- filters (one row, above the charts)
f1, f2 = st.columns([3, 2])
with f1:
    lo, hi = st.select_slider(
        "Audience: model-estimated likelihood to buy *without* ads",
        options=list(range(1, 11)), value=(1, 10), format_func=lambda d: DECILE_LABELS[d],
        help="Deciles of the control model's predicted conversion probability. D10 = the users most "
             "likely to buy anyway. A pre-treatment score, so treated vs. control stay comparable.")
with f2:
    segs = st.multiselect("Uplift segments", SEGMENTS, default=SEGMENTS,
                          help="From the uplift model (see the Opportunity page).")

cube_f = filter_cube(b.cube, (lo, hi), segs or None)
if cube_f.empty or cube_f[cube_f.treatment == 0].n.sum() == 0:
    st.warning("No users match this filter.")
    st.stop()

lift = lift_stats(cube_f)
econ = economics(lift, aov, cost)
v = verdict(lift, econ)
st.session_state["verdict_state"] = {"prop_range": [lo, hi], "segments": segs, "lift": lift.as_dict(),
                                     "economics": econ.as_dict(), "verdict": v.headline}

verdict_box(v.level, v.headline, v.detail)

k = st.columns(5)
k[0].metric("Treated users", num(lift.n_t), help="Users eligible for ads in this audience.")
k[1].metric("Incremental conversions", num(econ.incremental_conversions),
            delta=f"95% CI {num(econ.inc_conv_low)}–{num(econ.inc_conv_high)}", delta_color="off", delta_arrow="off")
k[2].metric("Incremental revenue", money(econ.incremental_revenue),
            delta=f"{money(econ.inc_rev_low)}–{money(econ.inc_rev_high)}", delta_color="off", delta_arrow="off")
k[3].metric("Incremental ROAS", roas(econ.roas),
            delta=f"{roas(econ.roas_low)}–{roas(econ.roas_high)}", delta_color="off", delta_arrow="off",
            help="Incremental revenue ÷ ad spend. Below 1.0× the ads cost more than the revenue they cause.")
k[4].metric("What a ROAS dashboard shows", roas(econ.naive_roas),
            delta=f"{econ.naive_roas / econ.roas:.1f}× overstated" if econ.roas > 0 else None,
            delta_color="inverse", delta_arrow="off",
            help="All treated-user revenue ÷ spend: credits ads for people who would have bought anyway.")

c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(charts.revenue_waterfall(econ), use_container_width=True)
with c2:
    st.plotly_chart(charts.cvr_bars(lift), use_container_width=True)

c3, c4 = st.columns([3, 2])
with c3:
    st.plotly_chart(charts.roas_vs_aov(lift, cost, aov), use_container_width=True)
with c4:
    st.markdown("#### The CFO's question, answered")
    be_aov = econ.breakeven_aov
    st.markdown(
        f"- **Spend:** {money(econ.spend)} for {num(lift.n_t)} treated users at ${cost:.2f} each.\n"
        f"- **Caused by ads:** {num(econ.incremental_conversions)} of {num(econ.attributed_conversions)} "
        f"conversions ({econ.incremental_conversions / econ.attributed_conversions:.0%}). "
        f"The other {num(econ.baseline_conversions)} would have happened anyway.\n"
        f"- **Break-even:** at ${cost:.2f}/user you need an AOV of **{money(be_aov)}**, "
        f"or at ${aov:,.0f} AOV a cost of at most **${econ.breakeven_cost:.3f}/user**.\n"
        f"- **Reach:** only {lift.exposure_rate:.1%} of treated users were actually served an ad, "
        f"so the lift per *exposed* user is about {pp(lift.diff / lift.exposure_rate, 2)}."
    )
    st.caption("Go to the Opportunity page to see whether a narrower audience clears break-even.")

with st.expander("Numbers behind the charts"):
    st.dataframe({
        "": ["Control", "Treated"], "Users": [lift.n_c, lift.n_t], "Conversions": [lift.k_c, lift.k_t],
        "Conversion rate": [f"{lift.p_c:.4%}", f"{lift.p_t:.4%}"],
    }, hide_index=True, use_container_width=True)
    st.write(f"Absolute lift {pp(lift.diff)} (95% CI {pp(lift.ci_low)} to {pp(lift.ci_high)}), "
             f"relative lift {lift.rel_lift:.0%}, two-sided p = {lift.p_value:.2g}.")
