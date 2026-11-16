"""Opportunity: find the wasted spend, then simulate moving it."""

import streamlit as st

from incrementality import charts
from incrementality.metrics import SEGMENTS, reallocate, segment_table, targeting_curve
from incrementality.ui import SEGMENT_HELP, assumptions, money, num, pp, roas

b = st.session_state["bundle"]
aov, cost = assumptions()

st.title("Where is the wasted spend, and what if we moved it?")
st.caption("An uplift model ranks every user by how much the ads change their behaviour. "
           "Observed lift is measured on the full randomized population, never on the model's own predictions.")

seg = segment_table(b.cube, aov, cost)

# ---------------------------------------------------------------- 1. segments
st.subheader("1 · Which users does the money actually move?")
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(charts.segment_lift(seg), use_container_width=True)
with c2:
    view = seg.assign(**{
        "Users": seg.users.map(num), "Lift": seg.lift.map(lambda v: pp(v)),
        "Spend at current mix": seg.spend.map(money), "Incremental revenue": seg.incremental_revenue.map(money),
        "ROAS": seg.roas.map(roas), "What it means": seg.segment.map(SEGMENT_HELP),
    })[["segment", "Users", "Lift", "Spend at current mix", "Incremental revenue", "ROAS", "What it means"]]
    st.dataframe(view.rename(columns={"segment": "Segment"}), hide_index=True, use_container_width=True)
    top = seg.set_index("segment")
    others = top.drop("Persuadable").lift.max()
    st.markdown(f"**Persuadables respond {top.at['Persuadable', 'lift'] / others:.0f}× more** than the "
                f"next-best segment, yet get only "
                f"{top.at['Persuadable', 'treated_users'] / top.treated_users.sum():.0%} of the spend.")

# ---------------------------------------------------------------- 2. targeting curve
st.subheader("2 · Who should we stop paying to reach?")
curve = targeting_curve(b.cube, aov, cost)
profitable = curve[curve.roas_low > 1]
best_default = int(profitable.top_pct.max()) if not profitable.empty else 25
cut = st.slider("Advertise only to the top X% of users by predicted uplift", 5, 100, best_default, 5,
                format="%d%%", help="Users below the cut are not targeted, so their spend is saved.")
row = curve[curve.top_pct == cut].iloc[0]
full = curve[curve.top_pct == 100].iloc[0]
m = st.columns(4)
m[0].metric("Spend", money(row.spend), delta=f"−{money(full.spend - row.spend)} saved", delta_color="normal")
m[1].metric("Incremental revenue", money(row.incremental_revenue),
            delta=f"{row.incremental_revenue / full.incremental_revenue:.0%} of today's", delta_color="off", delta_arrow="off")
m[2].metric("Incremental ROAS", roas(row.roas), delta=f"{roas(row.roas_low)}–{roas(row.roas_high)}",
            delta_color="off", delta_arrow="off")
m[3].metric("Profit vs. today", money((row.incremental_revenue - row.spend) - (full.incremental_revenue - full.spend)),
            help="(Incremental revenue − spend) at this cut minus the same at 100% targeting. "
                 "Revenue-based: subtract margin for true profit.")
if profitable.empty:
    st.error(f"At ${aov:,.0f} AOV and ${cost:.2f}/user, no audience cut is profitable with 95% confidence.")
else:
    st.success(f"Targeting the top **{best_default}%** or narrower clears break-even with 95% confidence "
               f"at these assumptions. Everything below that cut costs more than it returns.")
c3, c4 = st.columns(2)
c3.plotly_chart(charts.targeting_revenue(curve, cut), use_container_width=True)
c4.plotly_chart(charts.targeting_roas(curve, cut), use_container_width=True)

# ---------------------------------------------------------------- 3. simulator
st.subheader("3 · Budget reallocation simulator")
s1, s2, s3 = st.columns([1.2, 1.5, 2])
budget = s1.number_input("Quarterly budget to simulate ($)", 100_000, 50_000_000, 1_000_000, 100_000)
shift = s2.slider("Share of budget to move out of the source segments", 0, 100, 50, 5, format="%d%%") / 100
sources = s3.multiselect("Move it out of", [s for s in SEGMENTS if s != "Persuadable"],
                         default=["Weak Responder", "Lost Cause", "Sure Thing"])
sim = reallocate(seg, budget, shift, sources, aov, cost)
st.session_state["sim_state"] = {k: v for k, v in sim.items() if k != "table"} | {
    "budget": budget, "shift": shift, "sources": sources}

r = st.columns(4)
r[0].metric("Incremental revenue now", money(sim["revenue_now"]), delta=f"ROAS {roas(sim['roas_now'])}", delta_color="off", delta_arrow="off")
r[1].metric("After reallocation", money(sim["revenue_new"]), delta=f"ROAS {roas(sim['roas_new'])}", delta_color="off", delta_arrow="off")
r[2].metric("Change, same budget", money(sim["delta"]),
            delta=f"95% CI {money(sim['delta_low'])} to {money(sim['delta_high'])}", delta_color="off", delta_arrow="off")
r[3].metric("Budget moved", money(sim["moved"]))
if sim["capped"]:
    st.warning(f"Persuadables hit their real population cap. {money(sim['unspent'])} can't be spent on them "
               "and is shown as unspent rather than imagined users.")
st.plotly_chart(charts.reallocation_bars(sim["table"]), use_container_width=True)
if sim["roas_new"] < 1:
    st.info(f"Reallocation raises revenue by {money(sim['delta'])} on the same budget, but ROAS is still "
            f"{roas(sim['roas_new'])}. Moving money is not enough. Combine it with the narrower audience above, "
            "or renegotiate the cost per user.", icon="💡")
st.caption("Projection, not a promise: it assumes each extra user in a segment responds like that segment's "
           "average user in the experiment. The interval reflects only sampling error in the segment lifts. "
           "Validate with a one-quarter holdout test before committing the full budget.")
