"""AI component 1: the two-model uplift classifier and how we check it."""

import numpy as np
import pandas as pd
import streamlit as st

from incrementality import charts
from incrementality.metrics import filter_cube, lift_by_uplift_bucket, lift_stats, score
from incrementality.ui import num, pp

b = st.session_state["bundle"]
model, val = b.model, b.validation

st.title("The uplift model, and why you can trust its ranking")
st.markdown(
    f"""
**Role.** Rank users by how much ads *change* their chance to buy, not by how likely they are to buy.
**Input.** 12 anonymized user features (f0–f11). **Output.** An uplift score and a segment label for every user.

**How.** A *two-model* (T-learner) classifier: one logistic regression learns P(convert | ads) on treated users,
a second learns P(convert | no ads) on control users. Uplift = the difference. Trained on
{num(model['train_rows'])} rows ({num(model['train_treated'])} treated / {num(model['train_control'])} control),
then scored over all {num(b.meta['rows'])} users. The control model's prediction is also the
*baseline likelihood to buy* used by the audience filter.
"""
)

# ---- check 1: two holdouts
st.subheader("Check 1 · Does a high score mean high real lift on users the model never saw?")
c1, c2 = st.columns([3, 2])
c1.plotly_chart(charts.validation_bars(val), use_container_width=True)
with c2:
    for name in ("A", "B"):
        q = val["holdouts"][name]["quartiles"]
        top, bot = q[3], q[0]
        st.metric(f"Holdout {name}: top vs. bottom quartile", f"{pp(top['lift'], 2)} vs {pp(bot['lift'], 2)}",
                  delta=f"top CI {pp(top['ci_low'], 2)} to {pp(top['ci_high'], 2)}", delta_color="off", delta_arrow="off")
    st.caption("Both holdouts were drawn from rows excluded from training. The model's top quartile shows "
               "clearly positive lift in both; the lower quartiles sit near zero.")

# ---- check 2: stability
st.subheader("Check 2 · Would a different 800K sample rank users the same way?")
stab = val["stability"]
s = st.columns(len(stab["pairs"]) * 2)
for i, p in enumerate(stab["pairs"]):
    s[2 * i].metric(f"Rank correlation vs. retrain #{i + 2}", f"{p['spearman']:.2f}",
                    help="Spearman correlation of uplift scores on a common 200K evaluation set.")
    s[2 * i + 1].metric(f"Top-quartile overlap vs. retrain #{i + 2}", f"{p['top_quartile_overlap']:.0%}",
                        help="Share of the main model's top-25% users that the retrained model also puts in its top 25%.")
st.caption(f"Two extra models were trained on independent {num(stab['train_rows_each'])}-row samples and "
           f"scored on the same {num(stab['eval_rows'])} users. High overlap means the Persuadable list is a "
           "property of the data, not of one lucky sample.")

# ---- check 3: full population deciles + bottom 1%
st.subheader("Check 3 · Is there a segment where ads backfire?")
c3, c4 = st.columns([3, 2])
c3.plotly_chart(charts.decile_lift(lift_by_uplift_bucket(b.cube, 10)), use_container_width=True)
with c4:
    bot = lift_stats(filter_cube(b.cube, uplift_range=(1, 1)))
    st.metric("Bottom 1% of predicted uplift: observed lift", pp(bot.diff, 3),
              delta=f"95% CI {pp(bot.ci_low, 3)} to {pp(bot.ci_high, 3)}", delta_color="off", delta_arrow="off")
    st.markdown(
        "No decile shows a significant **negative** effect, down to the bottom 1%. That's why the old "
        "*Sleeping Dog* label (users ads actively hurt) was renamed **Weak Responder**. The data shows "
        "they barely respond, not that they react badly.")

# ---- check 4: re-scoring wired into the app
st.subheader("Check 4 · Live re-scoring inside the app")
smp = b.sample
X = smp[model["features"]].to_numpy(np.float64)
p_t = score(X, model["treated"], model["features"])
p_c = score(X, model["control"], model["features"])
up = p_t - p_c
upct = np.searchsorted(model["uplift_cuts"], up, side="right") + 1
agree = float((upct == smp.uplift_pct.to_numpy()).mean())
d = st.columns(3)
d[0].metric("Users re-scored just now", num(len(smp)))
d[1].metric("Max score difference vs. build", f"{np.abs(up - smp.uplift.to_numpy()).max():.1e}")
d[2].metric("Same uplift percentile", f"{agree:.2%}")
st.caption("The app re-applies the shipped coefficients to a 100K-user sample on every load and compares with "
           "the offline scores. Small float32 rounding is expected, so a user exactly on a cut point can move one percentile.")
st.plotly_chart(charts.uplift_hist(smp, model["uplift_cuts"]), use_container_width=True)

with st.expander("What the model keys on"):
    st.plotly_chart(charts.coefficients(model), use_container_width=True)
    st.caption("Features are anonymized by Criteo, so we can't name what f0–f11 mean. That's why every "
               "client-facing filter uses the model's *outputs* (baseline likelihood, uplift segment) rather "
               "than raw features.")
    st.dataframe(pd.DataFrame({
        "feature": model["features"],
        "treated weight": [model["treated"]["std_weights"][f] for f in model["features"]],
        "control weight": [model["control"]["std_weights"][f] for f in model["features"]],
    }).round(3), hide_index=True, use_container_width=True)
