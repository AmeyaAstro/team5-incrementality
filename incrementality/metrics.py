"""Pure, testable incrementality math. No Streamlit in here.

Everything works off the aggregated cube (one row per treatment x exposure x
baseline-decile x uplift-percentile cell), so every number in the app is an
exact aggregate of the full 13.98M-row experiment, never a sample estimate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SEGMENTS = ["Persuadable", "Sure Thing", "Lost Cause", "Weak Responder"]
Z95 = 1.959964


# --------------------------------------------------------------------------- filters
def filter_cube(
    cube: pd.DataFrame,
    prop_range: tuple[int, int] = (1, 10),
    segments: list[str] | None = None,
    uplift_range: tuple[int, int] = (1, 100),
) -> pd.DataFrame:
    """Filter on pre-treatment model scores only, so treatment/control stay balanced."""
    m = cube.prop_decile.between(*prop_range) & cube.uplift_pct.between(*uplift_range)
    if segments:
        m &= cube.segment.isin(segments)
    return cube[m]


# --------------------------------------------------------------------------- stats
@dataclass
class Lift:
    n_t: int
    n_c: int
    k_t: int
    k_c: int
    p_t: float
    p_c: float
    diff: float          # absolute lift per treated user (ITT)
    se: float
    ci_low: float
    ci_high: float
    rel_lift: float      # diff / p_c
    p_value: float
    exposure_rate: float  # share of treated users actually served an impression

    def as_dict(self) -> dict:
        return asdict(self)


def lift_stats(cube_f: pd.DataFrame, outcome: str = "conversions") -> Lift:
    t = cube_f[cube_f.treatment == 1]
    c = cube_f[cube_f.treatment == 0]
    n_t, n_c = int(t.n.sum()), int(c.n.sum())
    k_t, k_c = int(t[outcome].sum()), int(c[outcome].sum())
    exposed = int(t.loc[t.exposure == 1, "n"].sum())
    p_t = k_t / n_t if n_t else float("nan")
    p_c = k_c / n_c if n_c else float("nan")
    diff = p_t - p_c
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c) if n_t and n_c else float("nan")
    z = diff / se if se and se > 0 else float("nan")
    p_value = math.erfc(abs(z) / math.sqrt(2)) if not math.isnan(z) else float("nan")
    return Lift(
        n_t=n_t, n_c=n_c, k_t=k_t, k_c=k_c, p_t=p_t, p_c=p_c, diff=diff, se=se,
        ci_low=diff - Z95 * se, ci_high=diff + Z95 * se,
        rel_lift=diff / p_c if p_c else float("nan"), p_value=p_value,
        exposure_rate=exposed / n_t if n_t else float("nan"),
    )


# --------------------------------------------------------------------------- dollars
@dataclass
class Economics:
    treated_users: int
    spend: float
    attributed_conversions: float   # what a last-touch dashboard credits to ads
    baseline_conversions: float     # would have happened with zero ad spend
    incremental_conversions: float
    inc_conv_low: float
    inc_conv_high: float
    attributed_revenue: float
    baseline_revenue: float
    incremental_revenue: float
    inc_rev_low: float
    inc_rev_high: float
    naive_roas: float
    roas: float
    roas_low: float
    roas_high: float
    cost_per_incremental_conversion: float
    breakeven_aov: float            # AOV needed for incremental ROAS = 1.0
    breakeven_cost: float           # max cost per treated user for ROAS = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


def economics(lift: Lift, aov: float, cost_per_user: float) -> Economics:
    n = lift.n_t
    spend = n * cost_per_user
    attributed = lift.k_t
    baseline = lift.p_c * n
    inc = lift.diff * n
    inc_lo, inc_hi = lift.ci_low * n, lift.ci_high * n
    safe = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return Economics(
        treated_users=n,
        spend=spend,
        attributed_conversions=attributed,
        baseline_conversions=baseline,
        incremental_conversions=inc,
        inc_conv_low=inc_lo,
        inc_conv_high=inc_hi,
        attributed_revenue=attributed * aov,
        baseline_revenue=baseline * aov,
        incremental_revenue=inc * aov,
        inc_rev_low=inc_lo * aov,
        inc_rev_high=inc_hi * aov,
        naive_roas=safe(attributed * aov, spend),
        roas=safe(inc * aov, spend),
        roas_low=safe(inc_lo * aov, spend),
        roas_high=safe(inc_hi * aov, spend),
        cost_per_incremental_conversion=safe(spend, inc) if inc > 0 else float("inf"),
        breakeven_aov=safe(spend, inc) if inc > 0 else float("inf"),
        breakeven_cost=safe(inc * aov, n),
    )


# --------------------------------------------------------------------------- verdict
@dataclass
class Verdict:
    level: str   # "profit" | "loss" | "straddle" | "noise"
    headline: str
    detail: str


def verdict(lift: Lift, econ: Economics) -> Verdict:
    """The significance guardrail on the headline number."""
    if lift.ci_low <= 0:
        return Verdict(
            "noise",
            "No proven lift for this audience",
            f"The 95% interval on extra conversions includes zero (p = {lift.p_value:.2f}). "
            "The gap between treated and control could be chance, so no ROAS claim is defensible here.",
        )
    if econ.roas_high < 1:
        return Verdict(
            "loss",
            f"The ads work, but they lose money: {econ.roas:.2f}× incremental ROAS",
            f"Even at the top of the 95% interval ({econ.roas_high:.2f}×) every $1 of spend returns less than $1 "
            f"of incremental revenue. To break even at this cost per user, average order value would need to be "
            f"${econ.breakeven_aov:,.0f}.",
        )
    if econ.roas_low > 1:
        return Verdict(
            "profit",
            f"Profitable with 95% confidence: {econ.roas:.2f}× incremental ROAS",
            f"The whole 95% interval ({econ.roas_low:.2f}×–{econ.roas_high:.2f}×) sits above break-even.",
        )
    return Verdict(
        "straddle",
        f"Real lift, but ROAS ({econ.roas:.2f}×) can't be told apart from break-even",
        f"The 95% interval runs {econ.roas_low:.2f}×–{econ.roas_high:.2f}×, so it includes 1.0. "
        "More data or a narrower audience is needed before calling this profitable.",
    )


# --------------------------------------------------------------------------- segments
def segment_table(cube_f: pd.DataFrame, aov: float, cost_per_user: float, outcome: str = "conversions") -> pd.DataFrame:
    rows = []
    for seg in SEGMENTS:
        sub = cube_f[cube_f.segment == seg]
        if sub.empty or sub[sub.treatment == 0].n.sum() == 0:
            continue
        lf = lift_stats(sub, outcome)
        ec = economics(lf, aov, cost_per_user)
        rows.append({
            "segment": seg,
            "users": lf.n_t + lf.n_c,
            "treated_users": lf.n_t,
            "baseline_rate": lf.p_c,
            "treated_rate": lf.p_t,
            "lift": lf.diff,
            "lift_low": lf.ci_low,
            "lift_high": lf.ci_high,
            "lift_se": lf.se,
            "p_value": lf.p_value,
            "spend": ec.spend,
            "incremental_revenue": ec.incremental_revenue,
            "roas": ec.roas,
            "roas_low": ec.roas_low,
            "roas_high": ec.roas_high,
        })
    return pd.DataFrame(rows)


def targeting_curve(cube_f: pd.DataFrame, aov: float, cost_per_user: float,
                    outcome: str = "conversions", step: int = 5) -> pd.DataFrame:
    """'Only advertise to the top X% by predicted uplift': what happens to revenue and ROAS?"""
    rows = []
    for top in range(step, 101, step):
        sub = cube_f[cube_f.uplift_pct > 100 - top]
        if sub[sub.treatment == 0].n.sum() == 0:
            continue
        lf = lift_stats(sub, outcome)
        ec = economics(lf, aov, cost_per_user)
        rows.append({
            "top_pct": top, "treated_users": lf.n_t, "spend": ec.spend,
            "incremental_revenue": ec.incremental_revenue, "inc_rev_low": ec.inc_rev_low,
            "inc_rev_high": ec.inc_rev_high, "roas": ec.roas, "roas_low": ec.roas_low,
            "roas_high": ec.roas_high, "lift": lf.diff,
        })
    return pd.DataFrame(rows)


def lift_by_uplift_bucket(cube_f: pd.DataFrame, buckets: int = 10, outcome: str = "conversions") -> pd.DataFrame:
    """Observed lift per predicted-uplift bucket (deciles by default), full population."""
    width = 100 // buckets
    rows = []
    for b in range(1, buckets + 1):
        sub = cube_f[cube_f.uplift_pct.between((b - 1) * width + 1, b * width)]
        if sub[sub.treatment == 0].n.sum() == 0:
            continue
        lf = lift_stats(sub, outcome)
        rows.append({"bucket": b, "lift": lf.diff, "low": lf.ci_low, "high": lf.ci_high, "users": lf.n_t + lf.n_c})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- simulator
def reallocate(seg: pd.DataFrame, budget: float, shift_pct: float, from_segments: list[str],
               aov: float, cost_per_user: float, to_segment: str = "Persuadable") -> dict:
    """Move `shift_pct` of the budget sitting in `from_segments` into `to_segment`.

    Assumes each extra treated user in a segment behaves like that segment's average
    treated user (constant per-user lift) and caps the receiving segment at its
    real population, so the simulator can't buy users who don't exist.
    """
    s = seg.set_index("segment").copy()
    share = s.treated_users / s.treated_users.sum()
    s["budget_now"] = budget * share
    moved = sum(s.at[x, "budget_now"] * shift_pct for x in from_segments if x in s.index and x != to_segment)
    s["budget_new"] = s.budget_now
    for x in from_segments:
        if x in s.index and x != to_segment:
            s.at[x, "budget_new"] = s.at[x, "budget_now"] * (1 - shift_pct)

    cap_users = float(s.at[to_segment, "users"])
    cap_budget = cap_users * cost_per_user
    want = s.at[to_segment, "budget_now"] + moved
    s.at[to_segment, "budget_new"] = min(want, cap_budget)
    unspent = max(0.0, want - cap_budget)

    for col in ("now", "new"):
        users = s[f"budget_{col}"] / cost_per_user
        s[f"users_{col}"] = users
        s[f"revenue_{col}"] = users * s.lift * aov
    delta_users = s.users_new - s.users_now
    delta = float((s.revenue_new - s.revenue_now).sum())
    delta_se = float(np.sqrt(((delta_users * aov * s.lift_se) ** 2).sum()))
    return {
        "table": s.reset_index(),
        "revenue_now": float(s.revenue_now.sum()),
        "revenue_new": float(s.revenue_new.sum()),
        "delta": delta,
        "delta_low": delta - Z95 * delta_se,
        "delta_high": delta + Z95 * delta_se,
        "moved": float(moved),
        "unspent": float(unspent),
        "capped": unspent > 0,
        "roas_now": float(s.revenue_now.sum() / budget) if budget else float("nan"),
        "roas_new": float(s.revenue_new.sum() / (budget - unspent)) if budget - unspent else float("nan"),
    }


# --------------------------------------------------------------------------- model
def score(X: np.ndarray, arm: dict, features: list[str]) -> np.ndarray:
    w = np.array([arm["weights"][f] for f in features])
    return 1.0 / (1.0 + np.exp(-(X @ w + arm["intercept"])))
