"""Plotly figures. Each one answers one CFO question; encodings are justified on the Method page.

Conventions (dataviz checklist): one y-axis per chart, thin bars with direct
labels, recessive grid, color follows the entity (segment / arm) not its rank,
and every estimate carries its 95% interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .metrics import SEGMENTS, Economics, Lift
from .ui import BLUE, GOLD, GRAY, INK, MUTED, RED, SEGMENT_COLORS, money, pct, pp

FONT = dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK)


def _layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    kw.setdefault("bargap", 0.35)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        template="plotly_white", font=FONT, height=height,
        margin=dict(l=10, r=10, t=56, b=10),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor="#d4d7dc", ticks="")
    fig.update_yaxes(gridcolor="#eef0f3", zerolinecolor="#c9cdd3", ticks="")
    return fig


def _labels_above(fig: go.Figure, x, top, texts) -> None:
    """Value labels placed above the CI whisker so the two never collide."""
    for xi, yi, t in zip(x, top, texts):
        fig.add_annotation(x=xi, y=yi, text=t, showarrow=False, yshift=11, font=dict(color=INK, size=12))


# --------------------------------------------------------------------------- mandate
def revenue_waterfall(econ: Economics) -> go.Figure:
    """Total -> minus baseline -> incremental, with spend as a reference line on the same $ axis."""
    a, b, inc = econ.attributed_revenue, econ.baseline_revenue, econ.incremental_revenue
    x = ["What the dashboard<br>credits to ads", "Would have happened<br>with zero ad spend", "Actually caused<br>by the ads"]
    fig = go.Figure()
    fig.add_bar(x=[x[0]], y=[a], marker_color=BLUE, name="Attributed",
                text=[money(a)], textposition="outside",
                hovertemplate="Revenue from treated users' conversions: %{y:$,.0f}<extra></extra>")
    fig.add_bar(x=[x[1]], y=[b], base=[inc], marker_color=RED, name="Baseline",
                text=[f"−{money(b)}"], textposition="outside",
                hovertemplate="Control-group rate × treated users × AOV: %{y:$,.0f}<extra></extra>")
    fig.add_bar(x=[x[2]], y=[inc], marker_color=GOLD, name="Incremental",
                error_y=dict(type="data", symmetric=False, array=[econ.inc_rev_high - inc],
                             arrayminus=[inc - econ.inc_rev_low], color=INK, thickness=1.5, width=6),
                hovertemplate=(f"Incremental revenue: %{{y:$,.0f}}<br>95% CI {money(econ.inc_rev_low)}"
                               f" – {money(econ.inc_rev_high)}<extra></extra>"))
    _labels_above(fig, [x[2]], [econ.inc_rev_high], [money(inc)])
    fig.add_hline(y=econ.spend, line_dash="dash", line_color=INK, line_width=1.5,
                  annotation_text=f"Ad spend {money(econ.spend)}", annotation_position="top left",
                  annotation_font_color=INK)
    _layout(fig, "Where the credited revenue really comes from", showlegend=False)
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero",
                     range=[0, max(a, econ.spend) * 1.18])
    return fig


def cvr_bars(lift: Lift) -> go.Figure:
    se_t = np.sqrt(lift.p_t * (1 - lift.p_t) / lift.n_t)
    se_c = np.sqrt(lift.p_c * (1 - lift.p_c) / lift.n_c)
    fig = go.Figure()
    for name, p, se, n, color in (("Control (no ads)", lift.p_c, se_c, lift.n_c, GRAY),
                                  ("Treated (saw ads)", lift.p_t, se_t, lift.n_t, BLUE)):
        fig.add_bar(x=[name], y=[p], marker_color=color, width=0.45,
                    error_y=dict(type="data", array=[1.96 * se], color=INK, thickness=1.5, width=8),
                    hovertemplate=f"{name}<br>Conversion rate %{{y:.4%}}<br>±{1.96 * se:.4%} (95%)"
                                  f"<br>n = {n:,}<extra></extra>")
        _labels_above(fig, [name], [p + 1.96 * se], [pct(p)])
    _layout(fig, f"Is the gap real? Lift {pp(lift.diff)} (p {'< 0.001' if lift.p_value < 1e-3 else f'= {lift.p_value:.3f}'})",
            showlegend=False, height=340)
    fig.update_yaxes(tickformat=".2%", rangemode="tozero", range=[0, max(lift.p_t, lift.p_c) * 1.3])
    return fig


def roas_vs_aov(lift: Lift, cost: float, aov_now: float) -> go.Figure:
    """What-if: incremental ROAS as a function of AOV, with its 95% band and the break-even line."""
    be = cost / lift.diff if lift.diff > 0 else np.nan
    hi = max(aov_now * 2, (be * 1.4) if np.isfinite(be) else 0, 50)
    x = np.linspace(1, min(hi, 2000), 200)
    k = 1 / cost
    y, lo, up = x * lift.diff * k, x * lift.ci_low * k, x * lift.ci_high * k
    fig = go.Figure()
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[up, lo[::-1]], fill="toself", line_width=0,
                    fillcolor="rgba(91,123,214,0.15)", hoverinfo="skip", name="95% interval")
    fig.add_scatter(x=x, y=y, line=dict(color=BLUE, width=2), name="Incremental ROAS",
                    hovertemplate="AOV $%{x:,.0f} → ROAS %{y:.2f}×<extra></extra>")
    fig.add_hline(y=1, line_color=RED, line_dash="dash", line_width=1.5,
                  annotation_text="Break-even (1.0×)", annotation_position="bottom right",
                  annotation_font_color=RED)
    now = aov_now * lift.diff * k
    fig.add_scatter(x=[aov_now], y=[now], mode="markers+text", marker=dict(size=11, color=INK),
                    text=[f"Your AOV ${aov_now:,.0f}: {now:.2f}×"], textposition="bottom right",
                    hoverinfo="skip", showlegend=False)
    if np.isfinite(be) and be <= x.max():
        fig.add_vline(x=be, line_color=MUTED, line_dash="dot", line_width=1,
                      annotation_text=f"Break-even AOV ${be:,.0f}", annotation_position="top right",
                      annotation_font_color=MUTED)
    _layout(fig, f"What-if: at ${cost:.2f} per user, what order value pays for the ads?", height=340)
    fig.update_xaxes(title="Average order value ($)", tickprefix="$")
    fig.update_yaxes(title="Incremental ROAS", ticksuffix="×", rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- segments
def segment_lift(seg: pd.DataFrame) -> go.Figure:
    seg = seg.set_index("segment").reindex([s for s in SEGMENTS if s in set(seg.segment)]).reset_index()
    fig = go.Figure()
    fig.add_bar(
        x=seg.segment, y=seg.lift, marker_color=[SEGMENT_COLORS[s] for s in seg.segment],
        error_y=dict(type="data", symmetric=False, array=seg.lift_high - seg.lift,
                     arrayminus=seg.lift - seg.lift_low, color=INK, thickness=1.5, width=8),
        customdata=np.c_[seg.users, seg.lift_low * 100, seg.lift_high * 100],
        hovertemplate="%{x}<br>Lift %{y:.3%}<br>95% CI %{customdata[1]:.3f} to %{customdata[2]:.3f}pp"
                      "<br>%{customdata[0]:,} users<extra></extra>")
    _labels_above(fig, seg.segment, seg.lift_high, [pp(v) for v in seg.lift])
    fig.add_hline(y=0, line_color=INK, line_width=1)
    _layout(fig, "Extra conversions per treated user, by segment (all 13.98M users)", showlegend=False)
    fig.update_yaxes(tickformat=".2%", title="Conversion lift")
    return fig


def targeting_revenue(curve: pd.DataFrame, cutoff: int) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=curve.top_pct, y=curve.spend, name="Ad spend", line=dict(color=GRAY, width=2, dash="dash"),
                    hovertemplate="Top %{x}%: spend %{y:$,.0f}<extra></extra>")
    fig.add_scatter(x=np.r_[curve.top_pct, curve.top_pct[::-1]], y=np.r_[curve.inc_rev_high, curve.inc_rev_low[::-1]],
                    fill="toself", fillcolor="rgba(192,138,30,0.15)", line_width=0, hoverinfo="skip",
                    showlegend=False)
    fig.add_scatter(x=curve.top_pct, y=curve.incremental_revenue, name="Incremental revenue",
                    line=dict(color=GOLD, width=2), hovertemplate="Top %{x}%: %{y:$,.0f}<extra></extra>")
    fig.add_vline(x=cutoff, line_color=INK, line_width=1)
    _layout(fig, "Revenue caused vs. money spent", height=340)
    fig.update_xaxes(title="Advertise only to the top X% by predicted uplift", ticksuffix="%")
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
    return fig


def targeting_roas(curve: pd.DataFrame, cutoff: int) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=np.r_[curve.top_pct, curve.top_pct[::-1]], y=np.r_[curve.roas_high, curve.roas_low[::-1]],
                    fill="toself", fillcolor="rgba(91,123,214,0.15)", line_width=0, hoverinfo="skip",
                    name="95% interval")
    fig.add_scatter(x=curve.top_pct, y=curve.roas, name="Incremental ROAS", line=dict(color=BLUE, width=2),
                    hovertemplate="Top %{x}%: %{y:.2f}×<extra></extra>")
    fig.add_hline(y=1, line_color=RED, line_dash="dash", line_width=1.5, annotation_text="Break-even",
                  annotation_position="bottom right", annotation_font_color=RED)
    fig.add_vline(x=cutoff, line_color=INK, line_width=1)
    _layout(fig, "Return on each dollar as you narrow the audience", height=340)
    fig.update_xaxes(title="Advertise only to the top X% by predicted uplift", ticksuffix="%")
    fig.update_yaxes(ticksuffix="×", rangemode="tozero")
    return fig


def reallocation_bars(table: pd.DataFrame) -> go.Figure:
    t = table.set_index("segment").reindex([s for s in SEGMENTS if s in set(table.segment)]).reset_index()
    fig = go.Figure()
    fig.add_bar(x=t.segment, y=t.revenue_now, name="Current allocation", marker_color=BLUE,
                text=[money(v) for v in t.revenue_now], textposition="outside",
                customdata=t.budget_now, hovertemplate="%{x}<br>Budget %{customdata:$,.0f}"
                                                         "<br>Incremental revenue %{y:$,.0f}<extra></extra>")
    fig.add_bar(x=t.segment, y=t.revenue_new, name="Reallocated", marker_color=GOLD,
                text=[money(v) for v in t.revenue_new], textposition="outside",
                customdata=t.budget_new, hovertemplate="%{x}<br>Budget %{customdata:$,.0f}"
                                                         "<br>Incremental revenue %{y:$,.0f}<extra></extra>")
    fig.add_hline(y=0, line_color=INK, line_width=1)
    _layout(fig, "Incremental revenue by segment: before vs. after", barmode="group", bargroupgap=0.08)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return fig


# --------------------------------------------------------------------------- model
def validation_bars(validation: dict) -> go.Figure:
    fig = go.Figure()
    for name, color in (("A", BLUE), ("B", GOLD)):
        h = validation["holdouts"][name]
        q = pd.DataFrame(h["quartiles"])
        fig.add_bar(x=[f"Q{k}" for k in q.quartile], y=q.lift, name=f"Holdout {name} (n={h['n']:,})",
                    marker_color=color,
                    error_y=dict(type="data", symmetric=False, array=q.ci_high - q.lift,
                                 arrayminus=q.lift - q.ci_low, color=INK, thickness=1.2, width=5),
                    hovertemplate="Quartile %{x}<br>Observed lift %{y:.3%}<extra></extra>")
    fig.add_hline(y=0, line_color=INK, line_width=1)
    _layout(fig, "Observed lift by predicted-uplift quartile, two untouched holdouts", barmode="group")
    fig.update_xaxes(title="Predicted uplift quartile (Q4 = highest)")
    fig.update_yaxes(tickformat=".2%", title="Observed conversion lift")
    return fig


def decile_lift(dec: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = [RED if b == 1 else GOLD if b == dec.bucket.max() else BLUE for b in dec.bucket]
    fig.add_bar(x=[f"D{b}" for b in dec.bucket], y=dec.lift, marker_color=colors,
                error_y=dict(type="data", symmetric=False, array=dec.high - dec.lift, arrayminus=dec.lift - dec.low,
                             color=INK, thickness=1.2, width=5),
                hovertemplate="%{x}<br>Observed lift %{y:.3%}<extra></extra>")
    ends = dec.iloc[[0, -1]]
    _labels_above(fig, [f"D{b}" for b in ends.bucket], ends.high, [pp(v, 3) for v in ends.lift])
    fig.add_hline(y=0, line_color=INK, line_width=1)
    _layout(fig, "Every predicted-uplift decile, scored on the full population", showlegend=False)
    fig.update_xaxes(title="Predicted uplift decile (D10 = highest)")
    fig.update_yaxes(tickformat=".2%")
    return fig


def coefficients(model: dict) -> go.Figure:
    feats = model["features"]
    t = np.array([model["treated"]["std_weights"][f] for f in feats])
    c = np.array([model["control"]["std_weights"][f] for f in feats])
    order = np.argsort(np.abs(t - c))
    fig = go.Figure()
    fig.add_bar(y=np.array(feats)[order], x=c[order], orientation="h", name="Control model (baseline)",
                marker_color=GRAY, hovertemplate="%{y}: %{x:.2f}<extra>control</extra>")
    fig.add_bar(y=np.array(feats)[order], x=t[order], orientation="h", name="Treated model",
                marker_color=BLUE, hovertemplate="%{y}: %{x:.2f}<extra>treated</extra>")
    fig.add_vline(x=0, line_color=INK, line_width=1)
    _layout(fig, "Standardised weights: where the two models disagree is where uplift comes from",
            height=440, barmode="group")
    fig.update_xaxes(title="Weight per 1 SD of feature (log-odds)")
    return fig


def uplift_hist(sample: pd.DataFrame, cuts: list[float]) -> go.Figure:
    """Uplift scores are heavily right-skewed, so bin on a log scale (non-positive scores counted in the note)."""
    pos = sample[sample.uplift > 0]
    lo, hi = np.quantile(pos.uplift, [0.001, 0.999])
    edges = np.geomspace(lo, hi, 61)
    mids = np.sqrt(edges[:-1] * edges[1:])
    fig = go.Figure()
    widths = np.diff(edges)
    for seg in SEGMENTS:
        counts, _ = np.histogram(pos[pos.segment == seg].uplift.clip(lo, hi), bins=edges)
        fig.add_bar(x=mids, y=counts, width=widths, name=seg, marker_color=SEGMENT_COLORS[seg],
                    marker_line_width=0,
                    hovertemplate=f"{seg}<br>uplift ≈ %{{x:.4%}}<br>%{{y:,}} users<extra></extra>")
    for q, label, anchor in ((24, "25th pct", "right"), (74, "75th pct", "left")):
        if cuts[q] > 0:
            fig.add_vline(x=cuts[q], line_color=INK, line_dash="dot", line_width=1)
            # log axes: annotation x is given in log10 units
            fig.add_annotation(x=np.log10(cuts[q]), y=1, yref="paper", text=label, showarrow=False,
                               xanchor=anchor, xshift=-4 if anchor == "right" else 4,
                               font=dict(color=MUTED, size=12))
    n_nonpos = int((sample.uplift <= 0).sum())
    _layout(fig, f"Predicted uplift per user, 100K-user sample (log scale; {n_nonpos:,} users ≤ 0 not shown)",
            barmode="stack", height=340, bargap=0)
    ticks = [t for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1) if lo <= t <= hi]
    fig.update_xaxes(type="log", title="Predicted uplift: P(convert | ads) − P(convert | no ads)",
                     tickvals=ticks, ticktext=[f"{t * 100:g}%" for t in ticks])
    return fig
