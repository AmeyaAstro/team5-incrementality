"""AI component 2: an analyst that answers the CFO in plain English, grounded in the data.

The LLM (Google Gemini, function calling) never sees raw rows and never does arithmetic itself. It can only call
the same functions that drive the charts (tools below), and every number in its
answer is then checked against the tool outputs (`grounding_check`). Without an
API key, a deterministic brief is produced from the same numbers.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field

import pandas as pd

from .metrics import (SEGMENTS, economics, filter_cube, lift_stats, reallocate, segment_table,
                      targeting_curve, verdict)
from .ui import money, num, pp, roas

DEFAULT_MODEL = "gemini-3.8-flash"


# --------------------------------------------------------------------------- tools
TOOLS = [
    {
        "name": "get_incrementality",
        "description": "Incremental lift, revenue and ROAS (with 95% intervals) for an audience. "
                       "Audience = baseline-likelihood deciles (1 = least likely to buy without ads, 10 = most) "
                       "and/or uplift segments. Use for 'is the spend working', 'what is the true ROAS', break-even.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prop_min": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
                "prop_max": {"type": "integer", "minimum": 1, "maximum": 10, "default": 10},
                "segments": {"type": "array", "items": {"type": "string", "enum": SEGMENTS}},
                "aov": {"type": "number", "description": "average order value $; omit to use the CFO's current input"},
                "cost_per_user": {"type": "number", "description": "$ per treated user; omit to use current input"},
            },
        },
    },
    {
        "name": "get_segments",
        "description": "Lift, spend, incremental revenue and ROAS for each uplift segment "
                       "(Persuadable, Sure Thing, Lost Cause, Weak Responder).",
        "input_schema": {"type": "object", "properties": {"aov": {"type": "number"}, "cost_per_user": {"type": "number"}}},
    },
    {
        "name": "get_targeting_curve",
        "description": "What happens if ads go only to the top X% of users by predicted uplift, for X = 5..100. "
                       "Returns spend, incremental revenue and ROAS (with intervals) per cut. Use for 'who should we stop targeting'.",
        "input_schema": {"type": "object", "properties": {"aov": {"type": "number"}, "cost_per_user": {"type": "number"}}},
    },
    {
        "name": "simulate_reallocation",
        "description": "Move a share of a budget out of some segments into Persuadables (capped at their real size). "
                       "Returns revenue before/after, change with 95% interval, ROAS before/after.",
        "input_schema": {
            "type": "object",
            "properties": {
                "budget": {"type": "number", "default": 1_000_000},
                "shift_share": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "from_segments": {"type": "array", "items": {"type": "string", "enum": SEGMENTS[1:]}},
                "aov": {"type": "number"}, "cost_per_user": {"type": "number"},
            },
            "required": ["budget", "shift_share"],
        },
    },
    {
        "name": "get_model_validation",
        "description": "How the uplift model was validated: two holdout samples (observed lift per predicted quartile) "
                       "and stability across three independently trained models.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _r(x, sig: int = 6):
    """Round floats for the LLM (and for grounding) without losing meaning."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(f"{x:.{sig}g}")
    if isinstance(x, dict):
        return {k: _r(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, sig) for v in x]
    return x


class Toolbox:
    def __init__(self, bundle, aov: float, cost: float):
        self.b, self.aov, self.cost = bundle, aov, cost

    def _ac(self, args):
        return float(args.get("aov") or self.aov), float(args.get("cost_per_user") or self.cost)

    def run(self, name: str, args: dict) -> dict:
        aov, cost = self._ac(args)
        if name == "get_incrementality":
            cf = filter_cube(self.b.cube, (int(args.get("prop_min", 1)), int(args.get("prop_max", 10))),
                             args.get("segments") or None)
            lf = lift_stats(cf)
            ec = economics(lf, aov, cost)
            v = verdict(lf, ec)
            return _r({"assumptions": {"aov": aov, "cost_per_user": cost},
                       "control_rate": lf.p_c, "treated_rate": lf.p_t, "lift": lf.diff,
                       "lift_ci": [lf.ci_low, lf.ci_high], "relative_lift": lf.rel_lift, "p_value": lf.p_value,
                       "treated_users": lf.n_t, "control_users": lf.n_c, "spend": ec.spend,
                       "attributed_conversions": ec.attributed_conversions,
                       "baseline_conversions": ec.baseline_conversions,
                       "incremental_conversions": ec.incremental_conversions,
                       "incremental_revenue": ec.incremental_revenue,
                       "incremental_revenue_ci": [ec.inc_rev_low, ec.inc_rev_high],
                       "incremental_roas": ec.roas, "incremental_roas_ci": [ec.roas_low, ec.roas_high],
                       "dashboard_roas": ec.naive_roas, "breakeven_aov": ec.breakeven_aov,
                       "breakeven_cost_per_user": ec.breakeven_cost, "exposure_rate": lf.exposure_rate,
                       "verdict": v.headline})
        if name == "get_segments":
            t = segment_table(self.b.cube, aov, cost)
            return _r({"assumptions": {"aov": aov, "cost_per_user": cost},
                       "segments": t[["segment", "users", "baseline_rate", "lift", "lift_low", "lift_high",
                                      "spend", "incremental_revenue", "roas"]].to_dict("records")})
        if name == "get_targeting_curve":
            c = targeting_curve(self.b.cube, aov, cost, step=10)
            return _r({"assumptions": {"aov": aov, "cost_per_user": cost},
                       "rows": c[["top_pct", "spend", "incremental_revenue", "roas", "roas_low", "roas_high"]]
                       .to_dict("records")})
        if name == "simulate_reallocation":
            seg = segment_table(self.b.cube, aov, cost)
            src = args.get("from_segments") or ["Weak Responder", "Lost Cause", "Sure Thing"]
            sim = reallocate(seg, float(args["budget"]), float(args["shift_share"]), src, aov, cost)
            return _r({"assumptions": {"aov": aov, "cost_per_user": cost}, "from_segments": src,
                       **{k: v for k, v in sim.items() if k != "table"}})
        if name == "get_model_validation":
            v = self.b.validation
            return _r({"holdouts": {k: [{"quartile": q["quartile"], "lift": q["lift"], "ci": [q["ci_low"], q["ci_high"]]}
                                        for q in h["quartiles"]] for k, h in v["holdouts"].items()},
                       "stability": v["stability"]["pairs"], "train_rows": self.b.model["train_rows"]})
        raise ValueError(f"unknown tool {name}")


# --------------------------------------------------------------------------- grounding
NUM_RE = re.compile(r"(?<![\w.])[-−+]?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(%|pp|×|x\b|[KMB]\b|k\b|m\b)?")


@dataclass
class Grounding:
    checked: list[tuple[str, bool]] = field(default_factory=list)

    @property
    def ok(self) -> int:
        return sum(1 for _, g in self.checked if g)

    @property
    def total(self) -> int:
        return len(self.checked)


def _flatten(obj, out: list[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))


def grounding_check(text: str, tool_outputs: list[dict], context_numbers: list[float]) -> Grounding:
    """Every figure in the answer must match a tool output (or a stated assumption) within its displayed precision."""
    vals: list[float] = []
    for o in tool_outputs:
        _flatten(o, vals)
    derived = set()
    for v in vals:
        derived.update({v, v * 100, abs(v), abs(v) * 100})
        if v:
            derived.add(1 / v)  # e.g. "1 in 300 users"
    base = derived | set(context_numbers)
    g = Grounding()
    for m in NUM_RE.finditer(text):
        whole, dec, unit = m.group(1), m.group(2) or "", (m.group(3) or "").strip()
        raw = m.group(0).strip()
        x = float(whole.replace(",", "") + dec)
        if not unit and not dec and x <= 12 and "$" not in raw:
            continue  # ordinals, deciles, quartiles, "2 holdouts"
        if 1990 <= x <= 2100 and not unit:
            continue  # years
        mult = {"K": 1e3, "k": 1e3, "M": 1e6, "m": 1e6, "B": 1e9}.get(unit, 1.0)
        value = x * mult
        step = (10 ** -len(dec[1:]) if dec else 1) * mult
        tol = lambda t: max(step * 0.51, abs(t) * 0.006)  # noqa: E731
        hit = any(abs(abs(value) - abs(t)) <= tol(t) for t in base)
        g.checked.append((raw, hit))
    return g


# --------------------------------------------------------------------------- LLM
SYSTEM = """You are the incrementality analyst for Vantage Performance Group, answering their CFO.
The CFO is deciding whether to renew a $6M/quarter ad budget and no longer trusts ROAS dashboards.
Data: a 13.98M-user randomized experiment (85% ad-eligible, 15% held out), so treated-vs-control gaps are causal.

Rules:
- Get every number from a tool call. Never estimate or compute figures yourself, except simple
  comparisons of two tool numbers, and say so.
- If a 95% interval includes zero (or ROAS straddles 1.0), say the result is not proven.
- ROAS here is revenue-based incremental ROAS; below 1.0× the ads cost more than the revenue they cause.
- AOV and cost per user are the CFO's inputs (currently ${aov:,.2f} and ${cost:.2f}){illustrative}.
- Features f0-f11 are anonymized; never claim to know what they mean.
- Answer in under 170 words, plain English, lead with the decision-relevant number. Use $, ×, %, and pp units.
"""


def _secret(name: str) -> str | None:
    try:
        import streamlit as st
        return st.secrets.get(name) or os.getenv(name)
    except Exception:
        return os.getenv(name)


def api_key() -> str | None:
    try:
        import streamlit as st
        user = st.session_state.get("user_api_key")
    except Exception:
        user = None
    return user or _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")


def model_name() -> str:
    return _secret("GEMINI_MODEL") or DEFAULT_MODEL


def _clean_schema(schema: dict) -> dict:
    """Drop JSON-schema keys Gemini's function declarations don't need."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "default"}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


def _declarations(types):
    decls = []
    for t in TOOLS:
        kw = {"name": t["name"], "description": t["description"]}
        if t["input_schema"].get("properties"):
            kw["parameters_json_schema"] = _clean_schema(t["input_schema"])
        decls.append(types.FunctionDeclaration(**kw))
    return [types.Tool(function_declarations=decls)]


def _pick_model(client, failed: str, error: str) -> str:
    """After a 'model not found/retired' error: use the model Google's error suggests,
    else the newest available 'flash' text model other than the one that failed."""
    suggested = [m for m in re.findall(r"models/([\w.\-]+)", error) if m != failed]
    if suggested:
        return suggested[0]
    try:
        names = [m.name.removeprefix("models/") for m in client.models.list()]
    except Exception:
        return failed
    flash = sorted(n for n in names if "flash" in n and n != failed
                   and not any(x in n for x in ("image", "tts", "audio", "live", "lite", "preview", "exp")))
    return flash[-1] if flash else failed


def _log(msg: str) -> None:
    print(f"[analyst] {msg}", file=sys.stderr, flush=True)  # visible in Streamlit Cloud logs


def ask(question: str, history: list[dict], toolbox: Toolbox, illustrative: bool, key: str,
        on_step=None) -> dict:
    """Run a Gemini function-calling loop. Returns {'text', 'calls': [(name, input, output)], 'grounding', 'model'}.

    `on_step(msg)` is called with progress messages so the UI can show what the AI is doing.
    """
    from google import genai
    from google.genai import types

    step = on_step or (lambda m: None)
    # 45 s per request: a stuck call fails visibly instead of spinning forever.
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=45_000))
    system = SYSTEM.format(aov=toolbox.aov, cost=toolbox.cost,
                           illustrative=" — ILLUSTRATIVE DEFAULTS, flag this when quoting dollars" if illustrative else "")
    config = types.GenerateContentConfig(
        system_instruction=system, tools=_declarations(types), temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    contents = [types.Content(role="model" if h["role"] == "assistant" else "user",
                              parts=[types.Part.from_text(text=h["content"])]) for h in history]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    model = model_name()
    calls = []
    for attempt in range(7):
        step(f"Asking {model}…" if attempt == 0 else "Writing the answer from the data…")
        t0 = time.time()
        _log(f"request {attempt} -> {model}")
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            # Model names change over time: on "not found", fall back to an available flash model once.
            if attempt == 0 and ("404" in str(e) or "not found" in str(e).lower()):
                model = _pick_model(client, model, str(e))
                resp = client.models.generate_content(model=model, contents=contents, config=config)
            else:
                _log(f"request {attempt} failed after {time.time() - t0:.1f}s: {e}")
                raise
        fcs = resp.function_calls or []
        _log(f"request {attempt} ok in {time.time() - t0:.1f}s, tool calls: {[f.name for f in fcs]}")
        if not fcs:
            break
        contents.append(resp.candidates[0].content)
        parts = []
        for fc in fcs:
            args = dict(fc.args or {})
            step(f"Running `{fc.name}` on the 13.98M-user experiment…")
            try:
                out = toolbox.run(fc.name, args)
            except Exception as e:  # surface tool errors to the model rather than crash
                out = {"error": str(e)}
            calls.append((fc.name, args, out))
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": out}))
        contents.append(types.Content(role="user", parts=parts))
    text = (resp.text or "").strip()
    ctx = [toolbox.aov, toolbox.cost, 95, 1.0, 13.98, 85, 15, 6, 100, 25, 50, 75, 800_000, 0.95]
    for _, inp, _ in calls:
        _flatten(inp, ctx)
        ctx += [v * 100 for v in ctx if isinstance(v, float) and v < 1]
    return {"text": text, "calls": calls, "model": model,
            "grounding": grounding_check(text, [c[2] for c in calls], ctx)}


# --------------------------------------------------------------------------- no-key brief
def deterministic_brief(toolbox: Toolbox) -> tuple[str, list[dict]]:
    """Template narration from the same tool outputs: always available, fully grounded."""
    allu = toolbox.run("get_incrementality", {})
    segs = toolbox.run("get_segments", {})
    curve = toolbox.run("get_targeting_curve", {})
    rows = pd.DataFrame(curve["rows"])
    prof = rows[rows.roas_low > 1]
    persu = next(s for s in segs["segments"] if s["segment"] == "Persuadable")
    lines = [
        f"**Bottom line:** across all {num(allu['treated_users'])} treated users the ads caused "
        f"{num(allu['incremental_conversions'])} conversions, {money(allu['incremental_revenue'])} of revenue on "
        f"{money(allu['spend'])} of spend: an incremental ROAS of **{roas(allu['incremental_roas'])}** "
        f"(95% CI {roas(allu['incremental_roas_ci'][0])}–{roas(allu['incremental_roas_ci'][1])}). "
        f"A ROAS dashboard would report {roas(allu['dashboard_roas'])}.",
        f"**Why:** the hold-out group converted at {allu['control_rate'] * 100:.3f}% anyway, so most credited "
        f"conversions were not caused by ads. Break-even needs an AOV of {money(allu['breakeven_aov'])} or a "
        f"cost of ${allu['breakeven_cost_per_user']:.3f} per user.",
        f"**Where it works:** Persuadables show {pp(persu['lift'])} lift and {roas(persu['roas'])} ROAS on their own.",
    ]
    if prof.empty:
        lines.append("**Recommendation:** no audience cut clears break-even with 95% confidence at these inputs. "
                     "Renegotiate cost per user before renewing.")
    else:
        best = prof.sort_values("top_pct").iloc[-1]
        lines.append(f"**Recommendation:** renew only for the top {int(best.top_pct)}% of users by predicted uplift: "
                     f"{money(best.spend)} spend, {money(best.incremental_revenue)} incremental revenue, "
                     f"{roas(best.roas)} ROAS (CI {roas(best.roas_low)}–{roas(best.roas_high)}).")
    return "\n\n".join(lines), [allu, segs, curve]
