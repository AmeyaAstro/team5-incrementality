import math

import pandas as pd
import pytest

from incrementality import ai
from incrementality.metrics import economics, filter_cube, lift_stats, reallocate, segment_table, verdict


@pytest.fixture(scope="module")
def cube():
    return pd.read_parquet("data/cube.parquet")


def test_cube_covers_every_user(cube):
    import json
    meta = json.load(open("data/meta.json"))
    assert cube.n.sum() == meta["rows"]


def test_hand_computed_lift():
    toy = pd.DataFrame({"treatment": [1, 0], "exposure": [0, 0], "prop_decile": [5, 5], "uplift_pct": [50, 50],
                        "n": [1000, 1000], "visits": [0, 0], "conversions": [30, 20], "segment": ["Sure Thing"] * 2})
    lf = lift_stats(toy)
    assert lf.diff == pytest.approx(0.01)
    assert lf.se == pytest.approx(math.sqrt(0.03 * 0.97 / 1000 + 0.02 * 0.98 / 1000))
    ec = economics(lf, aov=100, cost_per_user=0.5)
    assert ec.incremental_conversions == pytest.approx(10)
    assert ec.roas == pytest.approx(10 * 100 / 500)
    assert ec.breakeven_aov == pytest.approx(50)


def test_verdict_matches_numbers(cube):
    lf = lift_stats(cube)
    v = verdict(lf, economics(lf, 85, 0.35))
    if lf.ci_low > 0 and economics(lf, 85, 0.35).roas_high < 1:
        assert v.level == "loss"


def test_filters_keep_both_arms(cube):
    f = filter_cube(cube, (9, 10), ["Persuadable"])
    assert set(f.treatment) == {0, 1}


def test_reallocation_conserves_budget(cube):
    seg = segment_table(cube, 85, 0.35)
    sim = reallocate(seg, 1_000_000, 0.5, ["Weak Responder", "Lost Cause"], 85, 0.35)
    spent = sim["table"].budget_new.sum() + sim["unspent"]
    assert spent == pytest.approx(1_000_000)
    assert sim["delta"] > 0


def test_grounding_flags_invented_numbers():
    outs = [{"incremental_revenue": 1163421.0, "incremental_roas": 0.2796, "lift": 0.0011519}]
    good = ai.grounding_check("Ads caused $1.16M of revenue, a 0.28× ROAS, lift +0.115pp.", outs, [])
    assert good.ok == good.total == 3
    bad = ai.grounding_check("Ads caused $2.4M of revenue.", outs, [])
    assert bad.ok == 0 and bad.total == 1


def test_llm_tool_loop_with_fake_client(monkeypatch):
    """The analyst must call a tool, feed the result back, and ground its final answer."""
    import json as _json
    import types

    from google import genai
    from google.genai import types as gtypes

    from incrementality.data import DATA_DIR

    bundle = types.SimpleNamespace(
        cube=pd.read_parquet(DATA_DIR / "cube.parquet"),
        validation=_json.load(open(DATA_DIR / "validation.json")),
        model=_json.load(open(DATA_DIR / "model.json")))
    tb = ai.Toolbox(bundle, 85, 0.35)
    truth = tb.run("get_incrementality", {})
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append(contents)
            assert config.tools[0].function_declarations  # tools were declared
            if len(calls) == 1:
                fc = gtypes.FunctionCall(name="get_incrementality", args={})
                content = gtypes.Content(role="model", parts=[gtypes.Part(function_call=fc)])
                return types.SimpleNamespace(function_calls=[fc], candidates=[types.SimpleNamespace(content=content)],
                                             text=None)
            last = contents[-1].parts[0].function_response
            assert last.name == "get_incrementality" and "result" in last.response
            text = f"Incremental ROAS is {truth['incremental_roas']:.2f}× on ${truth['spend'] / 1e6:.2f}M of spend."
            return types.SimpleNamespace(function_calls=None, candidates=[], text=text)

    monkeypatch.setattr(genai, "Client", lambda api_key: types.SimpleNamespace(models=FakeModels()))
    ans = ai.ask("Is it working?", [{"role": "assistant", "content": "hi"}], tb, illustrative=True, key="x")
    assert ans["calls"][0][0] == "get_incrementality"
    assert ans["grounding"].ok == ans["grounding"].total == 2


def test_tool_schemas_build_for_gemini():
    from google.genai import types as gtypes
    tools = ai._declarations(gtypes)
    assert len(tools[0].function_declarations) == len(ai.TOOLS)
