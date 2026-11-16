# Vantage Incrementality Audit: Which Ad Spend Actually Works

**Team 5 · Account C (The Incrementality Lab)** · Lavannya Patil · Amey Joshi · Meghana Thummalapally

## Project Description

The CFO of a DTC brand spends **$6M a quarter** through Vantage Performance Group and no longer trusts the
ROAS dashboard. Before renewing, they need to know how much of the credited revenue would have happened with
**zero** ad spend.

This Streamlit app answers that with the **Criteo Uplift dataset**: 13,979,592 users from real randomized
ad experiments (85% ad-eligible, 15% held out). Because assignment is random, the treated-vs-control gap is
causal, not correlational.

| Page | What the client decides |
|---|---|
| **The verdict (Mandate)** | Is the program incremental, and does it pay for itself? Revenue waterfall, lift with 95% intervals, a significance guardrail on the headline ROAS, and a what-if on order value and cost. |
| **Wasted-spend finder (Opportunity)** | Which users to stop paying for: uplift segments, a "target only the top X%" curve, and a budget reallocation simulator capped at real segment sizes. |
| **Ask the analyst (AI)** | Plain-English Q&A. The LLM can only answer through tool calls into the same functions that draw the charts, and every figure it quotes is automatically checked against the data. A deterministic CFO brief works without an API key. |
| **Uplift model & validation (AI)** | Two-model (T-learner) uplift classifier trained on 800K rows and scored on all 13.98M. Checked on two untouched holdouts, three independent retrains (rank stability), a bottom-1% backfire check, and live in-app re-scoring. |
| **Method & visual choices** | Glossary, marks-and-channels justification for every chart, data limits, and changes made after the AI-client critique. |

**Headline at the illustrative inputs ($85 AOV, $0.35 per treated user):** the ads cause 13,687 extra
conversions ($1.16M), but spend is $4.16M, so incremental ROAS is **0.28×** against a dashboard ROAS of 0.75×.
Advertising only to the **top 20% by predicted uplift** reaches **1.15×** (95% CI 1.07–1.23×) while cutting
spend by $3.3M. Enter real AOV and cost per user in the sidebar and every number updates.

## App Deployment URL

**https://team5-incrementality.streamlit.app**

## Local Setup Instructions

```bash
git clone https://github.com/AmeyaAstro/team5-incrementality.git
cd team5-incrementality
uv sync
uv run streamlit run app.py
```

The pre-built data bundle (`data/`, about 3 MB) is committed, so the app runs right after cloning. No download or
API key is needed.

**Optional: AI Q&A.** Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add an
Anthropic API key, or paste a key into the *Ask the analyst* page (kept for that browser session only).

**Run the tests:** `uv run pytest`

## Rebuilding the data bundle (optional)

The raw dataset (311 MB compressed) is too large for GitHub and for Streamlit Cloud's memory.
`scripts/build_data.py` streams over it, trains the uplift model, scores every row, and writes the small
aggregates the app uses (`cube.parquet`, `sample.parquet`, `model.json`, `validation.json`, `meta.json`).
It takes about 30 seconds after download.

```bash
# downloads criteo/criteo-uplift from Hugging Face
uv run --group prep python scripts/build_data.py
# or from the file downloaded from https://huggingface.co/datasets/criteo/criteo-uplift
uv run --group prep python scripts/build_data.py --csv ~/Downloads/criteo-research-uplift-v2.1.csv.gz
```

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. At https://share.streamlit.io choose **Create app → Deploy from GitHub**, pick the repo, set the main file to
   `app.py`, and under *Advanced settings* pick Python 3.12. Cloud installs from `uv.lock`.
3. Optional: under *Secrets*, add `ANTHROPIC_API_KEY = "sk-ant-..."`.
4. Paste the resulting URL into **App Deployment URL** above.

## Repository layout

```
app.py                     entry point: navigation + shared CFO-assumption sidebar
pages/                     verdict (Mandate), segments (Opportunity), analyst + model (AI), method
incrementality/
  metrics.py               lift, 95% CIs, dollars, verdict guardrail, targeting curve, simulator
  charts.py                Plotly figures (encodings justified on the Method page)
  ai.py                    LLM tools, tool-use loop, number-grounding check, no-key brief
  data.py, ui.py           bundle loader, formatting, palette, sidebar
scripts/build_data.py      raw Criteo → data/ bundle (train, score, aggregate, validate)
scripts/make_synthetic.py  synthetic stand-in for testing the pipeline only
data/                      committed bundle the app runs on
tests/                     unit tests + headless smoke test of every page
```

## Data & license

Criteo Uplift Prediction Dataset v2.1, Diemert et al. (2018), via Hugging Face `criteo/criteo-uplift`,
licensed CC BY-NC-SA 4.0. Features f0–f11 are anonymized by Criteo. The dataset contains no prices, so
average order value and cost per user are client inputs.
