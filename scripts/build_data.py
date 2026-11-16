"""Build the small data bundle the Streamlit app runs on.

The raw Criteo uplift table is 13.98M rows (~300 MB parquet), too big for GitHub
and for Streamlit Community Cloud's memory. This script streams over it once to
train the uplift model, then once more to score every row, and writes only what
the app needs (a few MB) to ./data:

    data/cube.parquet        counts for every (treatment, exposure, baseline decile,
                             uplift percentile) cell: n, visits, conversions
    data/sample.parquet      100K scored rows, used for in-app re-scoring + histograms
    data/model.json          two-model logistic uplift classifier (coefficients, cut points)
    data/validation.json     2 independent holdout checks + 3-retrain stability check
    data/meta.json           provenance (source, row counts, build time)

Usage (real data, needs internet to Hugging Face; ~10-20 min on a laptop):

    uv run --group prep python scripts/build_data.py

or from the file downloaded by hand from the dataset page:

    uv run --group prep python scripts/build_data.py --csv ~/Downloads/criteo-research-uplift-v2.1.csv.gz

or, if you already have the parquet from Progress I:

    uv run --group prep python scripts/build_data.py --parquet path/to/criteo_uplift.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ["visit", "conversion"]
SEED = 20261117

# Disjoint slices of a deterministic per-row uniform hash u in [0, 1).
# Widths are expressed in rows and converted to u-widths once N is known.
SLICES = {
    "train": (0.00, 800_000),     # main model (matches the deck: 800K rows)
    "retrain_2": (0.10, 800_000),  # stability check: independent 800K
    "retrain_3": (0.20, 800_000),  # stability check: independent 800K
    "calib": (0.30, 1_000_000),   # cut points for percentiles / deciles
    "val_a": (0.50, 160_000),     # holdout sample A (deck: N=160K)
    "val_b": (0.60, 400_000),     # holdout sample B (deck: N=400K)
    "eval": (0.80, 200_000),      # common set scored by all 3 models
    "app_sample": (0.90, 100_000),  # shipped with the app
}


# --------------------------------------------------------------------------- utils
def row_uniform(idx: np.ndarray, seed: int = SEED) -> np.ndarray:
    """splitmix64 hash of the global row index -> uniform float in [0, 1)."""
    with np.errstate(over="ignore"):
        z = idx.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def fetch_parquet(raw_dir: Path) -> Path:
    out = raw_dir / "criteo_uplift.parquet"
    if out.exists():
        print(f"using cached {out}")
        return out
    from datasets import load_dataset  # heavy import, only when downloading

    print("downloading criteo/criteo-uplift from Hugging Face (311 MB compressed)...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("criteo/criteo-uplift", split="train")
    ds.to_parquet(str(out))
    return out


def csv_to_parquet(csv_path: Path, raw_dir: Path) -> Path:
    """Stream the Hugging Face file (criteo-research-uplift-v2.1.csv.gz) to parquet."""
    import pyarrow.csv as pacsv

    out = raw_dir / "criteo_uplift.parquet"
    if out.exists() and out.stat().st_mtime > csv_path.stat().st_mtime:
        print(f"using cached {out}")
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"converting {csv_path.name} -> {out} ...")
    reader = pacsv.open_csv(csv_path, read_options=pacsv.ReadOptions(block_size=64 << 20))
    writer = None
    for batch in reader:
        writer = writer or pq.ParquetWriter(out, batch.schema, compression="zstd")
        writer.write_batch(batch)
    writer.close()
    return out


def iter_batches(path: Path, columns: list[str], batch_rows: int = 1_000_000):
    pf = pq.ParquetFile(path)
    start = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        df = batch.to_pandas()
        idx = np.arange(start, start + len(df))
        start += len(df)
        yield idx, df


# --------------------------------------------------------------------------- model
class ArmModel:
    """Standardised logistic regression, folded to raw-feature weights for scoring."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=0.5, max_iter=2000)
        self.clf.fit(self.scaler.transform(X), y)
        w_std = self.clf.coef_[0]
        self.w = w_std / self.scaler.scale_
        self.b = float(self.clf.intercept_[0] - np.sum(w_std * self.scaler.mean_ / self.scaler.scale_))
        self.w_std = w_std  # standardised weights, comparable across features

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


class TwoModel:
    def __init__(self, df: pd.DataFrame):
        t = df[df.treatment == 1]
        c = df[df.treatment == 0]
        self.treated = ArmModel(t[FEATURES].to_numpy(np.float64), t.conversion.to_numpy())
        self.control = ArmModel(c[FEATURES].to_numpy(np.float64), c.conversion.to_numpy())
        self.n_treated, self.n_control = len(t), len(c)

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_t = self.treated.predict(X)
        p_c = self.control.predict(X)
        return p_t, p_c, p_t - p_c


def segment_of(upct: np.ndarray, pdec: np.ndarray) -> np.ndarray:
    """Persuadable = top uplift quartile; Weak Responder = bottom quartile;
    the middle half splits on baseline likelihood into Sure Thing / Lost Cause."""
    seg = np.where(pdec > 5, "Sure Thing", "Lost Cause").astype(object)
    seg[upct > 75] = "Persuadable"
    seg[upct <= 25] = "Weak Responder"
    return seg


def bucket(values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """1-based bucket index given ascending interior cut points."""
    return np.searchsorted(cuts, values, side="right") + 1


def observed_lift(df: pd.DataFrame, outcome: str = "conversion") -> dict:
    t = df[df.treatment == 1][outcome]
    c = df[df.treatment == 0][outcome]
    pt, pc = t.mean(), c.mean()
    se = np.sqrt(pt * (1 - pt) / max(len(t), 1) + pc * (1 - pc) / max(len(c), 1))
    return {"n_t": int(len(t)), "n_c": int(len(c)), "cvr_t": float(pt), "cvr_c": float(pc),
            "lift": float(pt - pc), "ci_low": float(pt - pc - 1.96 * se), "ci_high": float(pt - pc + 1.96 * se)}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, help="existing Criteo uplift parquet (skips download)")
    ap.add_argument("--csv", type=Path, help="criteo-research-uplift-v2.1.csv.gz downloaded from Hugging Face")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--synthetic", action="store_true", help="input is the synthetic test file; stamp the bundle")
    args = ap.parse_args()

    t0 = time.time()
    if args.csv:
        src = csv_to_parquet(args.csv, args.raw_dir)
    else:
        src = args.parquet or fetch_parquet(args.raw_dir)
    N = pq.ParquetFile(src).metadata.num_rows
    print(f"source: {src}  rows: {N:,}")

    # Real data (13.98M rows) fits every slice at full size. Smaller test files
    # get proportionally smaller slices so the bands never overlap.
    scale = min(1.0, 0.095 * N / 800_000)
    if scale < 1:
        print(f"small input: scaling every slice to {scale:.0%} of its full size")
    bands = {k: (lo, lo + scale * rows / N) for k, (lo, rows) in SLICES.items()}

    cols = FEATURES + ["treatment", "exposure", "visit", "conversion"]

    # ---- pass 1: collect slices -------------------------------------------------
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in bands}
    n_treat = 0
    for idx, df in iter_batches(src, cols):
        u = row_uniform(idx)
        n_treat += int(df.treatment.sum())
        for k, (lo, hi) in bands.items():
            m = (u >= lo) & (u < hi)
            if m.any():
                parts[k].append(df[m])
    S = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    print(f"pass 1 done in {time.time() - t0:,.0f}s: " + ", ".join(f"{k}={len(v):,}" for k, v in S.items()))

    # ---- train ------------------------------------------------------------------
    main_model = TwoModel(S["train"])
    retrains = [TwoModel(S["retrain_2"]), TwoModel(S["retrain_3"])]
    print(f"trained 3 two-model classifiers in {time.time() - t0:,.0f}s")

    Xcal = S["calib"][FEATURES].to_numpy(np.float64)
    _, pc_cal, up_cal = main_model.score(Xcal)
    uplift_cuts = np.quantile(up_cal, np.arange(1, 100) / 100)   # 99 cuts -> percentiles 1..100
    prop_cuts = np.quantile(pc_cal, np.arange(1, 10) / 10)       # 9 cuts -> deciles 1..10

    # ---- pass 2: score every row, aggregate to cube -----------------------------
    shape = (2, 2, 10, 100)  # treatment, exposure, prop decile, uplift pct
    n_arr = np.zeros(shape, np.int64)
    v_arr = np.zeros(shape, np.int64)
    c_arr = np.zeros(shape, np.int64)
    flat = int(np.prod(shape))
    for _, df in iter_batches(src, cols):
        _, p_c, up = main_model.score(df[FEATURES].to_numpy(np.float64))
        pdec = bucket(p_c, prop_cuts)
        upct = bucket(up, uplift_cuts)
        key = np.ravel_multi_index(
            (df.treatment.to_numpy(int), df.exposure.to_numpy(int), pdec - 1, upct - 1), shape)
        n_arr += np.bincount(key, minlength=flat).reshape(shape)
        v_arr += np.bincount(key, weights=df.visit.to_numpy(), minlength=flat).astype(np.int64).reshape(shape)
        c_arr += np.bincount(key, weights=df.conversion.to_numpy(), minlength=flat).astype(np.int64).reshape(shape)
    print(f"pass 2 (scored {N:,} rows) done in {time.time() - t0:,.0f}s")

    tt, ee, pp, uu = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    cube = pd.DataFrame({
        "treatment": tt.ravel().astype(np.int8), "exposure": ee.ravel().astype(np.int8),
        "prop_decile": (pp.ravel() + 1).astype(np.int8), "uplift_pct": (uu.ravel() + 1).astype(np.int8),
        "n": n_arr.ravel(), "visits": v_arr.ravel(), "conversions": c_arr.ravel(),
    })
    cube = cube[cube.n > 0].reset_index(drop=True)
    cube["segment"] = segment_of(cube.uplift_pct.to_numpy(), cube.prop_decile.to_numpy())
    assert cube.n.sum() == N

    # ---- validation: two independent holdouts ------------------------------------
    def quartile_table(df: pd.DataFrame) -> list[dict]:
        _, _, up = main_model.score(df[FEATURES].to_numpy(np.float64))
        q = np.ceil(bucket(up, uplift_cuts) / 25).astype(int)
        rows = []
        for k in (1, 2, 3, 4):
            r = observed_lift(df[q == k])
            r["quartile"] = k
            rows.append(r)
        return rows

    validation = {"holdouts": {
        "A": {"n": len(S["val_a"]), "quartiles": quartile_table(S["val_a"])},
        "B": {"n": len(S["val_b"]), "quartiles": quartile_table(S["val_b"])},
    }}

    # ---- stability: does a different 800K sample rank users the same way? -------
    Xe = S["eval"][FEATURES].to_numpy(np.float64)
    scores = [m.score(Xe)[2] for m in [main_model, *retrains]]
    ranks = [pd.Series(s).rank().to_numpy() for s in scores]
    top = [r > np.quantile(r, 0.75) for r in ranks]
    stab = []
    for j, name in ((1, "retrain_2"), (2, "retrain_3")):
        stab.append({
            "model": name,
            "spearman": float(np.corrcoef(ranks[0], ranks[j])[0, 1]),
            "top_quartile_overlap": float((top[0] & top[j]).sum() / top[0].sum()),
        })
    validation["stability"] = {"eval_rows": len(S["eval"]), "train_rows_each": len(S["train"]), "pairs": stab}

    # ---- shipped sample --------------------------------------------------------
    smp = S["app_sample"].copy()
    p_t, p_c, up = main_model.score(smp[FEATURES].to_numpy(np.float64))
    smp["p_treated"], smp["p_control"], smp["uplift"] = p_t, p_c, up
    smp["prop_decile"] = bucket(p_c, prop_cuts).astype(np.int8)
    smp["uplift_pct"] = bucket(up, uplift_cuts).astype(np.int8)
    smp["segment"] = segment_of(smp.uplift_pct.to_numpy(), smp.prop_decile.to_numpy())
    for c in FEATURES + ["p_treated", "p_control", "uplift"]:
        smp[c] = smp[c].astype(np.float32)
    for c in ["treatment", "exposure", "visit", "conversion"]:
        smp[c] = smp[c].astype(np.int8)

    # ---- write ------------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    cube.to_parquet(args.out / "cube.parquet", index=False, compression="zstd")
    smp.to_parquet(args.out / "sample.parquet", index=False, compression="zstd")

    def arm(m: ArmModel) -> dict:
        return {"intercept": m.b, "weights": dict(zip(FEATURES, map(float, m.w))),
                "std_weights": dict(zip(FEATURES, map(float, m.w_std)))}

    model = {
        "type": "two-model logistic uplift (T-learner)",
        "outcome": "conversion",
        "features": FEATURES,
        "train_rows": len(S["train"]),
        "train_treated": main_model.n_treated,
        "train_control": main_model.n_control,
        "treated": arm(main_model.treated),
        "control": arm(main_model.control),
        "uplift_cuts": list(map(float, uplift_cuts)),
        "prop_cuts": list(map(float, prop_cuts)),
        "segments": {
            "Persuadable": "top 25% of predicted uplift",
            "Sure Thing": "middle 50% of uplift, above-median baseline likelihood",
            "Lost Cause": "middle 50% of uplift, below-median baseline likelihood",
            "Weak Responder": "bottom 25% of predicted uplift",
        },
    }
    (args.out / "model.json").write_text(json.dumps(model, indent=2))
    (args.out / "validation.json").write_text(json.dumps(validation, indent=2))
    meta = {
        "source": "SYNTHETIC test data (not Criteo)" if args.synthetic else "criteo/criteo-uplift (Hugging Face)",
        "synthetic": bool(args.synthetic),
        "rows": int(N), "treated": n_treat, "control": int(N - n_treat),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(time.time() - t0, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in args.out.glob("*.*") if p.is_file()}
    print(f"done in {time.time() - t0:,.0f}s -> {sizes}")


if __name__ == "__main__":
    main()
