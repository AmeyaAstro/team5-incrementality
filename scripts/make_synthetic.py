"""Generate a SYNTHETIC stand-in for the Criteo uplift dataset (testing only).

Same columns as `criteo/criteo-uplift` (f0..f11, treatment, exposure, visit,
conversion) and roughly the same rates (85% treated, ~0.2% / ~0.3% CVR), so the
pipeline and the app can be exercised without the 300 MB download.

Never ship a bundle built from this file as if it were real data:
`build_data.py --synthetic` stamps the bundle so the app shows a banner.

    uv run python scripts/make_synthetic.py --rows 2000000 --out data/raw/synthetic.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def make_chunk(rng: np.random.Generator, n: int) -> pa.Table:
    f = rng.normal(size=(n, 12)).astype(np.float32)
    treatment = (rng.random(n) < 0.85).astype(np.int8)

    base_logit = -6.6 + 0.85 * f[:, 0] + 0.45 * f[:, 4] - 0.25 * f[:, 7]
    # Heterogeneous effect: strong for one region of feature space, ~0 elsewhere.
    effect = 0.75 * (1 + np.tanh(1.4 * (f[:, 2] - 0.4))) + 0.15 * f[:, 9].clip(-1, 1)
    logit = base_logit + treatment * effect
    p_conv = 1 / (1 + np.exp(-logit))
    conversion = (rng.random(n) < p_conv).astype(np.int8)

    p_visit = 1 / (1 + np.exp(-(base_logit + 3.0 + treatment * 0.35 * effect)))
    visit = np.maximum(conversion, (rng.random(n) < p_visit).astype(np.int8))
    exposure = (treatment * (rng.random(n) < 0.036 + 0.02 * (f[:, 5] > 1))).astype(np.int8)

    cols = {f"f{i}": f[:, i] for i in range(12)}
    cols.update(treatment=treatment, conversion=conversion, visit=visit, exposure=exposure)
    return pa.table(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--out", type=Path, default=Path("data/raw/synthetic.parquet"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    writer = None
    left = args.rows
    while left > 0:
        n = min(500_000, left)
        t = make_chunk(rng, n)
        writer = writer or pq.ParquetWriter(args.out, t.schema)
        writer.write_table(t)
        left -= n
    writer.close()
    print(f"wrote {args.rows:,} synthetic rows -> {args.out}")


if __name__ == "__main__":
    main()
