"""Load the pre-built data bundle (see scripts/build_data.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Bundle:
    cube: pd.DataFrame
    sample: pd.DataFrame
    model: dict
    validation: dict
    meta: dict


@st.cache_data(show_spinner="Loading experiment data…")
def _load(data_dir: str) -> tuple:
    d = Path(data_dir)
    cube = pd.read_parquet(d / "cube.parquet")
    sample = pd.read_parquet(d / "sample.parquet")
    model = json.loads((d / "model.json").read_text())
    validation = json.loads((d / "validation.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    return cube, sample, model, validation, meta


def load_bundle() -> Bundle:
    missing = [f for f in ("cube.parquet", "sample.parquet", "model.json", "validation.json", "meta.json")
               if not (DATA_DIR / f).exists()]
    if missing:
        st.error(
            f"Data bundle not found ({', '.join(missing)} missing in `data/`). "
            "Build it with `uv run --group prep python scripts/build_data.py` (see README)."
        )
        st.stop()
    return Bundle(*_load(str(DATA_DIR)))
