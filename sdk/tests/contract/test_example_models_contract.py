"""Liskov check: every example model, with every spec file, meets the BaseModel contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelport.loader import instantiate
from modelport.spec import load_spec

ROOT = Path(__file__).resolve().parents[3]
SPECS = sorted(
    p for p in ROOT.glob("examples/models/*/model*.yaml") if (p.parent / "example" / "input.parquet").exists()
)


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_example_model_meets_contract(spec_path):
    spec = load_spec(spec_path)
    model = instantiate(spec_path.parent, spec)
    model.load(None)
    example = pd.read_parquet(spec_path.parent / "example" / "input.parquet")
    out = model.run_inference(example)
    assert len(out) == len(example)
    assert set(out["prediction"]) <= set(spec.labels)
    assert ((out["confidence"] >= 0) & (out["confidence"] <= 1)).all()
    stats = model.input_stats(example)
    assert isinstance(stats, pd.DataFrame)
    assert len(stats) == len(example)
    assert all(np.issubdtype(t, np.floating) for t in stats.dtypes)
