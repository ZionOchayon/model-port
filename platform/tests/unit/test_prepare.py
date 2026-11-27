import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "datasets"))
import prepare

from modelport.spec import load_spec

ROOT = Path(__file__).resolve().parents[3]


def test_contract_matches_example_signatures():
    fashion = prepare.contract("fashion-mnist")
    assert fashion == {
        "dataset": "fashion-mnist",
        "label_column": "label",
        "columns": {"image": "binary", "label": "string"},
        "min_rows": 1000,
        "min_classes": 2,
        "max_class_share": 0.6,
        "image_format": "png",
    }
    covertype = prepare.contract("covertype")
    spec = load_spec(ROOT / "examples" / "models" / "covertype-classifier" / "model.yaml")
    expected = {c.name: c.type.value for c in spec.signature.inputs} | {"cover_type": "string"}
    assert covertype["columns"] == expected
    assert covertype["image_format"] is None
    assert prepare.contract("eurosat")["columns"] == {"image": "binary", "label": "string"}


def test_deterministic_subsample_and_offset():
    frame = pd.DataFrame({"i": range(100)})
    first = prepare.subsample(frame, rows=10, offset=0)
    assert first["i"].tolist() == prepare.subsample(frame, rows=10, offset=0)["i"].tolist()
    second = prepare.subsample(frame, rows=10, offset=10)
    assert set(first["i"]).isdisjoint(second["i"])
    assert len(prepare.subsample(frame, rows=500, offset=0)) == 100


def test_covertype_one_hots_collapse_to_indices():
    raw = pd.DataFrame(np.zeros((2, 54)), columns=prepare.COVTYPE_RAW_COLUMNS)
    raw.loc[0, "Wilderness_Area_2"] = 1
    raw.loc[1, "Soil_Type_39"] = 1
    raw["Elevation"] = [2500.0, 3000.0]
    frame = prepare.covertype_frame(raw, np.array([1, 7]))
    assert frame["wilderness_area"].tolist() == [2, 0]
    assert frame["soil_type"].tolist() == [0, 39]
    assert frame["cover_type"].tolist() == ["Spruce/Fir", "Krummholz"]
    assert frame["elevation"].dtype == np.float64
    assert frame["soil_type"].dtype == np.int64


def test_corrupt_file_unreadable_by_pyarrow():
    data = prepare.parquet_bytes(pd.DataFrame({"a": range(1000)}), corrupt=True)
    with pytest.raises(pa.ArrowInvalid):
        pq.read_table(io.BytesIO(data))
    assert len(pq.read_table(io.BytesIO(prepare.parquet_bytes(pd.DataFrame({"a": range(10)}))))) == 10


def test_batch_input_drops_the_label():
    frame = pd.DataFrame({"image": [b"x"], "label": ["Bag"]})
    assert list(prepare.batch_rows(frame, "fashion-mnist").columns) == ["image"]


def test_destination_uris():
    assert (
        prepare.destination("covertype", batch_day=None, name="a.parquet")
        == "s3://incoming/covertype/a.parquet"
    )
    assert prepare.destination("covertype", batch_day="2026-01-01", name="a.parquet") == (
        "s3://batch-input/covertype/2026-01-01/a.parquet"
    )
