import numpy as np
import pandas as pd
import pytest

from modelport.errors import SignatureMismatchError
from modelport.signature import build_signature, check_input_frame, check_output_frame
from modelport.spec import parse_spec

SPEC = parse_spec(
    {
        "name": "shape-classifier",
        "class": "model:Shape",
        "framework": "torch",
        "labels": ["circle", "square"],
        "signature": {
            "inputs": [
                {"name": "image", "type": "binary"},
                {"name": "size", "type": "double"},
                {"name": "count", "type": "long"},
                {"name": "ratio", "type": "float"},
                {"name": "ok", "type": "boolean"},
                {"name": "tag", "type": "string"},
            ]
        },
        "data": {"dataset": "s3://datasets/shapes"},
    }
)
GOOD = pd.DataFrame(
    {
        "image": [b"\x89PNG", b"\x89PNG"],
        "size": [1.5, 2.0],
        "count": [1, 2],
        "ratio": [0.5, np.nan],
        "ok": [True, False],
        "tag": ["a", "b"],
    }
)


def mismatch(frame):
    with pytest.raises(SignatureMismatchError) as info:
        check_input_frame(frame, SPEC)
    assert info.value.fix
    return info.value.what


def output(labels, confidence):
    return pd.DataFrame({"prediction": labels, "confidence": confidence})


def test_build_signature_types():
    sig = build_signature(SPEC)
    assert [(c.name, c.type.name) for c in sig.inputs.inputs] == [
        ("image", "binary"),
        ("size", "double"),
        ("count", "long"),
        ("ratio", "float"),
        ("ok", "boolean"),
        ("tag", "string"),
    ]
    assert [(c.name, c.type.name) for c in sig.outputs.inputs] == [
        ("prediction", "string"),
        ("confidence", "double"),
    ]


def test_valid_frame_passes():
    check_input_frame(GOOD, SPEC)


def test_missing_input_column():
    assert mismatch(GOOD.drop(columns="tag")) == "Input is missing column 'tag'"


def test_binary_column_rejects_str():
    what = mismatch(GOOD.assign(image=["abc", "def"]))
    assert what == "Column 'image' must hold bytes (signature type binary) but row 0 holds str"


@pytest.mark.parametrize(
    ("column", "values", "row"),
    [
        ("size", [1.0, np.nan], 1),
        ("size", ["1.0", "2.0"], 0),
        ("count", [1.0, 2.5], 0),
        ("count", [True, False], 0),
        ("ok", [1, 0], 0),
        ("tag", ["a", 3], 1),
    ],
)
def test_incompatible_values_name_the_row(column, values, row):
    assert f"Column '{column}'" in mismatch(GOOD.assign(**{column: values}))
    assert f"row {row} holds" in mismatch(GOOD.assign(**{column: values}))


def test_extra_columns_allowed():
    check_input_frame(GOOD.assign(unused=[1, 2]), SPEC)


def test_output_wrong_length():
    with pytest.raises(SignatureMismatchError) as info:
        check_output_frame(output(["circle"] * 3, [0.5] * 3), 8, SPEC)
    assert info.value.what == "predict returned 3 predictions for 8 input rows"


def test_output_unknown_label():
    with pytest.raises(SignatureMismatchError, match=r"\['oval', 'star'\]"):
        check_output_frame(output(["circle", "star", "oval"], [0.5] * 3), 3, SPEC)


def test_output_confidence_out_of_range():
    with pytest.raises(SignatureMismatchError, match="row 1"):
        check_output_frame(output(["circle"] * 2, [0.5, 1.2]), 2, SPEC)


def test_output_nan_confidence():
    with pytest.raises(SignatureMismatchError, match="row 0"):
        check_output_frame(output(["circle"] * 2, [np.nan, 0.5]), 2, SPEC)


def test_output_wrong_columns():
    with pytest.raises(SignatureMismatchError, match="columns"):
        check_output_frame(pd.DataFrame({"label": ["circle"], "confidence": [0.5]}), 1, SPEC)
