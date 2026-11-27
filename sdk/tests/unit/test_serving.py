import base64
import json

import pandas as pd
import pytest
from pydantic import ValidationError

from modelport import serving
from modelport.errors import SignatureMismatchError
from modelport.spec import ColumnSpec, ColumnType

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"
IMAGE = [ColumnSpec(name="image", type=ColumnType.BINARY)]
TABULAR = [
    ColumnSpec(name="elevation", type=ColumnType.DOUBLE),
    ColumnSpec(name="soil_type", type=ColumnType.LONG),
    ColumnSpec(name="wet", type=ColumnType.BOOLEAN),
    ColumnSpec(name="note", type=ColumnType.STRING),
]


def test_encode_decode_binary_roundtrip():
    frame = pd.DataFrame({"image": [PNG, b"\xff\xd8\xffjpeg"]})
    request = serving.encode_rows(frame, IMAGE)
    assert request.instances[0]["image"] == base64.b64encode(PNG).decode()
    json.dumps(request.model_dump())  # the request is plain JSON
    back = serving.decode_request(request, IMAGE)
    assert back["image"].tolist() == frame["image"].tolist()


def test_decode_casts_to_signature_types():
    request = serving.PredictRequest(
        instances=[{"elevation": 3000, "soil_type": 4, "wet": True, "note": "a"}]
    )
    frame = serving.decode_request(request, TABULAR)
    assert frame.dtypes.to_dict() == {
        "elevation": "float64",
        "soil_type": "int64",
        "wet": "bool",
        "note": "object",
    }
    assert list(frame.columns) == ["elevation", "soil_type", "wet", "note"]


def test_decode_bad_base64_is_signature_error():
    with pytest.raises(SignatureMismatchError, match="image"):
        serving.decode_request(serving.PredictRequest(instances=[{"image": "not base64!"}]), IMAGE)


def test_decode_missing_column_is_signature_error():
    with pytest.raises(SignatureMismatchError, match="soil_type"):
        serving.decode_request(serving.PredictRequest(instances=[{"elevation": 1.0}]), TABULAR[:2])


def test_too_many_instances_rejected():
    with pytest.raises(ValidationError):
        serving.PredictRequest(instances=[{"x": 1}] * (serving.MAX_INSTANCES + 1))
    with pytest.raises(ValidationError):
        serving.PredictRequest(instances=[])
    with pytest.raises(ValidationError):
        serving.PredictRequest(instances=[{"x": 1}], extra=1)


def test_read_png_file_single_binary_column(tmp_path):
    path = tmp_path / "shirt.png"
    path.write_bytes(PNG)
    frame = serving.read_input_file(path, IMAGE)
    assert frame["image"].tolist() == [PNG]
    with pytest.raises(SignatureMismatchError, match="one binary input"):
        serving.read_input_file(path, TABULAR)


def test_read_csv_types_follow_signature(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("elevation,soil_type,wet,note\n3000,4,true,a\n2900.5,7,false,b\n", encoding="utf-8")
    frame = serving.read_input_file(path, TABULAR)
    assert frame["elevation"].tolist() == [3000.0, 2900.5]
    assert frame["soil_type"].tolist() == [4, 7]
    assert frame["wet"].tolist() == [True, False]
    assert frame["note"].tolist() == ["a", "b"]


def test_read_json_request_or_rows_and_parquet(tmp_path):
    rows = [{"elevation": 1.0, "soil_type": 2, "wet": False, "note": "x"}]
    (tmp_path / "a.json").write_text(json.dumps({"instances": rows}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(rows), encoding="utf-8")
    pd.DataFrame(rows).to_parquet(tmp_path / "c.parquet")
    for name in ("a.json", "b.json", "c.parquet"):
        assert serving.read_input_file(tmp_path / name, TABULAR).to_dict("records") == rows
    with pytest.raises(SignatureMismatchError, match=r"\.txt"):
        serving.read_input_file(tmp_path / "rows.txt", TABULAR)


def test_response_contract():
    response = serving.PredictResponse(
        model="m",
        version="3",
        variant="champion",
        predictions=[serving.PredictionItem(prediction="Bag", confidence=0.9)],
    )
    assert response.model_dump()["predictions"] == [{"prediction": "Bag", "confidence": 0.9}]
    with pytest.raises(ValidationError):
        serving.PredictResponse(model="m", version="3", variant="shadow", predictions=[])
