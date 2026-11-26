import numpy as np
import pandas as pd
import pytest

from modelport import BaseModel, Predictions
from modelport.errors import ContractError, SignatureMismatchError
from modelport.spec import parse_spec

SPEC = parse_spec(
    {
        "name": "toy-model",
        "class": "model:Toy",
        "framework": "xgboost",
        "labels": ["a", "b"],
        "signature": {"inputs": [{"name": "x", "type": "double"}, {"name": "tag", "type": "string"}]},
        "data": {"dataset": "s3://datasets/toy"},
    }
)
FRAME = pd.DataFrame({"x": [0.1, 0.9, 0.4], "tag": ["p", "q", "r"]})


class Toy(BaseModel):
    calls: list[str]

    def load(self, artifacts_dir):
        self.calls = ["load"]

    def preprocess(self, batch):
        self.calls.append("preprocess")
        return batch["x"].to_numpy()

    def predict(self, inputs):
        self.calls.append("predict")
        return Predictions(
            labels=["b" if v > 0.5 else "a" for v in inputs], confidence=np.full(len(inputs), 0.7)
        )


def model(cls=Toy):
    m = cls(SPEC)
    m.load(None)
    return m


def test_run_inference_order_and_output_columns():
    m = model()
    out = m.run_inference(FRAME)
    assert m.calls == ["load", "preprocess", "predict"]
    assert list(out.columns) == ["prediction", "confidence"]
    assert out["prediction"].tolist() == ["a", "b", "a"]
    assert out["confidence"].dtype == np.float64
    assert m.spec is SPEC


def test_predict_wrong_type_raises_contract_error():
    class Bad(Toy):
        def predict(self, inputs):
            return {"labels": ["a"]}

    with pytest.raises(ContractError) as info:
        model(Bad).run_inference(FRAME)
    assert "not Predictions" in info.value.what


def test_preprocess_exception_wrapped():
    class Bad(Toy):
        def preprocess(self, batch):
            raise KeyError("pixels")

    with pytest.raises(ContractError) as info:
        model(Bad).run_inference(FRAME)
    assert "preprocess raised KeyError" in info.value.what
    assert "x, tag" in info.value.fix
    assert isinstance(info.value.__cause__, KeyError)


def test_predict_exception_wrapped():
    class Bad(Toy):
        def predict(self, inputs):
            raise RuntimeError("boom")

    with pytest.raises(ContractError, match="predict raised RuntimeError"):
        model(Bad).run_inference(FRAME)


def test_predictions_length_mismatch():
    class Bad(Toy):
        def predict(self, inputs):
            return Predictions(labels=["a"] * 2, confidence=[0.5] * 2)

    with pytest.raises(SignatureMismatchError) as info:
        model(Bad).run_inference(FRAME)
    assert info.value.what == "predict returned 2 predictions for 3 input rows"


def test_labels_and_confidence_lengths_differ():
    class Bad(Toy):
        def predict(self, inputs):
            return Predictions(labels=["a"] * 3, confidence=[0.5] * 2)

    with pytest.raises(SignatureMismatchError, match="3 labels but 2 confidences"):
        model(Bad).run_inference(FRAME)


def test_input_checked_before_preprocess():
    m = model()
    with pytest.raises(SignatureMismatchError, match="missing column 'tag'"):
        m.run_inference(FRAME[["x"]])
    assert m.calls == ["load"]


def test_default_input_stats_numeric_only():
    stats = model().input_stats(FRAME.assign(extra=[1, 2, 3]))
    assert list(stats.columns) == ["x"]
    assert stats["x"].dtype == np.float64


def test_implements_train_loop_detects_override():
    class Trains(Toy):
        def train_loop(self, config):
            pass

    assert not Toy.implements_train_loop()
    assert Trains.implements_train_loop()
    with pytest.raises(NotImplementedError):
        model().train_loop(None)
