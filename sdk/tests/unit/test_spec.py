import pytest
import yaml

from modelport.errors import SpecError
from modelport.spec import Framework, MetricName, load_spec, spec_sha256

MINIMAL = {
    "name": "shape-classifier",
    "class": "model:ShapeClassifier",
    "framework": "torch",
    "labels": ["circle", "square"],
    "signature": {"inputs": [{"name": "image", "type": "binary"}]},
    "data": {"dataset": "s3://datasets/shapes"},
}


def write(tmp_path, doc, name="model.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def with_(**changes):
    doc = {**MINIMAL}
    for dotted, value in changes.items():
        *parents, leaf = dotted.split("__")
        node = doc
        for p in parents:
            node[p] = dict(node.get(p, {}))
            node = node[p]
        node[leaf] = value
    return doc


def spec_error(tmp_path, doc):
    with pytest.raises(SpecError) as info:
        load_spec(write(tmp_path, doc))
    return info.value


def test_minimal_spec_gets_defaults(tmp_path):
    spec = load_spec(write(tmp_path, MINIMAL))
    assert spec.framework is Framework.TORCH
    assert spec.class_ == "model:ShapeClassifier"
    assert spec.resources.train.workers == 2
    assert spec.resources.serve.memory == "1Gi"
    assert spec.train.epochs == 3
    assert spec.evaluation.metric is MetricName.F1_MACRO
    assert spec.evaluation.max_latency_increase == 0.20
    assert spec.serving.canary_weight == 0.1
    assert spec.drift.psi_critical == 0.25
    assert spec.data.label_column == "label"
    assert spec.runtime.pip == ()
    assert spec.data.dataset_name == "shapes"
    assert spec.data.pinned_version is None


def test_unknown_field_rejected(tmp_path):
    err = spec_error(tmp_path, with_(train__epoch=3))
    assert "train.epoch" in err.what


def test_unpinned_pip_rejected(tmp_path):
    err = spec_error(tmp_path, with_(runtime__pip=["torch==2.15.0", "torch>=2.0"]))
    expected = 'runtime.pip[1]: "torch>=2.0" must pin an exact version with == (e.g. torch==2.15.0)'
    assert expected in err.details["errors"]


def test_psi_critical_must_exceed_warning(tmp_path):
    err = spec_error(tmp_path, with_(drift__psi_warning=0.3, drift__psi_critical=0.25))
    assert "psi_critical" in err.what


def test_dataset_uri_with_version_parsed(tmp_path):
    spec = load_spec(write(tmp_path, with_(data__dataset="s3://datasets/fashion-mnist/v12/")))
    assert spec.data.dataset_name == "fashion-mnist"
    assert spec.data.pinned_version == 12


@pytest.mark.parametrize(
    "uri", ["s3://predictions/shapes", "s3://datasets/Shapes", "s3://datasets/shapes/v0"]
)
def test_dataset_uri_bad_bucket_rejected(tmp_path, uri):
    assert "data.dataset" in spec_error(tmp_path, with_(data__dataset=uri)).what


def test_workers_below_two_rejected(tmp_path):
    err = spec_error(tmp_path, with_(resources__train={"workers": 1}))
    assert "resources.train.workers" in err.what


def test_error_message_lists_field_paths(tmp_path):
    doc = with_(labels=["only-one"], signature={"inputs": [{"name": "image", "type": "jpeg"}]})
    err = spec_error(tmp_path, doc)
    paths = [line.split(":")[0] for line in err.details["errors"]]
    assert paths == ["labels", "signature.inputs[0].type"]
    assert all(p in err.what for p in paths)
    assert err.fix


@pytest.mark.parametrize(
    ("doc", "fragment"),
    [
        (with_(labels=["a", "a"]), "labels"),
        (with_(signature={"inputs": [{"name": "x", "type": "double"}] * 2}), "unique"),
        (with_(signature={"inputs": [{"name": "label", "type": "string"}]}), "label_column"),
        (with_(train__epochs="3"), "train.epochs"),
        (with_(resources__train={"workers": True}), "resources.train.workers"),
    ],
)
def test_cross_field_and_strict_types(tmp_path, doc, fragment):
    assert fragment in spec_error(tmp_path, doc).what


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(SpecError) as info:
        load_spec(tmp_path / "model.yaml")
    assert "not found" in info.value.what
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(SpecError) as info:
        load_spec(bad)
    assert "line" in info.value.what


def test_spec_sha256_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "a.yaml", tmp_path / "b.yaml"
    lf.write_bytes(b"name: x\nclass: y\n")
    crlf.write_bytes(b"name: x\r\nclass: y\r\n")
    assert spec_sha256(lf) == spec_sha256(crlf)
