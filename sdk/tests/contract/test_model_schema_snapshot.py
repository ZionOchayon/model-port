import copy
import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from modelport.errors import SpecError
from modelport.spec import export_json_schema, load_spec

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "sdk" / "src" / "modelport" / "schema" / "model.schema.json"
NORMATIVE = json.loads((Path(__file__).parent / "model.schema.normative.json").read_text(encoding="utf-8"))

VALID = {
    "name": "shape-classifier",
    "class": "pkg.model:Shape_1",
    "framework": "xgboost",
    "labels": ["a", "b"],
    "signature": {"inputs": [{"name": "x", "type": "double"}]},
    "runtime": {"pip": ["xgboost==3.4.1", "ray[data]==2.60.1"]},
    "resources": {"train": {"workers": 8, "cpus_per_worker": 0.5, "memory_per_worker": "512Mi"}},
    "train": {"params": {"rounds": 5, "tree_method": "hist", "flags": [1, "a"], "none": None, "on": True}},
    "evaluation": {"metric": "accuracy", "min_improvement": -0.02},
    "serving": {"canary_weight": 0.5, "auto_deploy": False},
    "drift": {"psi_warning": 0.1, "psi_critical": 0.2, "bins": 3},
    "data": {"dataset": "s3://datasets/covertype/v2", "label_column": "Cover_Type", "auto_retrain": False},
}


def mutate(path, value):
    doc = copy.deepcopy(VALID)
    *parents, leaf = path
    node = doc
    for p in parents:
        node = node.setdefault(p, {})
    if value is DELETE:
        del node[leaf]
    else:
        node[leaf] = value
    return doc


DELETE = object()
# Documents that differ from VALID in one place; JSON Schema can express each rule.
CASES = [
    ("valid", VALID),
    ("missing name", mutate(["name"], DELETE)),
    ("bad name", mutate(["name"], "Shape")),
    ("bad class", mutate(["class"], "model.Shape")),
    ("bad framework", mutate(["framework"], "sklearn")),
    ("one label", mutate(["labels"], ["a"])),
    ("duplicate labels", mutate(["labels"], ["a", "a"])),
    ("empty label", mutate(["labels"], ["a", ""])),
    ("no inputs", mutate(["signature", "inputs"], [])),
    ("bad column type", mutate(["signature", "inputs"], [{"name": "x", "type": "jpeg"}])),
    ("unpinned pip", mutate(["runtime", "pip"], ["torch>=2"])),
    ("one worker", mutate(["resources", "train", "workers"], 1)),
    ("nine workers", mutate(["resources", "train", "workers"], 9)),
    ("zero cpus", mutate(["resources", "train", "cpus_per_worker"], 0)),
    ("bad quantity", mutate(["resources", "train", "memory_per_worker"], "2G")),
    ("float epochs", mutate(["train", "epochs"], 2.5)),
    ("string epochs", mutate(["train", "epochs"], "2")),
    ("bool workers", mutate(["resources", "train", "workers"], True)),
    ("object param", mutate(["train", "params"], {"nested": {"a": 1}})),
    ("bad metric", mutate(["evaluation", "metric"], "auc")),
    ("canary weight too big", mutate(["serving", "canary_weight"], 0.6)),
    ("few repeats", mutate(["evaluation", "latency_repeats"], 2)),
    ("bad dataset", mutate(["data", "dataset"], "s3://datasets/x")),
    ("bad label column", mutate(["data", "label_column"], "label-1")),
    ("unknown top field", mutate(["owner"], "me")),
    ("unknown nested field", mutate(["serving", "weight"], 0.1)),
    ("schema version 2", mutate(["schema_version"], 2)),
]


def accepted_by_schema(schema, doc):
    return jsonschema.Draft202012Validator(schema).is_valid(doc)


def accepted_by_spec(tmp_path, doc):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    try:
        load_spec(path)
    except SpecError:
        return False
    return True


def test_exported_schema_matches_committed_file():
    exported = json.dumps(export_json_schema(), indent=2, sort_keys=True) + "\n"
    assert COMMITTED.read_text(encoding="utf-8") == exported


@pytest.mark.parametrize(("case", "doc"), CASES, ids=[c for c, _ in CASES])
def test_normative_generated_and_spec_agree(tmp_path, case, doc):
    expected = accepted_by_schema(NORMATIVE, doc)
    assert accepted_by_schema(export_json_schema(), doc) == expected
    assert accepted_by_spec(tmp_path, doc) == expected
    assert expected == (case == "valid")


def spec_files():
    return sorted(ROOT.glob("examples/models/*/model*.yaml")) + sorted(
        ROOT.glob("sdk/tests/fixtures/broken_models/*/model.yaml")
    )


@pytest.mark.parametrize("path", spec_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_examples_validate_against_json_schema(tmp_path, path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = accepted_by_schema(NORMATIVE, doc)
    assert accepted_by_schema(export_json_schema(), doc) == expected
    if "examples" in path.parts:
        assert expected, "every example model.yaml must be valid"
        load_spec(path)
