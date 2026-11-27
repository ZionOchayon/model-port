import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from modelport.naming import Tag
from modelport.registry import VersionInfo
from modelport.spec import load_spec
from modelport_platform.serve.config import (
    ServedModel,
    ServedVariant,
    parse_memory,
    render_serve_config,
    served_model_from_spec,
)

ROOT = Path(__file__).resolve().parents[4]
SNAPSHOTS = Path(__file__).resolve().parents[1] / "snapshots"
COVERTYPE = load_spec(ROOT / "examples" / "models" / "covertype-classifier" / "model.yaml")
FASHION = load_spec(ROOT / "examples" / "models" / "image-classifier" / "model.fashion.yaml")
ENV = {"MLFLOW_TRACKING_URI": "http://mlflow.platform.svc.cluster.local:5000"}


def info(name, version, pip=()):
    tags = {Tag.RUNTIME_PIP: json.dumps(list(pip))}
    return VersionInfo(name, version, "run", "src", tags, (), datetime(2026, 1, 1, tzinfo=UTC))


def covertype(candidate=None, weight=0.0, fault=0.0):
    return served_model_from_spec(
        COVERTYPE,
        info("covertype-classifier", "2", ["xgboost==3.4.1"]),
        candidate,
        candidate_weight=weight,
        fault_error_rate=fault,
    )


def snapshot(name, rendered):
    path = SNAPSHOTS / name
    assert yaml.safe_dump(rendered, sort_keys=False) == path.read_text(encoding="utf-8")


def test_single_model_no_candidate_snapshot():
    rendered = render_serve_config([covertype()], env_vars=ENV)
    snapshot("serve_config_single.yaml", rendered)
    router = rendered["applications"][0]["deployments"][0]["user_config"]
    assert router["candidate_app"] is None
    assert router["candidate_weight"] == 0.0
    assert [a["name"] for a in rendered["applications"]] == [
        "covertype-classifier",
        "covertype-classifier--champion",
    ]


def test_canary_snapshot():
    candidate = info("covertype-classifier", "3", ["xgboost==3.4.1"])
    rendered = render_serve_config([covertype(candidate, weight=0.1, fault=1.0)], env_vars=ENV)
    snapshot("serve_config_canary.yaml", rendered)
    apps = {a["name"]: a for a in rendered["applications"]}
    assert apps["covertype-classifier--candidate"]["args"] == {
        "model": "covertype-classifier",
        "version": "3",
        "variant": "candidate",
    }
    # the drill's fault rate applies to the candidate only
    assert apps["covertype-classifier--candidate"]["deployments"][0]["user_config"]["fault_error_rate"] == 1.0
    assert apps["covertype-classifier--champion"]["deployments"][0]["user_config"]["fault_error_rate"] == 0.0


def test_two_models_sorted_and_own_runtime_env():
    fashion = served_model_from_spec(FASHION, info("fashion-classifier", "4"), None, candidate_weight=0.0)
    rendered = render_serve_config([covertype(), fashion], env_vars=ENV)
    names = [a["name"] for a in rendered["applications"]]
    assert names == [
        "covertype-classifier",
        "covertype-classifier--champion",
        "fashion-classifier",
        "fashion-classifier--champion",
    ]
    apps = {a["name"]: a for a in rendered["applications"]}
    assert apps["covertype-classifier--champion"]["runtime_env"]["pip"]["packages"] == ["xgboost==3.4.1"]
    assert "pip" not in apps["fashion-classifier--champion"]["runtime_env"]  # nothing beyond the image
    assert apps["fashion-classifier--champion"]["runtime_env"]["env_vars"] == ENV


def test_render_deterministic():
    first = yaml.safe_dump(render_serve_config([covertype()], env_vars=ENV))
    assert first == yaml.safe_dump(render_serve_config([covertype()], env_vars=dict(ENV)))
    assert render_serve_config([], env_vars=ENV)["applications"] == []


def test_weight_bounds():
    variant = ServedVariant(version="1", runtime_pip=(), replicas=1, num_cpus=0.5, memory_bytes=1)
    for weight in (-0.1, 1.1):
        with pytest.raises(ValueError, match="candidate_weight"):
            ServedModel("m", variant, variant, weight, 500.0, 0.15, 0.3, 1.0)
    # without a candidate the weight is forced to 0
    model = ServedModel("m", variant, None, 0.5, 500.0, 0.15, 0.3, 1.0)
    router = render_serve_config([model], env_vars={})["applications"][0]["deployments"][0]["user_config"]
    assert router["candidate_weight"] == 0.0


def test_parse_memory():
    assert parse_memory("1Gi") == 1073741824
    assert parse_memory("512Mi") == 536870912
    with pytest.raises(ValueError, match="1G"):
        parse_memory("1G")


def test_first_version_renders_without_champion():
    candidate = info("covertype-classifier", "1", ["xgboost==3.4.1"])
    model = served_model_from_spec(COVERTYPE, None, candidate, candidate_weight=0.1)
    rendered = render_serve_config([model], env_vars=ENV)
    assert [a["name"] for a in rendered["applications"]] == [
        "covertype-classifier",
        "covertype-classifier--candidate",
    ]
    router = rendered["applications"][0]["deployments"][0]["user_config"]
    assert (router["champion_app"], router["candidate_weight"]) == (None, 1.0)
    with pytest.raises(ValueError, match="neither"):
        served_model_from_spec(COVERTYPE, None, None, candidate_weight=0.0)
