import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from modelport.spec import Framework, spec_sha256
from modelport_platform.train import entrypoint

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "examples" / "models" / "image-classifier"


class FakeRegistry:
    def __init__(self):
        self.runs = []

    def ensure_run(self, experiment, run_name, key_tag, tags, params):
        self.runs.append((experiment, run_name, dict(tags)))
        return "train-run-1"

    def run_tags(self, run_id):
        assert run_id == "sub-1"
        return {"modelport.code_sha256": "c" * 64}


class FakeCheckpoint:
    def to_directory(self, path):
        return path


class FakeDataset:
    def __init__(self, split):
        self.split = split

    def to_pandas(self):
        return pd.DataFrame({"image": [b"x"], "label": ["Bag"]})


CHECKPOINT = FakeCheckpoint()


class FakeResult:
    def __init__(self, error=None, checkpoint=CHECKPOINT):
        self.error, self.checkpoint, self.metrics = error, checkpoint, {"epoch": 1}


class FakeStrategy:
    def __init__(self, result):
        self.result = result
        self.contexts = []

    def build_trainer(self, ctx):
        self.contexts.append(ctx)
        strategy = self

        class Trainer:
            def fit(self):
                if isinstance(strategy.result, Exception):
                    raise strategy.result
                return strategy.result

        return Trainer()


def deps(store, settings, result=None, cwd=MODEL):
    strategy = FakeStrategy(result or FakeResult())
    registered = []

    def register(**kwargs):
        registered.append(kwargs)
        return SimpleNamespace(version="7")

    register.calls = registered

    return entrypoint.Deps(
        store=store,
        registry=FakeRegistry(),
        settings=settings,
        strategies={"torch": strategy, "xgboost": strategy},
        read_split=lambda settings, version, split: FakeDataset(split),
        register=register,
        cwd=cwd,
    ), strategy


def args(**over):
    base = [
        "--model",
        "fashion-classifier",
        "--submission-run-id",
        "sub-1",
        "--spec-file",
        "model.fashion.yaml",
        "--rayjob",
        "train-fashion-classifier-1",
    ]
    extra = [x for k, v in over.items() for x in (f"--{k.replace('_', '-')}", str(v))]
    return entrypoint.parse_args(base + extra)


def with_version(store, version=1):
    store.write_bytes(
        f"s3://datasets/fashion-mnist/v{version}/manifest.json",
        json.dumps(
            {"version": version, "sha256": "ab", "rows_train": 90, "rows_val": 10, "labels": ["Bag", "Coat"]}
        ).encode(),
    )


def test_name_mismatch_exit_2(store, platform_settings):
    d, _ = deps(store, platform_settings)
    wrong = entrypoint.parse_args(
        [
            "--model",
            "other-model",
            "--submission-run-id",
            "s",
            "--spec-file",
            "model.fashion.yaml",
            "--rayjob",
            "r",
        ]
    )
    assert entrypoint.train(wrong, d) == 2


def test_no_dataset_version_exit_2(store, platform_settings, capsys):
    d, _ = deps(store, platform_settings)
    assert entrypoint.train(args(), d) == 2
    assert "Dataset 'fashion-mnist' has no versions yet" in capsys.readouterr().err
    with_version(store, 1)
    assert entrypoint.train(args(dataset_version=5), d) == 2


def test_no_checkpoint_exit_1_with_fix(store, platform_settings, capsys):
    with_version(store, 1)
    d, strategy = deps(store, platform_settings, result=FakeResult(checkpoint=None))
    assert entrypoint.train(args(), d) == 1
    err = capsys.readouterr().err
    assert "train_loop finished without reporting artifacts" in err
    assert "modelport.report(metrics, artifacts_dir)" in err
    ctx = strategy.contexts[0]
    assert (ctx.run_name, ctx.mlflow_run_id) == ("train-fashion-classifier-1", "train-run-1")
    assert {k: v.split for k, v in ctx.datasets.items()} == {"train": "train", "val": "val"}
    experiment, run_name, tags = d.registry.runs[0]
    assert (experiment, run_name) == ("modelport-train/fashion-classifier", "train-fashion-classifier-1")
    assert tags["modelport.dataset_version"] == "1"


def test_training_error_exit_1(store, platform_settings):
    with_version(store, 1)
    d, _ = deps(store, platform_settings, result=FakeResult(error=RuntimeError("worker died")))
    assert entrypoint.train(args(), d) == 1


def test_fit_raising_exit_1(store, platform_settings, capsys):
    with_version(store, 1)
    d, _ = deps(store, platform_settings, result=RuntimeError("TrainingFailedError: worker died"))
    assert entrypoint.train(args(), d) == 1
    assert "worker died" in capsys.readouterr().err


def test_pinned_version_used_when_no_override(store, platform_settings):
    with_version(store, 1)
    with_version(store, 2)
    d, _ = deps(store, platform_settings, result=FakeResult(checkpoint=None))
    entrypoint.train(args(dataset_version=1), d)
    assert d.registry.runs[0][2]["modelport.dataset_version"] == "1"
    entrypoint.train(args(), d)
    assert d.registry.runs[1][2]["modelport.dataset_version"] == "2"


def test_success_registers_with_lineage(store, platform_settings):
    with_version(store, 1)
    d, _ = deps(store, platform_settings)
    assert entrypoint.train(args(), d) == 0
    (kwargs,) = d.register.calls
    lineage = kwargs["lineage"]
    assert (lineage.dataset, lineage.dataset_version, lineage.dataset_sha256) == ("fashion-mnist", 1, "ab")
    assert lineage.val_uri == "s3://datasets/fashion-mnist/v1/val.parquet"
    assert (lineage.code_sha256, lineage.spec_sha256) == ("c" * 64, spec_sha256(MODEL / "model.fashion.yaml"))
    assert (lineage.submission_run_id, lineage.train_run_id) == ("sub-1", "train-run-1")
    assert (lineage.train_rayjob, lineage.framework) == ("train-fashion-classifier-1", Framework.TORCH)
    assert lineage.image == platform_settings.ray_image
    assert kwargs["spec_path"] == MODEL / "model.fashion.yaml"
    assert kwargs["model_dir"] == MODEL
    assert list(kwargs["val"].columns) == ["image", "label"]
    assert len(kwargs["example"]) == len(pd.read_parquet(MODEL / "example" / "input.parquet"))
