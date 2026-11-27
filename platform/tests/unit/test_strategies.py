from pathlib import Path

import pytest

from modelport.spec import Framework, load_spec
from modelport_platform.train.strategies import (
    STRATEGIES,
    TorchStrategy,
    TrainContext,
    XGBoostStrategy,
    run_config,
    scaling_config,
)

ROOT = Path(__file__).resolve().parents[3]
IMAGE = load_spec(ROOT / "examples/models/image-classifier/model.fashion.yaml")
TABULAR = load_spec(ROOT / "examples/models/covertype-classifier/model.yaml")


def ctx(spec, settings):
    return TrainContext(
        spec=spec,
        model_dir=Path("/tmp/model"),
        run_name="train-x-1",
        mlflow_run_id="r1",
        datasets={},
        settings=settings,
    )


def test_strategy_per_framework():
    assert isinstance(STRATEGIES[Framework.TORCH], TorchStrategy)
    assert isinstance(STRATEGIES[Framework.XGBOOST], XGBoostStrategy)
    assert set(STRATEGIES) == set(Framework)


def test_run_config_uses_checkpoint_root_and_failure_config(platform_settings):
    pytest.importorskip("ray.train")
    config = run_config(ctx(IMAGE, platform_settings))
    assert config.name == "train-x-1"
    assert config.storage_path == "checkpoints/fashion-classifier"
    assert config.failure_config.max_failures == IMAGE.train.max_failures
    assert config.checkpoint_config.num_to_keep == 2
    local = run_config(ctx(IMAGE, platform_settings), storage_path="/tmp/ckpt", storage_filesystem=None)
    assert (local.storage_path, local.storage_filesystem) == ("/tmp/ckpt", None)


def test_scaling_config_from_spec():
    pytest.importorskip("ray.train")
    config = scaling_config(TABULAR)
    assert config.num_workers == 2
    assert config.use_gpu is False
    assert config.resources_per_worker == {"CPU": 1.0}
