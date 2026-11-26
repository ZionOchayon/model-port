"""Spike (risk R4): the Ray Train V2 XGBoostTrainer with xgboost 3.4.1, a checkpoint per round, and resume.

The train loop is inline on purpose; the example Covertype model replaces it once it exists.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
xgboost = pytest.importorskip("xgboost")
# Ray workers cannot import this test module, so ship the inline train loop by value.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

ROWS, CLASSES, FEATURES = 2000, 7, 6


def train_loop(config):
    import ray.train
    import xgboost
    from ray.train.xgboost import RayTrainReportCallback

    frame = ray.train.get_dataset_shard("train").materialize().to_pandas()
    dtrain = xgboost.DMatrix(frame.drop(columns="y"), label=frame["y"])
    previous, done = None, 0
    if (ckpt := ray.train.get_checkpoint()) is not None:
        previous = RayTrainReportCallback.get_model(ckpt)
        done = previous.num_boosted_rounds()
    xgboost.train(
        {"objective": "multi:softprob", "num_class": CLASSES, "tree_method": "hist", "max_depth": 3},
        dtrain,
        num_boost_round=config["rounds"] - done,
        xgb_model=previous,
        callbacks=[RayTrainReportCallback(frequency=1)],
    )


def fit(storage, rounds):
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.xgboost import XGBoostTrainer

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(ROWS, FEATURES)), columns=[f"f{i}" for i in range(FEATURES)])
    frame["y"] = rng.integers(0, CLASSES, ROWS)
    trainer = XGBoostTrainer(
        train_loop,
        train_loop_config={"rounds": rounds},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False, resources_per_worker={"CPU": 1}),
        run_config=RunConfig(
            name="spike-xgb", storage_path=str(storage), checkpoint_config=CheckpointConfig(num_to_keep=2)
        ),
        datasets={"train": ray.data.from_pandas(frame)},
    )
    return trainer.fit()


def test_xgboost_trainer_checkpoints_per_round_and_resumes(local_ray, tmp_path):
    from ray.train.xgboost import RayTrainReportCallback

    first = fit(tmp_path, rounds=5)
    assert first.error is None
    with first.checkpoint.as_directory() as d:
        assert (Path(d) / "model.ubj").is_file()
    booster = RayTrainReportCallback.get_model(first.checkpoint)
    assert booster.num_boosted_rounds() == 5
    probs = booster.predict(
        xgboost.DMatrix(pd.DataFrame(np.zeros((3, FEATURES)), columns=[f"f{i}" for i in range(FEATURES)]))
    )
    assert probs.shape == (3, CLASSES)

    resumed = fit(tmp_path, rounds=7)
    assert resumed.error is None
    assert RayTrainReportCallback.get_model(resumed.checkpoint).num_boosted_rounds() == 7
