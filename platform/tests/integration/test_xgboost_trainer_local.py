"""Ray Train V2 XGBoostTrainer on local Ray, driving the example Covertype model's own train_loop.

Settles risk R4 (xgboost 3.4 with Ray 2.60): 2 workers, a checkpoint every boosting round, and resume.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
xgboost = pytest.importorskip("xgboost")
# Ray workers cannot import this test module, so ship the worker function by value.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "examples" / "models" / "covertype-classifier"
ROWS = 2000


def covertype_like_table(rows: int) -> pd.DataFrame:
    """Rows with the example's signature columns and one of its seven labels."""
    from modelport.spec import load_spec

    spec = load_spec(MODEL_DIR / "model.yaml")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            c.name: rng.normal(size=rows) if c.type == "double" else rng.integers(0, 4, rows)
            for c in spec.signature.inputs
        }
    )
    frame[spec.data.label_column] = [spec.labels[i % len(spec.labels)] for i in range(rows)]
    return frame


def worker(config):
    """What the platform's worker entry does: load the class from the model folder and call train_loop."""
    from modelport import TrainConfig
    from modelport.loader import instantiate
    from modelport.spec import load_spec

    spec = load_spec(Path(config["model_dir"]) / "model.yaml")
    instantiate(Path(config["model_dir"]), spec).train_loop(
        TrainConfig(
            model_name=spec.name,
            labels=tuple(spec.labels),
            label_column=spec.data.label_column,
            epochs=config["rounds"],
            batch_size=spec.train.batch_size,
            learning_rate=spec.train.learning_rate,
            params=dict(spec.train.params),
            num_workers=2,
        )
    )


def fit(storage, rounds):
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.xgboost import XGBoostTrainer

    trainer = XGBoostTrainer(
        worker,
        train_loop_config={"model_dir": str(MODEL_DIR), "rounds": rounds},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False, resources_per_worker={"CPU": 1}),
        run_config=RunConfig(
            name="xgb", storage_path=str(storage), checkpoint_config=CheckpointConfig(num_to_keep=2)
        ),
        datasets={"train": ray.data.from_pandas(covertype_like_table(ROWS))},
    )
    return trainer.fit()


def test_example_covertype_model_checkpoints_per_round_and_resumes(local_ray, tmp_path):
    from modelport.loader import instantiate
    from modelport.spec import load_spec

    first = fit(tmp_path, rounds=5)
    assert first.error is None
    spec = load_spec(MODEL_DIR / "model.yaml")
    model = instantiate(MODEL_DIR, spec)
    with first.checkpoint.as_directory() as d:
        assert (Path(d) / "model.ubj").is_file()
        model.load(Path(d))
    assert model.booster.num_boosted_rounds() == 5
    example = pd.read_parquet(MODEL_DIR / "example" / "input.parquet")
    assert model.booster.predict(model.preprocess(example)).shape == (len(example), 7)

    resumed = fit(tmp_path, rounds=7)
    assert resumed.error is None
    with resumed.checkpoint.as_directory() as d:
        model.load(Path(d))
    assert model.booster.num_boosted_rounds() == 7
