"""TorchTrainer with DDP over gloo on local Ray, driving the example image model's own train_loop.

Checks torch 2.15 with Ray 2.60 on the real example: 2 workers, a checkpoint per epoch.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")
# Ray workers cannot import this test module, so ship the worker function by value.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "examples" / "models" / "image-classifier"
ROWS = 600


def fashion_like_table(rows: int) -> pd.DataFrame:
    """Fashion-MNIST-shaped rows: 28x28 grayscale PNG bytes and one of the ten labels."""
    from PIL import Image

    from modelport.spec import load_spec

    labels = load_spec(MODEL_DIR / "model.fashion.yaml").labels
    rng = np.random.default_rng(0)
    images = []
    for _ in range(rows):
        buffer = io.BytesIO()
        Image.fromarray((rng.random((28, 28)) * 255).astype(np.uint8), mode="L").save(buffer, format="PNG")
        images.append(buffer.getvalue())
    return pd.DataFrame({"image": images, "label": [labels[i % len(labels)] for i in range(rows)]})


def worker(config):
    """What the platform's worker entry does: load the class from the model folder and call train_loop."""
    from modelport import TrainConfig
    from modelport.loader import instantiate
    from modelport.spec import load_spec

    spec = load_spec(Path(config["model_dir"]) / "model.fashion.yaml")
    model = instantiate(Path(config["model_dir"]), spec)
    model.train_loop(
        TrainConfig(
            model_name=spec.name,
            labels=tuple(spec.labels),
            label_column=spec.data.label_column,
            epochs=config["epochs"],
            batch_size=50,
            learning_rate=spec.train.learning_rate,
            params=dict(spec.train.params),
            num_workers=2,
        )
    )


def test_example_image_model_trains_with_ddp_and_checkpoints(local_ray, tmp_path):
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    trainer = TorchTrainer(
        worker,
        train_loop_config={"model_dir": str(MODEL_DIR), "epochs": 2},
        torch_config=TorchConfig(backend="gloo"),
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False, resources_per_worker={"CPU": 1}),
        run_config=RunConfig(
            name="torch", storage_path=str(tmp_path), checkpoint_config=CheckpointConfig(num_to_keep=2)
        ),
        datasets={"train": ray.data.from_pandas(fashion_like_table(ROWS))},
    )
    result = trainer.fit()
    assert result.error is None
    assert result.metrics["epoch"] == 1
    assert len(result.best_checkpoints) == 2
    assert result.metrics["train_rows"] == ROWS // 2  # each of the two workers saw half the rows
    with result.checkpoint.as_directory() as d:
        assert {p.name for p in Path(d).iterdir()} >= {"model.pt", "meta.json"}
