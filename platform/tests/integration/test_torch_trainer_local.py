"""Spike (risk R5): TorchTrainer with DDP over gloo on local Ray, checkpoint per epoch, and resume.

The train loop is inline on purpose; the example image model replaces it once it exists.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")
# Ray workers cannot import this test module, so ship the inline train loop by value.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

ROWS = 600


def train_loop(config):
    import ray.train
    import torch
    import torch.distributed as dist
    from ray.train import Checkpoint
    from ray.train.torch import prepare_model

    model = prepare_model(torch.nn.Linear(4, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    start = 0
    if (ckpt := ray.train.get_checkpoint()) is not None:
        with ckpt.as_directory() as d:
            start = json.loads((Path(d) / "meta.json").read_text())["epoch"] + 1
            model.module.load_state_dict(torch.load(Path(d) / "model.pt", weights_only=True))
    rank = ray.train.get_context().get_world_rank()
    shard = ray.train.get_dataset_shard("train")
    for epoch in range(start, config["epochs"]):
        seen = 0
        for batch in shard.iter_torch_batches(batch_size=50):
            x, y = batch["x"].float(), batch["y"].long()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seen += len(y)
        total = torch.tensor([seen])
        dist.all_reduce(total)
        metrics = {"epoch": epoch, "rows_rank": seen, "rows_total": int(total.item()), "start": start}
        if rank == 0:
            with tempfile.TemporaryDirectory() as tmp:
                torch.save(model.module.state_dict(), Path(tmp) / "model.pt")
                (Path(tmp) / "meta.json").write_text(json.dumps({"epoch": epoch}))
                ray.train.report(metrics, checkpoint=Checkpoint.from_directory(tmp))
        else:
            ray.train.report(metrics)


def fit(storage, epochs):
    import numpy as np
    import pandas as pd
    from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": list(rng.normal(size=(ROWS, 4))), "y": rng.integers(0, 3, ROWS)})
    trainer = TorchTrainer(
        train_loop,
        train_loop_config={"epochs": epochs},
        torch_config=TorchConfig(backend="gloo"),
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False, resources_per_worker={"CPU": 1}),
        run_config=RunConfig(
            name="spike",
            storage_path=str(storage),
            failure_config=FailureConfig(max_failures=2),
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
        datasets={"train": ray.data.from_pandas(frame)},
    )
    return trainer.fit()


def test_torch_trainer_ddp_gloo_checkpoints_and_resume(local_ray, tmp_path):
    first = fit(tmp_path, epochs=2)
    assert first.error is None
    assert first.metrics["epoch"] == 1
    assert len(first.best_checkpoints) == 2
    assert first.metrics["rows_total"] == ROWS
    assert first.metrics["rows_rank"] == ROWS // 2

    resumed = fit(tmp_path, epochs=3)
    assert resumed.error is None
    assert resumed.metrics["epoch"] == 2
    assert resumed.metrics["start"] == 2
