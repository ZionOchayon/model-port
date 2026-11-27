"""How each framework trains on Ray Train (Strategy): Torch DDP over gloo, or XGBoost's V2 trainer.

Both run the same `worker_entry`, which builds the user's model on each worker and calls its `train_loop`.
Ray is imported inside functions, so this module imports without Ray.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from modelport import TrainConfig
from modelport.loader import instantiate
from modelport.naming import checkpoint_root
from modelport.spec import Framework, ModelSpec
from modelport.training import RUN_ID_ENV
from modelport_platform.s3 import arrow_filesystem
from modelport_platform.settings import PlatformSettings

if TYPE_CHECKING:
    import ray.train

_DEFAULT: Final = object()  # storage not overridden: use s3://checkpoints/<model>


class TrainerLike(Protocol):
    """A Ray Train trainer."""

    def fit(self) -> Any:  # noqa: ANN401 - ray.train.Result
        """Run training and return the result."""


@dataclass(frozen=True)
class TrainContext:
    """Everything a strategy needs to build a trainer for one training run."""

    spec: ModelSpec
    model_dir: Path
    run_name: str  # = RayJob name (the checkpoint folder)
    mlflow_run_id: str
    datasets: Mapping[str, Any]  # split -> ray.data.Dataset
    settings: PlatformSettings


class TrainerStrategy(Protocol):
    """Builds the trainer for one framework."""

    framework: Framework

    def build_trainer(self, ctx: TrainContext, run: "ray.train.RunConfig | None" = None) -> TrainerLike:
        """The trainer for this context (`run` overrides the RunConfig, e.g. local storage in tests)."""


def scaling_config(spec: ModelSpec) -> "ray.train.ScalingConfig":
    """One Train worker per worker pod, CPU only."""
    from ray.train import ScalingConfig

    train = spec.resources.train
    return ScalingConfig(
        num_workers=train.workers, use_gpu=False, resources_per_worker={"CPU": train.cpus_per_worker}
    )


def run_config(
    ctx: TrainContext,
    *,
    storage_path: str | None = None,
    storage_filesystem: Any = _DEFAULT,  # noqa: ANN401
) -> "ray.train.RunConfig":
    """Checkpoints in s3://checkpoints/<model>/<run name>, keep 2, up to `train.max_failures` restarts."""
    from ray.train import CheckpointConfig, FailureConfig, RunConfig

    if storage_filesystem is _DEFAULT:
        storage_filesystem = arrow_filesystem(ctx.settings)
    return RunConfig(
        name=ctx.run_name,
        storage_path=storage_path or checkpoint_root(ctx.spec.name).removeprefix("s3://"),
        storage_filesystem=storage_filesystem,
        failure_config=FailureConfig(max_failures=ctx.spec.train.max_failures),
        checkpoint_config=CheckpointConfig(num_to_keep=2),
    )


def worker_entry(config: dict[str, Any]) -> None:
    """`train_loop_per_worker` for both strategies: load the user's class on this worker and run its loop."""
    from modelport.spec import ModelSpec

    spec = ModelSpec.model_validate_json(config["spec_json"])
    if config.get("mlflow_run_id"):
        os.environ[RUN_ID_ENV] = config["mlflow_run_id"]  # lets modelport.report log metrics on rank 0
    model = instantiate(Path(config["model_dir"]), spec)
    model.train_loop(
        TrainConfig(
            model_name=spec.name,
            labels=tuple(spec.labels),
            label_column=spec.data.label_column,
            epochs=spec.train.epochs,
            batch_size=spec.train.batch_size,
            learning_rate=spec.train.learning_rate,
            params=dict(spec.train.params),
            num_workers=spec.resources.train.workers,
        )
    )


def _loop_config(ctx: TrainContext) -> dict[str, Any]:
    return {
        "spec_json": ctx.spec.model_dump_json(by_alias=True),
        "model_dir": str(ctx.model_dir),
        "mlflow_run_id": ctx.mlflow_run_id,
    }


class TorchStrategy:
    """PyTorch DDP over gloo (CPU)."""

    framework = Framework.TORCH

    def build_trainer(self, ctx: TrainContext, run: "ray.train.RunConfig | None" = None) -> TrainerLike:
        """A `TorchTrainer` around `worker_entry`."""
        from ray.train.torch import TorchConfig, TorchTrainer

        trainer: TrainerLike = TorchTrainer(
            worker_entry,
            train_loop_config=_loop_config(ctx),
            torch_config=TorchConfig(backend="gloo"),
            scaling_config=scaling_config(ctx.spec),
            run_config=run or run_config(ctx),
            datasets=dict(ctx.datasets),
        )
        return trainer


class XGBoostStrategy:
    """XGBoost with Ray Train's V2 trainer (the user's loop calls `xgboost.train`)."""

    framework = Framework.XGBOOST

    def build_trainer(self, ctx: TrainContext, run: "ray.train.RunConfig | None" = None) -> TrainerLike:
        """An `XGBoostTrainer` around `worker_entry`."""
        from ray.train.xgboost import XGBoostTrainer

        trainer: TrainerLike = XGBoostTrainer(
            worker_entry,
            train_loop_config=_loop_config(ctx),
            scaling_config=scaling_config(ctx.spec),
            run_config=run or run_config(ctx),
            datasets=dict(ctx.datasets),
        )
        return trainer


STRATEGIES: Final[Mapping[str, TrainerStrategy]] = {
    Framework.TORCH: TorchStrategy(),
    Framework.XGBOOST: XGBoostStrategy(),
}
