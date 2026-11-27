"""The small API a `train_loop` uses inside Ray Train: its configuration, data shards, reports and resume.

Ray is imported inside each function, so importing the SDK (and `modelport validate`) never needs Ray.
Only rank 0 writes artifacts and MLflow metrics.
"""

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from modelport.errors import ContractError
from modelport.logs import get_logger

if TYPE_CHECKING:
    import ray.data
    from ray.train.xgboost import RayTrainReportCallback

RUN_ID_ENV = "MODELPORT_MLFLOW_RUN_ID"
_log = get_logger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """What `BaseModel.train_loop` receives: settings from model.yaml for this training run."""

    model_name: str
    labels: tuple[str, ...]
    label_column: str
    epochs: int
    batch_size: int
    learning_rate: float
    params: Mapping[str, Any]
    num_workers: int


class _MetricsSink(Protocol):
    def log_metrics(self, run_id: str, metrics: Mapping[str, float], step: int) -> None: ...


def _metrics_sink() -> _MetricsSink:
    from modelport.registry import registry_from_env

    return registry_from_env()


def _context() -> Any:  # noqa: ANN401 - ray.train.TrainContext (Ray ships no stubs)
    import ray.train

    try:
        return ray.train.get_context()
    except Exception:  # Ray raises different errors outside a Train worker across versions
        raise ContractError(
            what="report() was called outside training", fix="Call it only inside train_loop."
        ) from None


def world_rank() -> int:
    """This worker's rank (0 on the worker that writes artifacts)."""
    return int(_context().get_world_rank())


def world_size() -> int:
    """Number of training workers."""
    return int(_context().get_world_size())


def dataset_shard(split: Literal["train", "val"]) -> "ray.data.DataIterator":
    """This worker's shard of the train or val split."""
    import ray.train

    shard: ray.data.DataIterator = ray.train.get_dataset_shard(split)
    return shard


def report(metrics: Mapping[str, float], artifacts_dir: Path | None = None) -> None:
    """Report one epoch; rank 0 also saves `artifacts_dir` as the checkpoint and logs metrics to MLflow.

    Args:
        metrics: must include `epoch` (int, 0-based).
        artifacts_dir: folder with everything `load()` needs to resume or serve (rank 0 only).

    Raises:
        ContractError: no `epoch` metric, or called outside a Ray Train worker.
    """
    if not isinstance(metrics.get("epoch"), int):
        raise ContractError(
            what="report() needs an 'epoch' metric (an int, starting at 0)",
            fix="Call modelport.report({'epoch': epoch, ...}) once per epoch.",
        )
    rank = _context().get_world_rank()
    import ray.train

    if rank == 0 and artifacts_dir is not None:
        ray.train.report(dict(metrics), checkpoint=ray.train.Checkpoint.from_directory(str(artifacts_dir)))
    else:
        ray.train.report(dict(metrics))
    if rank == 0:
        _log_epoch(metrics, step=int(metrics["epoch"]), checkpoint_saved=artifacts_dir is not None)


def _log_epoch(metrics: Mapping[str, float], *, step: int, checkpoint_saved: bool) -> None:
    """Log one epoch's metrics to the MLflow training run (rank 0 only; a no-op outside the platform)."""
    run_id = os.environ.get(RUN_ID_ENV)
    if not run_id:
        return
    values = {k: float(v) for k, v in metrics.items()}
    if checkpoint_saved:
        values["checkpoint_saved"] = 1.0
    try:
        _metrics_sink().log_metrics(run_id, values, step=step)
    except Exception as err:  # metrics are evidence, not state: never stop training for them
        _log.warning("report.mlflow_failed", run_id=run_id, error=f"{type(err).__name__}: {err}")


def latest_artifacts() -> Path | None:
    """The last reported checkpoint as a local folder, or None on the first attempt (resume support)."""
    import ray.train

    checkpoint = ray.train.get_checkpoint()
    if checkpoint is None:
        return None
    return Path(checkpoint.to_directory(tempfile.mkdtemp(prefix="modelport-ckpt-")))


def xgboost_callback() -> "RayTrainReportCallback":
    """Ray's XGBoost callback that checkpoints every boosting round (an "epoch" for XGBoost).

    Rank 0 also logs each round's metrics and `checkpoint_saved` to the MLflow run, as `report()` does.
    """
    from ray.train.xgboost import RayTrainReportCallback

    class _Callback(RayTrainReportCallback):
        def after_iteration(self, model: Any, epoch: int, evals_log: dict[str, Any]) -> bool:  # noqa: ANN401
            super().after_iteration(model, epoch, evals_log)
            if world_rank() == 0:
                round_metrics: dict[str, float] = self._get_report_dict(evals_log)  # type: ignore[no-untyped-call]
                metrics = {"epoch": epoch, **round_metrics}
                _log_epoch(metrics, step=epoch, checkpoint_saved=True)
            return False  # never stop training early

    return _Callback(frequency=1)
