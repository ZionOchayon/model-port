"""A rerun with the same run name resumes from the last checkpoint instead of starting over (FR-5)."""

import pytest

pytestmark = pytest.mark.integration
ray = pytest.importorskip("ray")
pytest.importorskip("torch")

from test_torch_trainer_local import MODEL_DIR, fashion_like_table  # noqa: E402

from modelport.spec import load_spec  # noqa: E402
from modelport_platform.train.strategies import STRATEGIES, TrainContext, run_config  # noqa: E402


def fit(tmp_path, epochs, settings):
    spec = load_spec(MODEL_DIR / "model.fashion.yaml").model_copy(
        update={
            "train": load_spec(MODEL_DIR / "model.fashion.yaml").train.model_copy(
                update={"epochs": epochs, "batch_size": 50}
            )
        }
    )
    data = ray.data.from_pandas(fashion_like_table(200))
    ctx = TrainContext(
        spec=spec,
        model_dir=MODEL_DIR,
        run_name="resume",
        mlflow_run_id="",
        datasets={"train": data, "val": data},
        settings=settings,
    )
    trainer = STRATEGIES[spec.framework].build_trainer(
        ctx, run=run_config(ctx, storage_path=str(tmp_path), storage_filesystem=None)
    )
    return trainer.fit()


def test_rerun_resumes_at_next_epoch(local_ray, tmp_path, platform_settings):
    first = fit(tmp_path, 2, platform_settings)
    assert first.error is None
    assert (first.metrics["epoch"], first.metrics["resumed_from"]) == (1, 0)
    second = fit(tmp_path, 3, platform_settings)
    assert second.error is None
    assert (second.metrics["epoch"], second.metrics["resumed_from"]) == (2, 2)
