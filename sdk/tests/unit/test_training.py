from collections import OrderedDict

import pytest

from modelport import training
from modelport.errors import ContractError


class FakeContext:
    def __init__(self, rank):
        self.rank = rank

    def get_world_rank(self):
        return self.rank

    def get_world_size(self):
        return 2


@pytest.fixture
def fake_train(monkeypatch):
    """Ray Train's session functions, recorded."""
    import ray.train

    calls = {"report": []}
    state = {"rank": 0}
    monkeypatch.setattr(ray.train, "get_context", lambda: FakeContext(state["rank"]))
    monkeypatch.setattr(
        ray.train, "report", lambda metrics, checkpoint=None: calls["report"].append((metrics, checkpoint))
    )
    return calls, state


class FakeRegistry:
    def __init__(self, fail=False):
        self.logged = []
        self.fail = fail

    def log_metrics(self, run_id, metrics, step):
        if self.fail:
            raise ConnectionError("mlflow down")
        self.logged.append((run_id, dict(metrics), step))


def test_report_requires_epoch(fake_train):
    with pytest.raises(ContractError) as info:
        training.report({"loss": 0.3})
    assert "epoch" in info.value.what


def test_report_outside_training_raises():
    with pytest.raises(ContractError) as info:
        training.report({"epoch": 0})
    assert info.value.what == "report() was called outside training"
    assert info.value.fix == "Call it only inside train_loop."


def test_rank0_reports_checkpoint_and_logs_metrics(fake_train, tmp_path, monkeypatch):
    calls, _ = fake_train
    registry = FakeRegistry()
    monkeypatch.setenv("MODELPORT_MLFLOW_RUN_ID", "run-1")
    monkeypatch.setattr(training, "_metrics_sink", lambda: registry)
    (tmp_path / "w.pt").write_bytes(b"x")
    training.report({"epoch": 1, "loss": 0.25}, artifacts_dir=tmp_path)
    ((metrics, checkpoint),) = calls["report"]
    assert metrics == {"epoch": 1, "loss": 0.25}
    assert checkpoint is not None
    assert registry.logged == [("run-1", {"epoch": 1.0, "loss": 0.25, "checkpoint_saved": 1.0}, 1)]


def test_other_ranks_report_without_checkpoint_or_mlflow(fake_train, tmp_path, monkeypatch):
    calls, state = fake_train
    state["rank"] = 1
    registry = FakeRegistry()
    monkeypatch.setenv("MODELPORT_MLFLOW_RUN_ID", "run-1")
    monkeypatch.setattr(training, "_metrics_sink", lambda: registry)
    training.report({"epoch": 0, "loss": 0.5}, artifacts_dir=tmp_path)
    assert calls["report"] == [({"epoch": 0, "loss": 0.5}, None)]
    assert registry.logged == []
    assert (training.world_rank(), training.world_size()) == (1, 2)


def test_mlflow_failure_does_not_stop_training(fake_train, monkeypatch):
    calls, _ = fake_train
    monkeypatch.setenv("MODELPORT_MLFLOW_RUN_ID", "run-1")
    monkeypatch.setattr(training, "_metrics_sink", lambda: FakeRegistry(fail=True))
    training.report({"epoch": 0})
    assert len(calls["report"]) == 1


def test_latest_artifacts_none_without_checkpoint(monkeypatch):
    import ray.train

    monkeypatch.setattr(ray.train, "get_checkpoint", lambda: None)
    assert training.latest_artifacts() is None


def test_xgboost_callback_checkpoints_every_round():
    callback = training.xgboost_callback()
    assert callback._frequency == 1


class FakeBooster:
    def save_model(self, path):
        with open(path, "w") as f:
            f.write("model")

    def num_boosted_rounds(self):
        return 3


def test_xgboost_callback_logs_checkpoint_evidence_on_rank0(fake_train, monkeypatch):
    calls, _ = fake_train
    registry = FakeRegistry()
    monkeypatch.setenv("MODELPORT_MLFLOW_RUN_ID", "run-1")
    monkeypatch.setattr(training, "_metrics_sink", lambda: registry)
    callback = training.xgboost_callback()
    log = OrderedDict(train=OrderedDict(mlogloss=[0.9, 0.8, 0.7]))  # xgboost passes OrderedDicts
    callback.after_iteration(FakeBooster(), 2, log)
    assert calls["report"][0][1] is not None  # Ray Train got the checkpoint
    assert registry.logged == [("run-1", {"epoch": 2.0, "train-mlogloss": 0.7, "checkpoint_saved": 1.0}, 2)]


def test_xgboost_callback_other_ranks_do_not_log(fake_train, monkeypatch):
    _, state = fake_train
    state["rank"] = 1
    registry = FakeRegistry()
    monkeypatch.setenv("MODELPORT_MLFLOW_RUN_ID", "run-1")
    monkeypatch.setattr(training, "_metrics_sink", lambda: registry)
    training.xgboost_callback().after_iteration(
        FakeBooster(), 0, OrderedDict(train=OrderedDict(mlogloss=[0.9]))
    )
    assert registry.logged == []
