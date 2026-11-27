import itertools

import pandas as pd

from modelport.errors import PlatformUnavailableError, RemoteOperationError
from modelport.serving import PredictionItem, PredictResponse
from modelport.spec import ColumnSpec, ColumnType
from modelport_platform.tools.loadgen import run_load

INPUTS = [ColumnSpec(name="x", type=ColumnType.DOUBLE)]
ROWS = pd.DataFrame({"x": [0.1, 0.2, 0.3], "label": ["a", "b", "a"]})


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Client:
    def __init__(self, clock, answers=None, cost_s=0.01):
        self.clock, self.answers, self.cost_s, self.sent = clock, answers or [], cost_s, []

    def predict(self, model, request):
        self.clock.now += self.cost_s
        self.sent.append((self.clock.now, model, len(request.instances)))
        answer = self.answers[len(self.sent) - 1] if len(self.sent) <= len(self.answers) else "champion"
        if isinstance(answer, Exception):
            raise answer
        items = [PredictionItem(prediction="a", confidence=0.9)] * len(request.instances)
        return PredictResponse(model=model, version="1", variant=answer, predictions=items)


def load(client, clock, **kwargs):
    defaults = {"rps": 5, "duration_s": 2, "batch_rows": 2}
    return run_load(
        client, "m", ROWS, spec_inputs=INPUTS, clock=clock, sleep=clock.sleep, **{**defaults, **kwargs}
    )


def test_rate_respected():
    clock = Clock()
    client = Client(clock)
    result = load(client, clock)
    assert result.sent == 10  # 5 rps for 2 s
    times = [t for t, _, _ in client.sent]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert all(abs(g - 0.2) < 1e-9 for g in gaps)
    assert {n for _, _, n in client.sent} == {2, 1}  # batches cycle over the 3 rows


def test_no_burst_catch_up():
    clock = Clock()
    client = Client(clock, cost_s=0.5)  # slower than the 0.2 s interval
    result = load(client, clock)
    assert result.sent == 4  # requests at 0.5, 1.0, 1.5, 2.0: no burst to make up the missed slots


def test_failures_counted_by_status():
    clock = Clock()
    answers = [
        "champion",
        RemoteOperationError(what="500", fix="-", details={"status": 500}),
        PlatformUnavailableError(what="down", fix="-"),
        "champion",
    ]
    result = load(Client(clock, answers), clock, duration_s=0.8)
    assert (result.sent, result.ok, result.failed) == (4, 2, 2)
    assert result.status_counts == {"200": 2, "500": 1, "conn": 1}
    assert result.latency_ms_p50 is not None


def test_variants_counted():
    clock = Clock()
    result = load(Client(clock, ["champion", "candidate", "candidate", "champion"]), clock, duration_s=0.8)
    assert result.variants == {"champion": 2, "candidate": 2}


def champion_services(serve):
    import importlib.util
    from pathlib import Path

    from modelport.cli.main import Services
    from modelport.naming import Alias, Tag

    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location("sdk_fakes", root / "sdk/tests/fakes.py")
    fakes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fakes)
    reg = fakes.InMemoryRegistry()
    model_yaml = (root / "examples/models/covertype-classifier/model.yaml").read_bytes()
    reg.runs["train-1"] = {"tags": {}, "artifacts": {"spec/model.yaml": model_yaml}}
    reg.add_version("covertype-classifier", {Tag.TRAIN_RUN_ID: "train-1"}, "train-1")
    reg.aliases["covertype-classifier"][Alias.CHAMPION] = "1"
    rows = root / "examples/models/covertype-classifier/example/input.parquet"
    return Services(registry=reg, store=reg, orchestrator=None, settings=None, serve=serve), rows


class Steady:
    def __init__(self, fail_first=0):
        self.calls, self.fail_first = 0, fail_first

    def predict(self, model, request):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise PlatformUnavailableError(what="Cannot reach Serve", fix="Run `make ui`.")
        items = [PredictionItem(prediction="Aspen", confidence=0.9)] * len(request.instances)
        return PredictResponse(model=model, version="1", variant="champion", predictions=items)


def test_main_writes_result_and_exit_code(tmp_path):
    import json

    from modelport_platform.tools.loadgen import main

    services, rows = champion_services(Steady())
    out = tmp_path / "result.json"
    argv = ["--model", "covertype-classifier", "--rows", str(rows), "--rps", "20", "--duration", "0.25"]
    assert main([*argv, "--out", str(out)], services=services) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["failed"] == 0
    assert result["sent"] >= 1  # pacing is tested with a fake clock above
    services, rows = champion_services(Steady(fail_first=1))
    assert main(argv, services=services) == 1  # any failed request fails the run (FR-11)


def test_main_without_champion_exits_with_the_error_code(capsys):
    from modelport_platform.tools.loadgen import main

    services, rows = champion_services(Steady())
    services.registry.aliases["covertype-classifier"].pop("champion")
    assert main(["--model", "covertype-classifier", "--rows", str(rows)], services=services) == 7
    assert "no champion" in capsys.readouterr().err
