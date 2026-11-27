import numpy as np
import pandas as pd

from modelport_platform.gate.evaluate import measure_latency, predict_batches


class Model:
    def __init__(self):
        self.calls = []

    def run_inference(self, batch):
        self.calls.append(len(batch))
        return pd.DataFrame(
            {"prediction": np.where(batch["x"] > 0.5, "hi", "lo"), "confidence": batch["x"].to_numpy()}
        )


def test_batched_prediction_equals_single_pass():
    frame = pd.DataFrame({"x": np.random.default_rng(0).random(2500)})
    model = Model()
    batched = predict_batches(model, frame, batch_size=1024)
    assert model.calls == [1024, 1024, 452]
    pd.testing.assert_frame_equal(batched, Model().run_inference(frame))


def test_latency_repeats_and_batch_size():
    model = Model()
    ticks = iter([0.0, 0.010, 1.0, 1.030, 2.0, 2.020])
    median = measure_latency(
        model, pd.DataFrame({"x": np.zeros(500)}), batch_size=256, repeats=3, clock=lambda: next(ticks)
    )
    assert model.calls == [256, 256, 256, 256]  # one warm-up + 3 timed
    assert abs(median - 20.0) < 1e-9
