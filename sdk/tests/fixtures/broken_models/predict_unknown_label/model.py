"""Fixture model for `modelport validate` tests (predict_unknown_label)."""

import numpy as np

from modelport import BaseModel, Predictions


class Toy(BaseModel):
    def load(self, artifacts_dir):
        self.threshold = 0.5

    def preprocess(self, batch):
        return batch["x"].to_numpy()

    def predict(self, inputs):
        return Predictions(
            labels=["c" if v > self.threshold else "a" for v in inputs], confidence=np.full(len(inputs), 0.8)
        )

    def train_loop(self, config):
        pass
