"""Fixture model for `modelport validate` tests (predict_raises)."""

from modelport import BaseModel


class Toy(BaseModel):
    def load(self, artifacts_dir):
        self.threshold = 0.5

    def preprocess(self, batch):
        return batch["x"].to_numpy()

    def predict(self, inputs):
        raise ValueError("weights not loaded")

    def train_loop(self, config):
        pass
