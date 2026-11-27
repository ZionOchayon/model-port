"""Example 2: a tabular classifier (XGBoost trained with Ray Train's XGBoostTrainer) for forest cover type."""

from pathlib import Path

import numpy as np
import xgboost

import modelport
from modelport import BaseModel, Predictions, TrainConfig

MODEL_FILE = "model.ubj"  # the file RayTrainReportCallback writes into each checkpoint


class CovertypeClassifier(BaseModel):
    """`multi:softprob` over the 12 signature columns; one boosting round counts as one epoch."""

    def load(self, artifacts_dir):
        self.labels = list(self.spec.labels)
        self.columns = [c.name for c in self.spec.signature.inputs]
        self.booster = None
        if artifacts_dir is not None:
            self.booster = xgboost.Booster()
            self.booster.load_model(str(Path(artifacts_dir) / MODEL_FILE))

    def preprocess(self, batch):
        return xgboost.DMatrix(batch[self.columns].astype(np.float32))

    def predict(self, inputs):
        if self.booster is None:  # untrained: every class equally likely
            probs = np.full((inputs.num_row(), len(self.labels)), 1.0 / len(self.labels))
        else:
            probs = self.booster.predict(inputs)
        index = probs.argmax(axis=1)
        return Predictions(labels=[self.labels[i] for i in index], confidence=probs.max(axis=1))

    def train_loop(self, config: TrainConfig):
        self.load(None)
        frame = modelport.training.dataset_shard("train").materialize().to_pandas()
        index = {label: i for i, label in enumerate(config.labels)}
        dtrain = xgboost.DMatrix(
            frame[self.columns].astype(np.float32), label=frame[config.label_column].map(index).to_numpy()
        )
        previous, done = None, 0
        if (resume := modelport.latest_artifacts()) is not None:
            previous = xgboost.Booster()
            previous.load_model(str(resume / MODEL_FILE))
            done = previous.num_boosted_rounds()
        params = {
            "objective": "multi:softprob",
            "num_class": len(config.labels),
            "tree_method": "hist",
            "eta": config.learning_rate,
            "max_depth": int(config.params.get("max_depth", 8)),
            "eval_metric": "mlogloss",
        }
        xgboost.train(
            params,
            dtrain,
            num_boost_round=max(config.epochs - done, 0),
            evals=[(dtrain, "train")],
            xgb_model=previous,
            callbacks=[modelport.training.xgboost_callback()],
        )
