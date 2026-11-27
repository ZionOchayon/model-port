"""Example 1: an image classifier (ResNet-18 trained from scratch with PyTorch DDP on Ray Train).

The same class serves Fashion-MNIST (model.fashion.yaml, used in CI) and EuroSAT land use (model.yaml).
"""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

import modelport
from modelport import BaseModel, Predictions, TrainConfig


class ImageClassifier(BaseModel):
    """ResNet-18 over PNG images; `train.params`: image_size, in_channels, shuffle_labels (broken variant)."""

    def load(self, artifacts_dir):
        params = self.spec.train.params
        self.image_size = int(params.get("image_size", 32))
        self.labels = list(self.spec.labels)
        self.net = torchvision.models.resnet18(weights=None, num_classes=len(self.labels))
        if artifacts_dir is not None:
            state = torch.load(Path(artifacts_dir) / "model.pt", weights_only=True, map_location="cpu")
            self.net.load_state_dict(state)
        self.net.eval()

    def _pixels(self, batch):
        """(n, 3, size, size) float32 in [0, 1]; grayscale images are repeated to three channels."""
        size = (self.image_size, self.image_size)
        images = [Image.open(io.BytesIO(raw)).convert("RGB").resize(size) for raw in batch["image"]]
        return np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in images]).transpose(0, 3, 1, 2)

    def preprocess(self, batch):
        return torch.from_numpy(self._pixels(batch))

    def predict(self, inputs):
        with torch.no_grad():
            probs = torch.softmax(self.net(inputs), dim=1)
        confidence, index = probs.max(dim=1)
        return Predictions(labels=[self.labels[i] for i in index.tolist()], confidence=confidence.numpy())

    def input_stats(self, batch):
        pixels = self._pixels(batch)
        return pd.DataFrame(
            {"brightness": pixels.mean(axis=(1, 2, 3)), "contrast": pixels.std(axis=(1, 2, 3))}
        ).astype(np.float64)

    def train_loop(self, config: TrainConfig):
        from ray.train.torch import prepare_model

        self.load(None)
        start = 0
        if (resume := modelport.latest_artifacts()) is not None:
            self.net.load_state_dict(torch.load(resume / "model.pt", weights_only=True, map_location="cpu"))
            start = json.loads((resume / "meta.json").read_text())["epoch"] + 1
        net = prepare_model(self.net)
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
        index = {label: i for i, label in enumerate(config.labels)}
        shuffle = bool(config.params.get("shuffle_labels", False))
        rng = np.random.default_rng(0)
        shard = modelport.training.dataset_shard("train")
        for epoch in range(start, config.epochs):
            net.train()
            losses, rows = [], 0
            for batch in shard.iter_batches(batch_size=config.batch_size, batch_format="pandas"):
                y = np.array([index[label] for label in batch[config.label_column]])
                if shuffle:  # the deliberately broken variant: labels carry no signal
                    y = rng.integers(0, len(config.labels), size=len(y))
                loss = torch.nn.functional.cross_entropy(net(self.preprocess(batch)), torch.from_numpy(y))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                rows += len(y)
            loss_mean = float(np.mean(losses)) if losses else 0.0
            metrics = {"epoch": epoch, "loss": loss_mean, "train_rows": rows, "resumed_from": start}
            if modelport.training.world_rank() == 0:
                with tempfile.TemporaryDirectory() as tmp:
                    torch.save(self.net.state_dict(), Path(tmp) / "model.pt")
                    (Path(tmp) / "meta.json").write_text(json.dumps({"epoch": epoch}))
                    modelport.report(metrics, artifacts_dir=Path(tmp))
            else:
                modelport.report(metrics)
