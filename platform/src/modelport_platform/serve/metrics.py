"""The catalogue's Serve-side model metrics for one version (Ray exports them as `ray_<name>`).

Serve adds `deployment`, `replica` and `application` labels by itself. `predicted_class` is bounded by the
model's labels; no label holds a request ID or user input.
"""

from collections.abc import Mapping

import pandas as pd

from modelport_platform.metrics import names


class ServeMetrics:
    """Histograms, counters and gauges of `ModelDeployment`, tagged with model and version."""

    def __init__(self, model: str, version: str) -> None:
        from ray.util import metrics

        self.model, self.version = model, version
        mv = ("model", "version")
        self._confidence = metrics.Histogram(
            names.PREDICTION_CONFIDENCE,
            description="Confidence of each prediction.",
            boundaries=list(names.CONFIDENCE_BUCKETS),
            tag_keys=mv,
        )
        self._predictions = metrics.Counter(
            names.PREDICTIONS, description="Predictions by class.", tag_keys=(*mv, "predicted_class")
        )
        self._input_stat = metrics.Histogram(
            names.INPUT_STAT,
            description="Per-request mean of each input_stats feature.",
            boundaries=list(names.INPUT_STAT_BUCKETS),
            tag_keys=(*mv, "stat"),
        )
        self._baseline_confidence = metrics.Gauge(
            names.BASELINE_CONFIDENCE_MEAN, description="Mean confidence on the validation set.", tag_keys=mv
        )
        self._baseline_share = metrics.Gauge(
            names.BASELINE_CLASS_SHARE,
            description="Share of each predicted class on the validation set.",
            tag_keys=(*mv, "predicted_class"),
        )
        self._live_threshold = metrics.Gauge(
            names.LIVE_DRIFT_THRESHOLD, description="Live drift thresholds.", tag_keys=("model", "kind")
        )
        self._info = metrics.Gauge(
            names.MODEL_INFO, description="1 for each served version.", tag_keys=(*mv, "variant")
        )

    def _tags(self, **extra: str) -> dict[str, str]:
        return {"model": self.model, "version": self.version, **extra}

    def set_baseline(self, confidence_mean: float, class_shares: Mapping[str, float]) -> None:
        """Gauges the live drift rule compares against."""
        self._baseline_confidence.set(confidence_mean, tags=self._tags())
        for label, share in class_shares.items():
            self._baseline_share.set(share, tags=self._tags(predicted_class=label))

    def set_model_info(self, variant: str) -> None:
        """`modelport_model_info{model, version, variant} = 1`."""
        self._info.set(1, tags=self._tags(variant=variant))

    def set_live_thresholds(self, *, confidence_drop: float, class_share_increase: float) -> None:
        """The thresholds from `drift.live_*` (per model, not per version)."""
        self._live_threshold.set(confidence_drop, tags={"model": self.model, "kind": "confidence_drop"})
        self._live_threshold.set(
            class_share_increase, tags={"model": self.model, "kind": "class_share_increase"}
        )

    def observe(self, output: pd.DataFrame, stats: pd.DataFrame) -> None:
        """One request: every confidence, one count per predicted row, each stat's mean."""
        for confidence in output["confidence"]:
            self._confidence.observe(float(confidence), tags=self._tags())
        for label, count in output["prediction"].value_counts().items():
            self._predictions.inc(int(count), tags=self._tags(predicted_class=str(label)))
        for column in stats.columns:
            self._input_stat.observe(float(stats[column].mean()), tags=self._tags(stat=str(column)))
