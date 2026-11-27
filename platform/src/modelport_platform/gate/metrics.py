"""Classification metrics for the gate (pure core, numpy only)."""

from collections.abc import Sequence

import numpy as np

from modelport.spec import MetricName


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Counts with rows = true label, columns = predicted label; pairs with an unknown label are ignored."""
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.divide(num, den, out=np.zeros(len(num), dtype=np.float64), where=den > 0)
    return out


def _f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    return _ratio(2 * tp, cm.sum(axis=0) + cm.sum(axis=1))  # 2TP / (2TP + FP + FN); 0 when undefined


def per_class_f1(cm: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    """F1 of every class (0.0 for a class never predicted and never present)."""
    return dict(zip(labels, (float(v) for v in _f1(cm)), strict=True))


def per_class_recall(cm: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    """Recall of every class."""
    recall = _ratio(np.diag(cm).astype(np.float64), cm.sum(axis=1))
    return dict(zip(labels, (float(v) for v in recall), strict=True))


def f1_macro(cm: np.ndarray) -> float:
    """Unweighted mean of the per-class F1 scores."""
    return float(np.mean(_f1(cm))) if len(cm) else 0.0


def accuracy(cm: np.ndarray) -> float:
    """Share of correct predictions."""
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


def score(
    metric: MetricName, y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> tuple[float, dict[str, float]]:
    """f1_macro -> (macro F1, per-class F1); accuracy -> (accuracy, per-class recall)."""
    cm = confusion_matrix(y_true, y_pred, labels)
    if metric is MetricName.F1_MACRO:
        return f1_macro(cm), per_class_f1(cm, labels)
    return accuracy(cm), per_class_recall(cm, labels)
