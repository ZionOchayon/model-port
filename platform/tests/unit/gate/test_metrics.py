import numpy as np
import pytest

from modelport.spec import MetricName
from modelport_platform.gate.metrics import (
    accuracy,
    confusion_matrix,
    f1_macro,
    per_class_f1,
    per_class_recall,
    score,
)

LABELS = ["a", "b", "c"]
Y_TRUE = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
Y_PRED = ["a", "a", "b", "b", "c", "c", "c", "c", "a", "c"]


def test_confusion_matrix_hand_example():
    cm = confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    # rows = true, columns = predicted
    assert cm.tolist() == [[2, 1, 0], [0, 1, 1], [1, 0, 4]]


def test_f1_macro_hand_example():
    cm = confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    # a: P 2/3, R 2/3 -> 2/3; b: P 1/2, R 1/2 -> 1/2; c: P 4/5, R 4/5 -> 4/5
    expected = {"a": 2 / 3, "b": 1 / 2, "c": 4 / 5}
    assert per_class_f1(cm, LABELS) == pytest.approx(expected, abs=1e-12)
    assert f1_macro(cm) == pytest.approx((2 / 3 + 1 / 2 + 4 / 5) / 3, abs=1e-12)


def test_accuracy():
    assert accuracy(confusion_matrix(Y_TRUE, Y_PRED, LABELS)) == pytest.approx(0.7)
    assert accuracy(np.zeros((2, 2))) == 0.0


def test_per_class_recall():
    recall = per_class_recall(confusion_matrix(Y_TRUE, Y_PRED, LABELS), LABELS)
    assert recall == pytest.approx({"a": 2 / 3, "b": 1 / 2, "c": 4 / 5})


def test_class_absent_from_predictions_f1_zero():
    cm = confusion_matrix(["a", "b", "c"], ["a", "a", "a"], LABELS)
    f1 = per_class_f1(cm, LABELS)
    assert (f1["b"], f1["c"]) == (0.0, 0.0)
    assert not np.isnan(f1_macro(cm))


def test_score_per_metric():
    value, per_class = score(MetricName.F1_MACRO, Y_TRUE, Y_PRED, LABELS)
    assert value == pytest.approx((2 / 3 + 1 / 2 + 4 / 5) / 3)
    assert set(per_class) == set(LABELS)
    value, per_class = score(MetricName.ACCURACY, Y_TRUE, Y_PRED, LABELS)
    assert value == pytest.approx(0.7)
    assert per_class["c"] == pytest.approx(0.8)


def test_unknown_labels_ignored_in_matrix():
    cm = confusion_matrix(["a", "zzz"], ["a", "a"], LABELS)
    assert cm.sum() == 1
