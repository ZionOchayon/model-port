"""Population Stability Index (pure core): quantile bins for continuous features, one bin per category.

PSI = sum_i (a_i - e_i) * ln(a_i / e_i) over the same bins; epsilon on every bin, then renormalised.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from modelport_platform.drift.baseline import FeatureBaseline

PSI_EPSILON: Final[float] = 1e-4
DEFAULT_BINS: Final[int] = 10
OTHER_CATEGORY: Final = "__other__"


def quantile_edges(baseline: np.ndarray, bins: int = DEFAULT_BINS) -> np.ndarray:
    """Edges at the baseline's quantiles, outer edges -inf/+inf, duplicates collapsed (fewer bins)."""
    inner = np.quantile(np.asarray(baseline, dtype=np.float64), np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(np.concatenate(([-np.inf], inner, [np.inf])))


def shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of `values` in each bin [edges[i], edges[i+1]) (the last bin is closed)."""
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    total = counts.sum()
    return counts / total if total else np.zeros(len(edges) - 1)


def categorical_shares(values: Sequence[str], categories: Sequence[str]) -> np.ndarray:
    """Share of each category, plus a last `__other__` bin for categories not in the baseline."""
    index = {c: i for i, c in enumerate(categories)}
    counts = np.zeros(len(categories) + 1)
    for value in values:
        counts[index.get(value, len(categories))] += 1
    total = counts.sum()
    return counts / total if total else counts


def psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = PSI_EPSILON) -> float:
    """PSI between two share vectors over the same bins."""
    e = np.asarray(expected, dtype=np.float64) + epsilon
    a = np.asarray(actual, dtype=np.float64) + epsilon
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(baseline: "FeatureBaseline", values: np.ndarray | Sequence[str]) -> float:
    """PSI of current `values` against one feature's baseline bins."""
    if baseline.categories is not None:
        actual = categorical_shares([str(v) for v in values], baseline.categories)
    else:
        assert baseline.edges is not None  # continuous features always have edges
        actual = shares(np.asarray(values, dtype=np.float64), np.asarray(baseline.edges))
    return psi(np.asarray(baseline.shares), actual)
