"""Deliberately drifted input for the drift demo (`simulate_drift`, FR-16).

Images (a model with exactly one binary input) are decoded, changed and written back as PNG; tabular models
get one numeric feature shifted by three standard deviations.
"""

import io
from enum import StrEnum

import numpy as np
import pandas as pd
from PIL import Image

from modelport.spec import ColumnType, ModelSpec

NUMERIC = {ColumnType.DOUBLE, ColumnType.FLOAT, ColumnType.LONG, ColumnType.INTEGER}


class Perturbation(StrEnum):
    """What `perturb` does to the rows."""

    BRIGHTNESS = "brightness"  # images: pixel values x2, clipped
    NOISE = "noise"  # images: + N(0, 40) noise
    INVERT = "invert"  # images: 255 - x, an input unlike any training image
    SHIFT_FEATURE = "shift_feature"  # tabular: feature + 3 x its std


def _image_column(spec: ModelSpec) -> str | None:
    inputs = spec.signature.inputs
    return inputs[0].name if len(inputs) == 1 and inputs[0].type is ColumnType.BINARY else None


def _numeric(spec: ModelSpec) -> list[str]:
    return [c.name for c in spec.signature.inputs if c.type in NUMERIC]


def check_perturbation(spec: ModelSpec, kind: Perturbation, feature: str | None) -> None:
    """Raise ValueError, with what to do instead, when `kind` does not fit the model."""
    image = _image_column(spec) is not None
    if kind is Perturbation.SHIFT_FEATURE:
        if image:
            raise ValueError(f"{spec.name} takes images: use brightness, noise or invert, not shift_feature")
        numeric = _numeric(spec)
        if feature is not None and feature not in numeric:
            raise ValueError(f"{spec.name} has no numeric input '{feature}' (numeric inputs: {numeric})")
        if not numeric:
            raise ValueError(f"{spec.name} has no numeric input to shift")
    elif not image:
        raise ValueError(f"{kind} is an image perturbation and {spec.name} takes no image: use shift_feature")


def _change(raw: bytes, kind: Perturbation, rng: np.random.Generator) -> bytes:
    with Image.open(io.BytesIO(raw)) as img:
        pixels = np.asarray(img, dtype=np.float64)
    if kind is Perturbation.BRIGHTNESS:
        pixels = pixels * 2.0
    elif kind is Perturbation.NOISE:
        pixels = pixels + rng.normal(0, 40, pixels.shape)
    else:
        pixels = 255 - pixels
    buffer = io.BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def perturb(
    frame: pd.DataFrame, spec: ModelSpec, kind: Perturbation, *, feature: str | None = None, seed: int = 0
) -> pd.DataFrame:
    """A drifted copy of `frame` (checked with `check_perturbation` first)."""
    check_perturbation(spec, kind, feature)
    out = frame.copy()
    column = _image_column(spec)
    if column is not None:
        rng = np.random.default_rng(seed)
        out[column] = [_change(raw, kind, rng) for raw in frame[column]]
        return out
    name = feature or _numeric(spec)[0]
    out[name] = frame[name] + 3 * frame[name].std()
    return out
