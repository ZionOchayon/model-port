import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from modelport.spec import load_spec
from modelport_platform.drift.simulate import Perturbation, check_perturbation, perturb

ROOT = Path(__file__).resolve().parents[4]
IMAGE_SPEC = load_spec(ROOT / "examples/models/image-classifier/model.fashion.yaml")
TABULAR_SPEC = load_spec(ROOT / "examples/models/covertype-classifier/model.yaml")


def png(value, size=8):
    buffer = io.BytesIO()
    Image.fromarray(np.full((size, size), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def pixels(raw):
    return np.asarray(Image.open(io.BytesIO(raw)), dtype=np.float64)


def images(*values):
    return pd.DataFrame({"image": [png(v) for v in values]})


def test_brightness_increases_mean():
    out = perturb(images(40, 100, 200), IMAGE_SPEC, Perturbation.BRIGHTNESS)
    means = [pixels(raw).mean() for raw in out["image"]]
    assert means == [80.0, 200.0, 255.0]  # x2, clipped at 255


def test_invert_png_roundtrip():
    out = perturb(images(40), IMAGE_SPEC, Perturbation.INVERT)
    assert out["image"][0][:8] == b"\x89PNG\r\n\x1a\n"
    assert pixels(out["image"][0]).mean() == 215.0


def test_noise_is_seeded():
    frame = images(100, 100)
    first = perturb(frame, IMAGE_SPEC, Perturbation.NOISE, seed=1)
    assert list(first["image"]) == list(perturb(frame, IMAGE_SPEC, Perturbation.NOISE, seed=1)["image"])
    assert pixels(first["image"][0]).std() > 10


def test_shift_feature_moves_mean_three_std():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({c.name: rng.normal(100, 10, 500) for c in TABULAR_SPEC.signature.inputs})
    out = perturb(frame, TABULAR_SPEC, Perturbation.SHIFT_FEATURE, feature="slope")
    assert out["slope"].mean() == pytest.approx(frame["slope"].mean() + 3 * frame["slope"].std())
    assert out["elevation"].equals(frame["elevation"])
    default = perturb(frame, TABULAR_SPEC, Perturbation.SHIFT_FEATURE)  # the first numeric input
    assert default["elevation"].mean() > frame["elevation"].mean() + 2 * frame["elevation"].std()


def test_perturbation_must_fit_the_model():
    with pytest.raises(ValueError, match="image perturbation"):
        check_perturbation(TABULAR_SPEC, Perturbation.BRIGHTNESS, None)
    with pytest.raises(ValueError, match="shift_feature"):
        check_perturbation(IMAGE_SPEC, Perturbation.SHIFT_FEATURE, None)
    with pytest.raises(ValueError, match="no numeric input 'nope'"):
        check_perturbation(TABULAR_SPEC, Perturbation.SHIFT_FEATURE, "nope")
    check_perturbation(TABULAR_SPEC, Perturbation.SHIFT_FEATURE, "slope")
    check_perturbation(IMAGE_SPEC, Perturbation.INVERT, None)
