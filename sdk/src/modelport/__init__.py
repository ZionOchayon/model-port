"""modelport SDK: subclass BaseModel, write model.yaml, run `modelport submit`."""

from importlib.metadata import version

from modelport.base import BaseModel, Predictions
from modelport.training import TrainConfig

__version__ = version("modelport")

__all__ = ["BaseModel", "Predictions", "TrainConfig", "__version__"]
