"""Load a model class by name (`class: <module>:<Class>` in model.yaml) from a model folder."""

import importlib
import inspect
import sys
from pathlib import Path

from modelport.base import BaseModel
from modelport.errors import ContractError
from modelport.spec import ModelSpec


def _forget_foreign_module(top: str, model_dir: Path) -> None:
    """Drop a cached module of the same name loaded from another folder (two models may both use `model`)."""
    for name in [m for m in sys.modules if m == top or m.startswith(f"{top}.")]:
        file = getattr(sys.modules[name], "__file__", None)
        if file is None or not Path(file).resolve().is_relative_to(model_dir):
            del sys.modules[name]


def _module_file(model_dir: Path, module: str) -> Path | None:
    base = model_dir.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def load_model_class(model_dir: Path, spec: ModelSpec) -> type[BaseModel]:
    """Import `<module>:<Class>` from `model_dir` and check it is a concrete BaseModel subclass.

    Raises:
        ContractError: the module or class is missing, fails to import, or breaks the contract.
    """
    model_dir = model_dir.resolve()
    module_name, class_name = spec.class_.split(":")
    if _module_file(model_dir, module_name) is None:
        raise ContractError(
            what=f"Module '{module_name}' not found in {model_dir}",
            fix="Check `class:` in model.yaml; it must be <file without .py>:<ClassName>.",
        )
    _forget_foreign_module(module_name.split(".")[0], model_dir)
    if str(model_dir) in sys.path:
        sys.path.remove(str(model_dir))
    sys.path.insert(0, str(model_dir))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        raise ContractError(
            what=f"Importing {module_name} failed: package '{err.name}' is not installed",
            fix=f"Add '{err.name}' with an exact version (name==x.y.z) to runtime.pip in model.yaml.",
        ) from err
    except Exception as err:
        raise ContractError(
            what=f"Importing {module_name} raised {type(err).__name__}: {err}",
            fix=f"Fix the error at import time in {module_name}.py (run `python -c 'import {module_name}'`).",
        ) from err
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ContractError(
            what=f"Class '{class_name}' not found in {module_name}",
            fix="Check `class:` in model.yaml; it must name a class defined in that module.",
        )
    if not (inspect.isclass(cls) and issubclass(cls, BaseModel)):
        raise ContractError(
            what=f"{class_name} does not inherit from modelport.BaseModel",
            fix=f"Declare it as `class {class_name}(BaseModel):` with `from modelport import BaseModel`.",
        )
    missing = sorted(getattr(cls, "__abstractmethods__", ()))
    if missing:
        raise ContractError(
            what=f"{class_name} does not implement {', '.join(missing)}",
            fix="Define " + " and ".join(f"`def {m}(self, ...)`" for m in missing) + ".",
        )
    return cls


def instantiate(model_dir: Path, spec: ModelSpec) -> BaseModel:
    """Load the class and create the model with `cls(spec)`.

    Raises:
        ContractError: as `load_model_class`, or `__init__` needs arguments the platform cannot pass.
    """
    cls = load_model_class(model_dir, spec)
    try:
        return cls(spec)
    except TypeError as err:
        raise ContractError(
            what=f"{cls.__name__}(spec) failed: {err}",
            fix="Do not add required arguments to __init__; read settings from self.spec.train.params.",
        ) from err
