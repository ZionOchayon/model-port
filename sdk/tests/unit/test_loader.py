import sys
import textwrap

import pytest

from modelport import BaseModel
from modelport.errors import ContractError
from modelport.loader import instantiate, load_model_class
from modelport.spec import parse_spec

GOOD = textwrap.dedent("""
    from modelport import BaseModel, Predictions

    class {cls}(BaseModel):
        def load(self, artifacts_dir): pass
        def preprocess(self, batch): return batch
        def predict(self, inputs):
            return Predictions(labels=["a"] * len(inputs), confidence=[1.0] * len(inputs))
""")


def spec(cls="model:Shape"):
    return parse_spec(
        {
            "name": "shape-classifier",
            "class": cls,
            "framework": "torch",
            "labels": ["a", "b"],
            "signature": {"inputs": [{"name": "x", "type": "double"}]},
            "data": {"dataset": "s3://datasets/shapes"},
        }
    )


def model_dir(tmp_path, source, name="model.py", folder="m"):
    d = tmp_path / folder
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(textwrap.dedent(source), encoding="utf-8")
    return d


def contract_error(d, cls="model:Shape"):
    with pytest.raises(ContractError) as info:
        load_model_class(d, spec(cls))
    assert info.value.fix
    return info.value


def test_loads_class_from_dir(tmp_path):
    cls = load_model_class(model_dir(tmp_path, GOOD.format(cls="Shape")), spec())
    assert issubclass(cls, BaseModel)
    assert cls.__name__ == "Shape"


def test_loads_class_from_subpackage(tmp_path):
    d = model_dir(tmp_path, GOOD.format(cls="Shape"), name="pkg/model.py")
    (d / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    assert load_model_class(d, spec("pkg.model:Shape")).__name__ == "Shape"


def test_module_missing(tmp_path):
    err = contract_error(model_dir(tmp_path, GOOD.format(cls="Shape"), name="other.py"))
    assert "Module 'model' not found" in err.what


def test_class_missing(tmp_path):
    err = contract_error(model_dir(tmp_path, GOOD.format(cls="Circle")))
    assert "Shape" in err.what


def test_not_subclass(tmp_path):
    err = contract_error(model_dir(tmp_path, "class Shape:\n    pass\n"))
    assert err.what == "Shape does not inherit from modelport.BaseModel"


def test_abstract_methods_listed(tmp_path):
    source = "from modelport import BaseModel\nclass Shape(BaseModel):\n    def load(self, d): pass\n"
    err = contract_error(model_dir(tmp_path, source))
    assert "predict" in err.what
    assert "preprocess" in err.what


def test_import_error_suggests_runtime_pip(tmp_path):
    err = contract_error(model_dir(tmp_path, "import not_a_real_package_xyz\n"))
    assert "not_a_real_package_xyz" in err.what
    assert "runtime.pip" in err.fix


def test_two_model_dirs_same_module_name_do_not_collide(tmp_path):
    a = load_model_class(
        model_dir(tmp_path, GOOD.format(cls="Shape") + "\nORIGIN = 'a'\n", folder="a"), spec()
    )
    b = load_model_class(
        model_dir(tmp_path, GOOD.format(cls="Shape") + "\nORIGIN = 'b'\n", folder="b"), spec()
    )
    assert sys.modules[a.__module__] is not None
    assert (a.__module__, b.__module__) == ("model", "model")
    assert sys.modules["model"].ORIGIN == "b"
    assert a is not b


def test_instantiate_rejects_required_init_args(tmp_path):
    source = (
        GOOD.format(cls="Shape") + "\n    def __init__(self, spec, extra):\n        super().__init__(spec)\n"
    )
    with pytest.raises(ContractError) as info:
        instantiate(model_dir(tmp_path, source), spec())
    assert "self.spec.train.params" in info.value.fix
