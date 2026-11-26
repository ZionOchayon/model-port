import ast
import re
from pathlib import Path

import pytest

from modelport import errors

SUBCLASSES = [
    c for c in vars(errors).values() if isinstance(c, type) and issubclass(c, errors.ModelportError)
]
K8S_WORDS = re.compile(r"\b(pods?|namespaces?|helm|kubectl|rayjobs?)\b", re.IGNORECASE)
SDK_SRC = Path(__file__).resolve().parents[2] / "src" / "modelport"


def test_str_contains_what_and_fix():
    err = errors.SpecError("model.yaml not found", "Create model.yaml.", details={"path": "x"})
    assert "model.yaml not found" in str(err)
    assert "How to fix: Create model.yaml." in str(err)
    assert err.details == {"path": "x"}
    assert errors.ModelportError("a", "b").details == {}


def test_exit_codes_unique():
    codes = [c.exit_code for c in SUBCLASSES if c is not errors.ModelportError]
    assert len(codes) == 11
    assert len(set(codes)) == len(codes)
    assert not {0, 1} & set(codes)
    assert errors.ModelportError.exit_code == 1


@pytest.mark.parametrize("name", ["SpecError", "WaitTimeoutError"])
def test_errors_are_exceptions_with_class_exit_code(name):
    cls = getattr(errors, name)
    with pytest.raises(errors.ModelportError) as info:
        raise cls("w", "f")
    assert info.value.exit_code == cls.exit_code


def message_literals():
    """Every string literal passed as what= or fix= anywhere in the SDK source."""
    for path in SDK_SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in {"what", "fix"}:
                        for const in ast.walk(kw.value):
                            if isinstance(const, ast.Constant) and isinstance(const.value, str):
                                yield path.name, const.value


def test_no_k8s_words_in_sdk_messages():
    found = [(f, s) for f, s in message_literals() if K8S_WORDS.search(s)]
    assert found == []
