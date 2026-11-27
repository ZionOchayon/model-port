"""Shared SDK test fixtures."""

import pytest


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run in an empty directory, so no real .modelport/credentials.env is discovered."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def memory_registry():
    """A fresh InMemoryRegistry (loaded by path: the platform tests also have a module named `fakes`)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("sdk_test_fakes", Path(__file__).with_name("fakes.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InMemoryRegistry()
