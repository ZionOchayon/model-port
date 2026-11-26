"""Shared SDK test fixtures."""

import pytest


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run in an empty directory, so no real .modelport/credentials.env is discovered."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
