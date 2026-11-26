import os
import zipfile

import pytest

from modelport import packaging
from modelport.errors import PackagingError
from modelport.packaging import build_code_zip


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "shape"
    (d / "example").mkdir(parents=True)
    (d / "model.py").write_text("print('model')\n", encoding="utf-8")
    (d / "model.yaml").write_text("name: shape\n", encoding="utf-8")
    (d / "example" / "input.parquet").write_bytes(b"PAR1\x00PAR1")
    return d


def test_same_dir_same_hash_twice(model_dir, tmp_path):
    first = build_code_zip(model_dir, tmp_path / "out1")
    second = build_code_zip(model_dir, tmp_path / "out2")
    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.files == ("model/example/input.parquet", "model/model.py", "model/model.yaml")


def test_mtime_change_same_hash(model_dir, tmp_path):
    before = build_code_zip(model_dir, tmp_path / "out1").sha256
    os.utime(model_dir / "model.py", (1_000_000_000, 1_000_000_000))
    assert build_code_zip(model_dir, tmp_path / "out2").sha256 == before


def test_content_change_new_hash(model_dir, tmp_path):
    before = build_code_zip(model_dir, tmp_path / "out1").sha256
    (model_dir / "model.py").write_text("print('changed')\n", encoding="utf-8")
    assert build_code_zip(model_dir, tmp_path / "out2").sha256 != before


def test_excludes_pycache(model_dir, tmp_path):
    (model_dir / "__pycache__").mkdir()
    (model_dir / "__pycache__" / "model.cpython-312.pyc").write_bytes(b"x")
    (model_dir / "stale.pyc").write_bytes(b"x")
    (model_dir / ".git").mkdir()
    (model_dir / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (model_dir / ".DS_Store").write_bytes(b"x")
    files = build_code_zip(model_dir, tmp_path / "out").files
    assert not any("__pycache__" in f or f.endswith(".pyc") or ".git" in f or ".DS_Store" in f for f in files)


def test_single_top_level_dir(model_dir, tmp_path):
    package = build_code_zip(model_dir, tmp_path / "out")
    with zipfile.ZipFile(package.path) as zf:
        names = zf.namelist()
        infos = zf.infolist()
    assert {n.split("/")[0] for n in names} == {"model"}
    assert {i.date_time for i in infos} == {(1980, 1, 1, 0, 0, 0)}
    assert {i.external_attr >> 16 for i in infos} == {0o644}
    assert package.path.name == "code.zip"


def test_symlink_outside_rejected(model_dir, tmp_path):
    (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
    try:
        (model_dir / "link.txt").symlink_to(tmp_path / "secret.txt")
    except OSError:
        pytest.skip("symlinks need extra privileges on this OS")
    with pytest.raises(PackagingError) as info:
        build_code_zip(model_dir, tmp_path / "out")
    assert "link.txt" in info.value.what


def test_too_large_rejected(model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(packaging, "MAX_PACKAGE_BYTES", 10)
    with pytest.raises(PackagingError) as info:
        build_code_zip(model_dir, tmp_path / "out")
    assert "Keep datasets and weights out" in info.value.fix
