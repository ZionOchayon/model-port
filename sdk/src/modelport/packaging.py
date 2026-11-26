"""Package a model folder into a deterministic zip whose sha256 identifies the submitted code.

The same files give the same bytes on every OS and at any time: entries are sorted, dated 1980-01-01,
mode 0644, deflated at level 6, and stored under one top-level folder `model/` (Ray's rule for remote
`working_dir` zips).
"""

import fnmatch
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from modelport.errors import PackagingError
from modelport.logs import get_logger

EXCLUDE_PATTERNS: Final[tuple[str, ...]] = (
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    ".ipynb_checkpoints",
    "*.egg-info",
    ".DS_Store",
)
MAX_PACKAGE_BYTES: Final = 50 * 1024 * 1024
ARCHIVE_ROOT: Final = "model"
_EPOCH: Final = (1980, 1, 1, 0, 0, 0)
_UNIX: Final = 3  # ZipInfo.create_system; the default differs between Windows and Unix
_log = get_logger(__name__)


@dataclass(frozen=True)
class CodePackage:
    """The built archive, its content hash and its sorted entry names."""

    path: Path
    sha256: str
    files: tuple[str, ...]


def _excluded(relative: Path) -> bool:
    return any(fnmatch.fnmatch(part, pattern) for part in relative.parts for pattern in EXCLUDE_PATTERNS)


def _files(model_dir: Path) -> list[tuple[str, Path]]:
    """(archive name, source path) for every file to package, sorted by archive name."""
    root = model_dir.resolve()
    entries = []
    for path in model_dir.rglob("*"):
        relative = path.relative_to(model_dir)
        if _excluded(relative) or path.is_dir():
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise PackagingError(
                what=f"{relative.as_posix()} links to {path.resolve()}, outside the model folder",
                fix="Copy the file into the model folder or remove the link.",
            )
        entries.append((f"{ARCHIVE_ROOT}/{relative.as_posix()}", path))
    return sorted(entries)


def build_code_zip(model_dir: Path, out_dir: Path) -> CodePackage:
    """Write `out_dir/code.zip` from `model_dir` and return its hash.

    Raises:
        PackagingError: a symlink points outside the folder, or the archive is larger than 50 MB.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "code.zip"
    entries = _files(model_dir)
    with zipfile.ZipFile(target, "w") as zf:
        for name, source in entries:
            info = zipfile.ZipInfo(name, date_time=_EPOCH)
            info.external_attr = 0o644 << 16
            info.create_system = _UNIX
            zf.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    size = target.stat().st_size
    if size > MAX_PACKAGE_BYTES:
        target.unlink()
        raise PackagingError(
            what=f"Model folder is {size / 1024 / 1024:.0f} MB after packaging",
            fix="Keep datasets and weights out of the model folder; "
            "the platform reads data from the dataset in model.yaml.",
        )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    files = tuple(name for name, _ in entries)
    _log.info("packaging.built", sha256=digest, files=len(files), bytes=size)
    return CodePackage(path=target, sha256=digest, files=files)
