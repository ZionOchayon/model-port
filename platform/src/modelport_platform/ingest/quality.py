"""Decide whether an incoming file is valid for its dataset contract (pure core: no I/O).

The caller reads the Parquet file; this module only inspects the `pyarrow.Table` (pyarrow as a data
structure, never `pyarrow.fs`). Every check runs, so a report lists every problem at once.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Self

import pyarrow as pa

from modelport.spec import ColumnType

CheckName = Literal["readable", "schema", "rows", "nulls", "classes", "class_balance", "images"]
CHECK_ORDER: Final[tuple[CheckName, ...]] = (
    "readable",
    "schema",
    "rows",
    "nulls",
    "classes",
    "class_balance",
    "images",
)
MAGIC: Final = {"png": b"\x89PNG\r\n\x1a\n", "jpeg": b"\xff\xd8\xff"}
UNREADABLE: Final = "file could not be read"


@dataclass(frozen=True)
class DatasetContract:
    """What every file of a dataset must look like (s3://datasets/<dataset>/contract.json)."""

    dataset: str
    label_column: str
    columns: Mapping[str, ColumnType]  # includes the label column
    min_rows: int = 1000
    min_classes: int = 2
    max_class_share: float = 0.6
    image_format: Literal["png", "jpeg"] | None = None

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Self:
        """Build from contract.json.

        Raises:
            ValueError: no columns, or the label column is not one of them.
        """
        columns = {name: ColumnType(kind) for name, kind in payload["columns"].items()}
        if not columns or payload["label_column"] not in columns:
            raise ValueError("contract.json must list the columns, including label_column")
        return cls(
            dataset=payload["dataset"],
            label_column=payload["label_column"],
            columns=columns,
            min_rows=int(payload.get("min_rows", 1000)),
            min_classes=int(payload.get("min_classes", 2)),
            max_class_share=float(payload.get("max_class_share", 0.6)),
            image_format=payload.get("image_format"),
        )


@dataclass(frozen=True)
class QualityCheck:
    """One check of a file."""

    name: CheckName
    passed: bool
    detail: str


@dataclass(frozen=True)
class QualityReport:
    """Every check of one incoming file."""

    dataset: str
    source_uri: str
    sha256: str
    rows: int
    class_counts: Mapping[str, int]
    checks: tuple[QualityCheck, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """All checks passed."""
        return all(c.passed for c in self.checks)

    def to_json(self) -> dict[str, Any]:
        """The report as stored next to a quarantined file and inside a version's manifest."""
        return {
            "dataset": self.dataset,
            "source_uri": self.source_uri,
            "sha256": self.sha256,
            "rows": self.rows,
            "class_counts": dict(self.class_counts),
            "passed": self.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
        }


_TYPE_CHECKS: Final = {
    ColumnType.BINARY: lambda t: pa.types.is_binary(t) or pa.types.is_large_binary(t),
    ColumnType.STRING: lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    ColumnType.DOUBLE: pa.types.is_floating,
    ColumnType.FLOAT: pa.types.is_floating,
    ColumnType.LONG: pa.types.is_integer,
    ColumnType.INTEGER: pa.types.is_integer,
    ColumnType.BOOLEAN: pa.types.is_boolean,
}


def _schema(table: pa.Table, contract: DatasetContract) -> QualityCheck:
    problems = []
    for name, kind in contract.columns.items():
        if name not in table.column_names:
            problems.append(f"column '{name}' missing")
        elif not _TYPE_CHECKS[kind](table.schema.field(name).type):
            problems.append(f"column '{name}' is {table.schema.field(name).type}, contract says {kind}")
    return QualityCheck(
        "schema", not problems, "; ".join(problems) or "all contract columns present with their types"
    )


def _images(table: pa.Table, contract: DatasetContract) -> QualityCheck:
    if contract.image_format is None:
        return QualityCheck("images", True, "no image columns")
    magic = MAGIC[contract.image_format]
    binary = [  # a wrongly typed column is the schema check's finding, not this one's
        n
        for n, k in contract.columns.items()
        if k is ColumnType.BINARY and n in table.column_names and _TYPE_CHECKS[k](table.schema.field(n).type)
    ]
    bad = sorted(
        {
            i
            for name in binary
            for i, v in enumerate(table.column(name).to_pylist())
            if not (v or b"").startswith(magic)
        }
    )
    detail = (
        f"{len(bad)} values are not {contract.image_format}; first rows {bad[:5]}"
        if bad
        else "all images valid"
    )
    return QualityCheck("images", not bad, detail)


def check_table(
    table: pa.Table | None, read_error: str | None, contract: DatasetContract, *, source_uri: str, sha256: str
) -> QualityReport:
    """Run every check; an unreadable file fails all of them."""
    if table is None or read_error is not None:
        failed = [QualityCheck("readable", False, read_error or UNREADABLE)]
        failed += [QualityCheck(name, False, UNREADABLE) for name in CHECK_ORDER[1:]]
        return QualityReport(contract.dataset, source_uri, sha256, 0, {}, tuple(failed))
    label = contract.label_column
    has_label = label in table.column_names
    counts = (
        Counter(str(v) for v in table.column(label).to_pylist() if v is not None) if has_label else Counter()
    )
    rows = table.num_rows
    present = [n for n in contract.columns if n in table.column_names]
    null_columns = [n for n in present if table.column(n).null_count]
    top_share = max(counts.values()) / rows if counts and rows else 1.0
    missing_label = f"label column '{label}' missing"
    checks = (
        QualityCheck("readable", True, f"{rows} rows"),
        _schema(table, contract),
        QualityCheck(
            "rows",
            rows >= contract.min_rows,
            f"{rows} {'>=' if rows >= contract.min_rows else '<'} {contract.min_rows}",
        ),
        QualityCheck("nulls", not null_columns, f"nulls in {null_columns}" if null_columns else "no nulls"),
        QualityCheck(
            "classes",
            len(counts) >= contract.min_classes,
            f"{len(counts)} classes (need {contract.min_classes})" if has_label else missing_label,
        ),
        QualityCheck(
            "class_balance",
            has_label and top_share <= contract.max_class_share,
            f"largest class share {top_share:.2f} (max {contract.max_class_share})"
            if has_label
            else missing_label,
        ),
        _images(table, contract),
    )
    return QualityReport(contract.dataset, source_uri, sha256, rows, dict(counts), checks)
