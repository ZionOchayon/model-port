import pyarrow as pa
import pytest

from modelport_platform.ingest.quality import DatasetContract, check_table

PNG = b"\x89PNG\r\n\x1a\n" + b"rest"
CONTRACT = DatasetContract.from_json(
    {
        "dataset": "fashion-mnist",
        "label_column": "label",
        "columns": {"image": "binary", "label": "string"},
        "min_rows": 4,
        "min_classes": 2,
        "max_class_share": 0.6,
        "image_format": "png",
    }
)


def table(images=None, labels=None):
    labels = labels or ["Bag", "Coat", "Bag", "Coat"]
    images = images or [PNG] * len(labels)
    return pa.table({"image": pa.array(images, pa.binary()), "label": pa.array(labels, pa.string())})


def failed(report):
    return {c.name for c in report.checks if not c.passed}


def check(t, read_error=None):
    return check_table(
        t, read_error, CONTRACT, source_uri="s3://incoming/fashion-mnist/x.parquet", sha256="ab" * 32
    )


def test_valid_table_passes():
    report = check(table())
    assert report.passed
    assert failed(report) == set()
    assert report.rows == 4
    assert report.class_counts == {"Bag": 2, "Coat": 2}
    assert [c.name for c in report.checks] == [
        "readable",
        "schema",
        "rows",
        "nulls",
        "classes",
        "class_balance",
        "images",
    ]


def test_unreadable_file():
    report = check(None, read_error="Parquet magic bytes not found")
    assert failed(report) == {"readable", "schema", "rows", "nulls", "classes", "class_balance", "images"}
    assert "Parquet magic bytes" in report.checks[0].detail
    assert all(c.detail == "file could not be read" for c in report.checks[1:])


def test_missing_column():
    report = check(pa.table({"label": ["Bag", "Coat", "Bag", "Coat"]}))
    assert "schema" in failed(report)
    assert "image" in next(c.detail for c in report.checks if c.name == "schema")


def test_wrong_type():
    t = pa.table({"image": pa.array([1, 2, 3, 4]), "label": ["Bag", "Coat", "Bag", "Coat"]})
    detail = next(c.detail for c in check(t).checks if c.name == "schema")
    assert "image" in detail
    assert "binary" in detail


def test_large_types_accepted():
    t = pa.table(
        {
            "image": pa.array([PNG] * 4, pa.large_binary()),
            "label": pa.array(["Bag", "Coat"] * 2, pa.large_string()),
        }
    )
    assert check(t).passed


def test_too_few_rows():
    assert "rows" in failed(check(table(labels=["Bag", "Coat"])))


def test_nulls():
    assert "nulls" in failed(check(table(labels=["Bag", None, "Bag", "Coat"])))


def test_single_class():
    report = check(table(labels=["Bag"] * 4))
    assert {"classes", "class_balance"} <= failed(report)


def test_class_imbalance():
    report = check(table(labels=["Bag", "Bag", "Bag", "Coat"]))
    assert failed(report) == {"class_balance"}
    assert "0.75" in next(c.detail for c in report.checks if c.name == "class_balance")


def test_bad_image_bytes():
    images = [PNG, b"\xff\xd8\xffjpeg", b"text", PNG]
    detail = next(c.detail for c in check(table(images=images)).checks if c.name == "images")
    assert "[1, 2]" in detail


def test_report_json_roundtrip_keys():
    payload = check(table()).to_json()
    assert payload["passed"] is True
    assert {c["name"] for c in payload["checks"]} >= {"rows", "images"}
    assert payload["sha256"] == "ab" * 32


@pytest.mark.parametrize("bad", [{"columns": {}}, {"columns": {"label": "string"}, "label_column": "y"}])
def test_contract_validation(bad):
    with pytest.raises(ValueError, match="label_column"):
        DatasetContract.from_json(
            {"dataset": "x", "label_column": "label", "columns": {"label": "string"}} | bad
        )
