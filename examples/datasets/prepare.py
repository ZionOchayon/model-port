"""Download a public dataset, convert it to modelport's Parquet layout, and upload it (or write it locally).

Usage:
  uv run python examples/datasets/prepare.py --dataset fashion-mnist --rows 6000 --upload
  uv run python examples/datasets/prepare.py --dataset fashion-mnist --corrupt --upload
  uv run python examples/datasets/prepare.py --dataset covertype --batch-input --day 2026-01-01 --upload
  uv run python examples/datasets/prepare.py --dataset covertype --rows 2000 --out .data/prep-check

Raw downloads are cached in .data/raw. `--upload` writes to s3://incoming/<dataset>/ (or
s3://batch-input/<dataset>/<day>/ with --batch-input) and writes s3://datasets/<dataset>/contract.json once.
S3 settings come from .modelport/credentials.env (written by `make up`) or the environment.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modelport.naming import batch_input_uri, dataset_root, incoming_prefix

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".data" / "raw"
DATASETS = ("fashion-mnist", "eurosat", "covertype")
COVER_TYPES = (
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
)
COVTYPE_NUMERIC = (
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
)
COVTYPE_RAW_COLUMNS = (
    *COVTYPE_NUMERIC,
    *(f"Wilderness_Area_{i}" for i in range(4)),
    *(f"Soil_Type_{i}" for i in range(40)),
)
LABEL_COLUMN = {"fashion-mnist": "label", "eurosat": "label", "covertype": "cover_type"}


def contract(dataset: str) -> dict[str, Any]:
    """The dataset contract (s3://datasets/<dataset>/contract.json) that ingest checks files against."""
    if dataset == "covertype":
        columns = {name.lower(): "double" for name in COVTYPE_NUMERIC}
        columns |= {"wilderness_area": "long", "soil_type": "long", "cover_type": "string"}
        image_format = None
    else:
        columns, image_format = {"image": "binary", "label": "string"}, "png"
    return {
        "dataset": dataset,
        "label_column": LABEL_COLUMN[dataset],
        "columns": columns,
        "min_rows": 1000,
        "min_classes": 2,
        "max_class_share": 0.6,
        "image_format": image_format,
    }


def subsample(frame: pd.DataFrame, *, rows: int, offset: int) -> pd.DataFrame:
    """Rows [offset, offset + rows) of a fixed permutation (seed 0): same arguments, same rows."""
    order = np.random.default_rng(0).permutation(len(frame))[offset : offset + rows]
    return frame.iloc[order].reset_index(drop=True)


def covertype_frame(raw: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    """Covertype with its 44 one-hot columns collapsed to `wilderness_area` and `soil_type` indices."""
    frame = pd.DataFrame({name.lower(): raw[name].astype(np.float64) for name in COVTYPE_NUMERIC})
    wilderness = raw[[f"Wilderness_Area_{i}" for i in range(4)]].to_numpy()
    soil = raw[[f"Soil_Type_{i}" for i in range(40)]].to_numpy()
    frame["wilderness_area"] = wilderness.argmax(axis=1).astype(np.int64)
    frame["soil_type"] = soil.argmax(axis=1).astype(np.int64)
    frame["cover_type"] = [COVER_TYPES[int(t) - 1] for t in target]  # the dataset labels classes 1..7
    return frame


def parquet_bytes(frame: pd.DataFrame, *, corrupt: bool = False) -> bytes:
    """The frame as Parquet; `corrupt` keeps only the first half of the bytes (an unreadable file)."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    data = buffer.getvalue()
    return data[: len(data) // 2] if corrupt else data


def batch_rows(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Batch input carries the model inputs only (no label)."""
    return frame.drop(columns=LABEL_COLUMN[dataset])


def destination(dataset: str, *, batch_day: str | None, name: str) -> str:
    """Where the file goes: the ingest drop zone, or the day's batch input."""
    prefix = (
        batch_input_uri(dataset, date.fromisoformat(batch_day)) if batch_day else incoming_prefix(dataset)
    )
    return f"{prefix}{name}"


def _png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_raw(dataset: str) -> pd.DataFrame:
    """Download (cached in .data/raw) and convert the whole dataset to modelport columns."""
    RAW.mkdir(parents=True, exist_ok=True)
    if dataset == "covertype":
        from sklearn.datasets import fetch_covtype

        raw = fetch_covtype(data_home=str(RAW), as_frame=True)
        return covertype_frame(raw.data, raw.target.to_numpy())
    import torchvision

    if dataset == "fashion-mnist":
        source = torchvision.datasets.FashionMNIST(str(RAW), train=True, download=True)
    else:
        source = torchvision.datasets.EuroSAT(str(RAW), download=True)
    return pd.DataFrame(
        {"image": [_png(img) for img, _ in source], "label": [source.classes[y] for _, y in source]}
    )


def _s3_settings() -> dict[str, str]:
    values = dict(os.environ)
    creds = ROOT / ".modelport" / "credentials.env"
    if creds.is_file():
        for line in creds.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                values.setdefault(key.strip(), value.strip())
    missing = [
        k
        for k in ("MODELPORT_S3_ENDPOINT", "MODELPORT_S3_ACCESS_KEY", "MODELPORT_S3_SECRET_KEY")
        if k not in values
    ]
    if missing:
        sys.exit(f"prepare: {', '.join(missing)} not set; run `make up` and `make ui` first")
    return values


def upload(dataset: str, data: bytes, target: str) -> None:
    """Put the file, and write the dataset contract if it does not exist yet."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    env = _s3_settings()
    s3 = boto3.client(
        "s3",
        endpoint_url=env["MODELPORT_S3_ENDPOINT"],
        aws_access_key_id=env["MODELPORT_S3_ACCESS_KEY"],
        aws_secret_access_key=env["MODELPORT_S3_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )
    bucket, contract_key = dataset_root(dataset).removeprefix("s3://").split("/", 1)
    contract_key += "/contract.json"
    try:
        s3.head_object(Bucket=bucket, Key=contract_key)
    except ClientError:
        s3.put_object(Bucket=bucket, Key=contract_key, Body=json.dumps(contract(dataset), indent=2).encode())
        print(f"prepare: wrote s3://{bucket}/{contract_key}")
    target_bucket, key = target.removeprefix("s3://").split("/", 1)
    s3.put_object(Bucket=target_bucket, Key=key, Body=data)
    print(f"prepare: uploaded {len(data)} bytes to {target}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the file, then upload it or write it locally."""
    parser = argparse.ArgumentParser(description="Prepare a public dataset for modelport.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--rows", type=int, default=6000, help="rows to take (default 6000)")
    parser.add_argument("--offset", type=int, default=0, help="skip this many rows of the fixed shuffle")
    parser.add_argument("--corrupt", action="store_true", help="write a truncated, unreadable Parquet file")
    parser.add_argument("--batch-input", action="store_true", help="write batch input (no label) for --day")
    parser.add_argument("--day", help="YYYY-MM-DD for --batch-input")
    parser.add_argument("--upload", action="store_true", help="upload to S3 (needs make up and make ui)")
    parser.add_argument("--out", type=Path, help="write locally to this folder instead")
    args = parser.parse_args(argv)
    if args.batch_input and not args.day:
        parser.error("--batch-input needs --day")
    if args.upload == bool(args.out):
        parser.error("use exactly one of --upload and --out")
    frame = subsample(load_raw(args.dataset), rows=args.rows, offset=args.offset)
    if args.batch_input:
        frame = batch_rows(frame, args.dataset)
    kind = "corrupt" if args.corrupt else f"rows{args.offset}-{args.offset + len(frame)}"
    name = f"{args.dataset}-{kind}.parquet"
    data = parquet_bytes(frame, corrupt=args.corrupt)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / name).write_bytes(data)
        (args.out / "contract.json").write_text(
            json.dumps(contract(args.dataset), indent=2), encoding="utf-8"
        )
        print(f"prepare: wrote {len(frame)} rows to {args.out / name}")
    else:
        upload(
            args.dataset,
            data,
            destination(args.dataset, batch_day=args.day if args.batch_input else None, name=name),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
