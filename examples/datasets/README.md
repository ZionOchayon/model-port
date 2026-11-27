# Example datasets

`prepare.py` downloads public datasets, converts them to the Parquet layout modelport ingests, and uploads them to the platform (or writes them locally with `--out`). Raw downloads are cached in `.data/raw`.

| Dataset | Rows | modelport columns | Source and license |
| --- | --- | --- | --- |
| `fashion-mnist` | 60,000 train images, 28×28 grayscale, 10 classes | `image` (PNG bytes), `label` | Han Xiao, Kashif Rasul and Roland Vollgraf (2017), "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms". MIT License. https://github.com/zalandoresearch/fashion-mnist (downloaded with `torchvision.datasets.FashionMNIST`) |
| `eurosat` | 27,000 Sentinel-2 RGB images, 64×64, 10 land-use classes | `image` (PNG bytes), `label` | Patrick Helber, Benjamin Bischke, Andreas Dengel and Damian Borth (2019), "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification". MIT License. https://github.com/phelber/EuroSAT (downloaded with `torchvision.datasets.EuroSAT`) |
| `covertype` | 581,012 cells, 7 forest cover types | 10 numeric variables, `wilderness_area` and `soil_type` indices, `cover_type` | Jock A. Blackard (1998), UCI Machine Learning Repository. CC BY 4.0. https://archive.ics.uci.edu/dataset/31/covertype (downloaded with `sklearn.datasets.fetch_covtype`) |

## Usage

```bash
# new training data -> s3://incoming/<dataset>/ (ingest checks it and creates the next version)
uv run python examples/datasets/prepare.py --dataset fashion-mnist --rows 6000 --upload
uv run python examples/datasets/prepare.py --dataset fashion-mnist --rows 6000 --offset 6000 --upload
# a corrupt file -> ingest quarantines it
uv run python examples/datasets/prepare.py --dataset fashion-mnist --corrupt --upload
# a day of batch input (no label) -> s3://batch-input/<dataset>/<day>/
uv run python examples/datasets/prepare.py --dataset covertype --batch-input --day 2026-01-01 --rows 2000 --upload
# local files only
uv run python examples/datasets/prepare.py --dataset covertype --rows 2000 --out .data/prep-check
```

- Rows are taken from a fixed random permutation (seed 0), so the same `--rows` and `--offset` give the same rows, and different offsets never overlap.
- The first upload of a dataset also writes `s3://datasets/<dataset>/contract.json`: the columns, types and quality rules the ingest step checks every file against.
- `--upload` needs a running stack (`make up`, then `make ui` for the port-forwards). It reads the S3 endpoint and keys from `.modelport/credentials.env`.
