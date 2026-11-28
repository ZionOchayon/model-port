"""The FR-13 table from the newest `docs/results/raw/bench-*.csv`: `docs/results/table.md`, README block.

Usage: `python bench/summarize.py` (write) or `python bench/summarize.py --check README.md` (exit 1 when
either file differs from what the newest CSV gives; CI and V-13 run this).
"""

import argparse
import csv
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- bench:start -->", "<!-- bench:end -->"
PATTERNS = {
    "rayjob": "RayJob (cluster per job)",
    "shared": "Shared cluster, workers from zero",
    "shared-warm": "Shared cluster, worker already up",
}
METRICS = ("startup_s", "duration_s", "cpu_s")
PLACEHOLDER = "No measurements yet: they come from the `bench` workflow (`make bench`), n = 5 per pattern."


def load(path: Path) -> list[dict[str, str]]:
    """The CSV's rows."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def counts(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    """Rows per pattern."""
    out: dict[str, int] = {}
    for row in rows:
        out[row["pattern"]] = out.get(row["pattern"], 0) + 1
    return out


def _cell(values: list[str]) -> str:
    numbers = [float(v) for v in values if v != ""]
    if not numbers:
        return "—"
    return f"{statistics.median(numbers):.1f} ({min(numbers):.1f}-{max(numbers):.1f})"


def table(rows: Sequence[dict[str, str]], csv_link: str) -> str:
    """Median (min-max) per pattern and metric, with n, the host and a link to the raw CSV."""
    lines = [
        "| Pattern | n | Startup s | Duration s | CPU-seconds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pattern, label in PATTERNS.items():
        mine = [r for r in rows if r["pattern"] == pattern]
        if mine:
            cells = [_cell([r[m] for r in mine]) for m in METRICS]
            lines.append(f"| {label} | {len(mine)} | {' | '.join(cells)} |")
    hosts = ", ".join(sorted({r["host"] for r in rows}))
    name = csv_link.rsplit("/", 1)[-1]
    lines += [
        "",
        f"Median (min-max) of n runs per pattern. Host: {hosts}. Raw data: [{name}]({csv_link}).",
    ]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    """`text` with everything between the bench markers replaced by `block`."""
    if START not in text or END not in text:
        raise ValueError(f"README has no {START} ... {END} block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block}\n{END}{tail}"


def expected(root: Path, readme_text: str) -> tuple[str, str]:
    """(table.md, README) as the newest CSV under `root` gives them."""
    raw = sorted((root / "docs/results/raw").glob("bench-*.csv"))
    if raw:
        rows = load(raw[-1])
        page = table(rows, f"raw/{raw[-1].name}")
        block = table(rows, f"docs/results/raw/{raw[-1].name}")
    else:
        page = block = PLACEHOLDER
    return f"# Bench results (FR-13)\n\nMethod: `bench/README.md`.\n\n{page}\n", replace_block(
        readme_text, block
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write (or with --check, compare) the table and the README block."""
    parser = argparse.ArgumentParser(description="FR-13 table and README block from the newest bench CSV")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", type=Path, metavar="README", help="compare instead of writing")
    args = parser.parse_args(argv)
    readme = args.check or args.root / "README.md"
    page_path = args.root / "docs/results/table.md"
    page, text = expected(args.root, readme.read_text(encoding="utf-8"))
    if args.check:
        stale = [
            str(p)
            for p, want in ((page_path, page), (readme, text))
            if not p.is_file() or p.read_text(encoding="utf-8") != want
        ]
        for path in stale:
            print(f"summarize: {path} is not what the newest bench CSV gives; run python bench/summarize.py")
        return 1 if stale else 0
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(page, encoding="utf-8", newline="\n")
    readme.write_text(text, encoding="utf-8", newline="\n")
    print(f"summarize: wrote {page_path} and the README block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
