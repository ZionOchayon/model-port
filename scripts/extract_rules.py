"""Write the `.spec` of a PrometheusRule manifest as a plain rule file, for `promtool check rules` and tests.

Usage: uv run python scripts/extract_rules.py RULE_MANIFEST OUT_FILE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    """Read the manifest, write `{"groups": ...}`; exit with a message when it is not a PrometheusRule."""
    parser = argparse.ArgumentParser(description="PrometheusRule -> promtool rule file")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    doc = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("kind") != "PrometheusRule":
        raise SystemExit(f"extract_rules: {args.manifest} is not a PrometheusRule")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump({"groups": doc["spec"]["groups"]}, sort_keys=False), encoding="utf-8")
    print(f"extract_rules: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
