"""Write the `.spec` of a PrometheusRule manifest as a plain rule file, for `promtool check rules` and tests.

Usage: uv run python scripts/extract_rules.py RULE_MANIFEST OUT_FILE
Fails when an alert has no `runbook_url`, or its anchor is not a heading of docs/runbook.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

RUNBOOK = Path(__file__).resolve().parents[1] / "docs" / "runbook.md"


def anchors(markdown: str) -> set[str]:
    """GitHub's heading anchors: lower case, punctuation dropped, spaces to dashes."""
    headings = re.findall(r"^#+ +(.+?) *$", markdown, re.MULTILINE)
    return {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}


def runbook_problems(groups: Iterable[Mapping[str, Any]], known: set[str]) -> list[str]:
    """Each alert without a runbook_url, or whose anchor is not in the runbook."""
    problems = []
    for group in groups:
        for rule in group["rules"]:
            if "alert" not in rule:
                continue
            url = rule.get("annotations", {}).get("runbook_url")
            if not url:
                problems.append(f"alert {rule['alert']}: no runbook_url")
            elif url.partition("#")[2] not in known:
                problems.append(
                    f"alert {rule['alert']}: runbook anchor #{url.partition('#')[2]} not in docs/runbook.md"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Read the manifest, write `{"groups": ...}`; exit with a message when it is not a PrometheusRule."""
    parser = argparse.ArgumentParser(description="PrometheusRule -> promtool rule file")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    doc = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("kind") != "PrometheusRule":
        raise SystemExit(f"extract_rules: {args.manifest} is not a PrometheusRule")
    problems = runbook_problems(doc["spec"]["groups"], anchors(RUNBOOK.read_text(encoding="utf-8")))
    if problems:
        raise SystemExit("\n".join(f"extract_rules: {p}" for p in problems))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump({"groups": doc["spec"]["groups"]}, sort_keys=False), encoding="utf-8")
    print(f"extract_rules: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
