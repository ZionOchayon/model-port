"""Documentation coverage: every item of the coverage table has a doc entry at the given stage.

Usage: uv run python scripts/check_docs.py [--stage v0.1|v0.2|v0.3]   (default: the latest released stage)
Exit 0 when everything is documented, 1 listing each missing item. A name counts only inside code text
(an inline code span or a fenced block) of the named document, matched as a whole name.
Docstrings of public functions are enforced by ruff (`D` rules), not here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("v0.1", "v0.2", "v0.3")

# items whose stage is fixed by the coverage table; everything else is checked as it exists now
COMMANDS = {"v0.1": ["validate", "submit", "status"], "v0.2": ["promote", "rollback", "predict"]}
DAGS = {
    "v0.1": {
        "ingest_dataset": "docs/components/ingest.md",
        "retrain_on_dataset": "docs/components/ingest.md",
        "train_model": "docs/components/training.md",
        "evaluate_and_promote": "docs/components/gate.md",
    },
    "v0.2": {
        "deploy_candidate": "docs/components/canary.md",
        "serve_sync": "docs/components/canary.md",
        "batch_inference": "docs/components/batch.md",
    },
    "v0.3": {"drift_check": "docs/components/drift.md", "simulate_drift": "docs/components/drift.md"},
}
HELM_FILES = {
    "v0.1": ["airflow.values.yaml", "airflow.smoke.values.yaml", "kuberay-operator.values.yaml"],
    "v0.2": ["kube-prometheus-stack.values.yaml", "kube-prometheus-stack.smoke.values.yaml"],
    "v0.3": ["pushgateway.values.yaml", "kube-prometheus-stack.slack.values.yaml"],
}
SDK_DOC = "docs/components/sdk.md"
CLI_DOC = "docs/components/cli.md"
SPEC_DOC = "docs/components/model-yaml.md"
INFRA_DOC = "docs/components/infrastructure.md"


def code_text(markdown: str) -> str:
    """The fenced blocks and inline code spans of a Markdown document, one per line."""
    blocks = re.findall(r"^```[^\n]*\n(.*?)^```", markdown, re.MULTILINE | re.DOTALL)
    prose = re.sub(r"^```[^\n]*\n.*?^```", "", markdown, flags=re.MULTILINE | re.DOTALL)
    return "\n".join([*blocks, *re.findall(r"`([^`\n]+)`", prose)])


def mentions(code: str, name: str) -> bool:
    """`name` appears in `code` as a whole name (not part of a longer identifier, flag or dotted path)."""
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-]|\.\w)", code) is not None


def missing(kind: str, names: Iterable[str], docs: Iterable[str], texts: Mapping[str, str]) -> list[str]:
    """Each name that none of `docs` mentions in code text."""
    docs = list(docs)
    code = "\n".join(code_text(texts.get(d, "")) for d in docs)
    return [f"{kind} `{n}` not in {' or '.join(docs)}" for n in names if not mentions(code, n)]


def schema_fields(schema: Mapping[str, Any]) -> list[str]:
    """Dotted paths of every property in a JSON Schema, following local `$ref`s."""
    defs = schema.get("$defs", {})

    def resolve(node: Mapping[str, Any]) -> Mapping[str, Any]:
        ref = node.get("$ref") or next((s["$ref"] for s in node.get("allOf", []) if "$ref" in s), None)
        return defs[ref.rsplit("/", 1)[-1]] if ref else node

    def walk(node: Mapping[str, Any], prefix: str) -> list[str]:
        paths = []
        for key, child in resolve(node).get("properties", {}).items():
            paths += [f"{prefix}{key}", *walk(child, f"{prefix}{key}.")]
        return paths

    return walk(schema, "")


def undocumented_helm_keys(text: str, documented_paths: set[str]) -> list[str]:
    """Leaf keys of a values file that nothing documents.

    A key is documented by a comment on its line or right above it (not the file's leading comment block),
    or when it or an ancestor is a dotted path in `documented_paths`; an ancestor's comment covers it too.
    """
    import yaml

    lines = text.splitlines()
    first_key = next(i for i, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("#"))

    def commented(line_no: int) -> bool:
        above = line_no - 1
        return "#" in lines[line_no] or (above >= first_key and lines[above].lstrip().startswith("#"))

    problems: list[str] = []

    def walk(node: Any, path: list[str], covered: bool) -> None:  # noqa: ANN401 - yaml node
        if not isinstance(node, yaml.MappingNode):
            if not covered:
                problems.append(".".join(path))
            return
        for key, value in node.value:
            here = [*path, key.value]
            ok = covered or commented(key.start_mark.line) or ".".join(here) in documented_paths
            if isinstance(value, yaml.MappingNode) and value.flow_style:
                if not ok:  # a one-line mapping is documented as a whole
                    problems.append(".".join(here))
                continue
            walk(value, here, ok)

    walk(yaml.compose(text), [], False)
    return problems


def make_targets(makefile: str) -> list[str]:
    """Targets with a `## ` help text."""
    return re.findall(r"^([a-z-]+):.*## ", makefile, re.MULTILINE)


def cli_items(commands: Iterable[str]) -> list[str]:
    """`modelport <command>` and every option of those commands and of the global callback."""
    import typer.main

    from modelport.cli.main import app

    group = typer.main.get_command(app)
    items = [o for p in group.params for o in p.opts if o.startswith("--")]
    for name in commands:
        command = group.commands[name]  # type: ignore[attr-defined]
        items += [f"modelport {name}", *(o for p in command.params for o in p.opts if o.startswith("--"))]
    return list(dict.fromkeys(items))


def default_stage(changelog: str) -> str:
    """The stage of the latest released `## [0.N.0]` heading; v0.1 before the first release."""
    released = re.findall(r"^## \[0\.(\d+)\.0\]", changelog, re.MULTILINE)
    return f"v0.{max(int(n) for n in released)}" if released else "v0.1"


def check(stage: str) -> list[str]:
    """Every missing doc entry at `stage`."""
    upto = STAGES[: STAGES.index(stage) + 1]
    texts = {str(p.relative_to(ROOT).as_posix()): p.read_text(encoding="utf-8") for p in ROOT.glob("**/*.md")}
    texts = {k: v for k, v in texts.items() if not k.startswith((".venv/", "drafts/", "node_modules/"))}
    problems: list[str] = []

    import modelport

    problems += missing("SDK symbol", modelport.__all__, [SDK_DOC], texts)
    problems += missing("CLI", cli_items(c for s in upto for c in COMMANDS.get(s, [])), [CLI_DOC], texts)
    schema = json.loads((ROOT / "sdk/src/modelport/schema/model.schema.json").read_text(encoding="utf-8"))
    problems += missing("model.yaml field", schema_fields(schema), [SPEC_DOC], texts)

    for s in upto:
        for dag_id, doc in DAGS[s].items():
            source = ROOT / "platform" / "dags" / f"{dag_id}.py"
            if not source.is_file() or "doc_md=" not in source.read_text(encoding="utf-8"):
                problems.append(f"DAG `{dag_id}` has no doc_md")
            problems += missing("DAG", [dag_id], [doc], texts)

    infra_code = code_text(texts.get(INFRA_DOC, ""))
    documented = set(re.findall(r"[\w.-]+", infra_code))
    for s in upto:
        for name in HELM_FILES.get(s, []):
            values = (ROOT / "deploy" / "helm" / name).read_text(encoding="utf-8")
            undocumented = undocumented_helm_keys(values, documented)
            problems += [
                f"Helm key `{k}` of {name} has no comment and is not in {INFRA_DOC}" for k in undocumented
            ]
    manifests = sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "deploy" / "manifests").rglob("*.yaml")
    )
    problems += missing("manifest", manifests, [INFRA_DOC], texts)

    problems += missing(
        "make target", make_targets((ROOT / "Makefile").read_text(encoding="utf-8")), ["README.md"], texts
    )
    adrs = sorted(p.name for p in (ROOT / "docs" / "adr").glob("[0-9]*.md"))
    index = texts.get("docs/adr/README.md", "")
    problems += [f"ADR {a} not linked from docs/adr/README.md" for a in adrs if f"({a})" not in index]
    return problems


def main(argv: list[str] | None = None) -> int:
    """Print the missing entries; exit 1 if there are any."""
    parser = argparse.ArgumentParser(description="documentation coverage check")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args(argv)
    stage = args.stage or default_stage((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    problems = check(stage)
    for problem in problems:
        print(f"check_docs: {problem}")
    print(f"check_docs: stage {stage}: {len(problems)} missing")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
