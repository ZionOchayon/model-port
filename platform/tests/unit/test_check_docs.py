import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import check_docs


def test_code_text_keeps_spans_and_blocks_only():
    md = "Call report in prose.\n\nUse `modelport.report()`.\n\n```bash\nmodelport submit DIR --wait\n```\n"
    code = check_docs.code_text(md)
    assert check_docs.mentions(code, "modelport.report")
    assert check_docs.mentions(code, "--wait")
    assert not check_docs.mentions(check_docs.code_text("Call report in prose."), "report")


def test_mentions_respects_name_boundaries():
    code = "`--timeout-s` `train.epochs` `BaseModel`"
    assert not check_docs.mentions(code, "--timeout")
    assert check_docs.mentions(code, "train.epochs")
    assert not check_docs.mentions(code, "train.epoch")
    assert not check_docs.mentions(code, "Base")


def test_schema_fields_walks_nested_refs():
    schema = {
        "properties": {
            "name": {"type": "string"},
            "train": {"$ref": "#/$defs/Train"},
        },
        "$defs": {"Train": {"properties": {"epochs": {"type": "integer"}}}},
    }
    assert check_docs.schema_fields(schema) == ["name", "train", "train.epochs"]


def test_helm_keys_documented_by_comment_or_ancestor():
    text = (
        "# header comment does not document the first key\n"
        "image:\n"
        "  tag: v1 # pinned\n"
        "  pullPolicy: IfNotPresent\n"
        "metrics:\n"
        "  # needs the CRDs\n"
        "  enabled: true\n"
        "scheduler:\n"
        "  resources:\n"
        "    requests: { cpu: 1, memory: 1Gi }\n"
        "executor: LocalExecutor\n"
        "replicas: 1\n"
    )
    missing = check_docs.undocumented_helm_keys(text, documented_paths={"scheduler.resources", "executor"})
    assert missing == ["image.pullPolicy", "replicas"]


def test_missing_items_names_the_doc():
    docs = {"docs/a.md": "`one`"}
    assert check_docs.missing("CLI", ["one", "two"], ["docs/a.md"], docs) == ["CLI `two` not in docs/a.md"]
