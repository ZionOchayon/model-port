import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import extract_rules

ROOT = Path(__file__).resolve().parents[3]

RULE = """
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata: {name: x, namespace: monitoring}
spec:
  groups:
    - name: g
      rules:
        - {record: "a:b", expr: "sum(up)"}
"""


def test_writes_the_spec_as_a_rule_file(tmp_path):
    src, out = tmp_path / "rules.yaml", tmp_path / "out" / "gen.yaml"
    src.write_text(RULE, encoding="utf-8")
    assert extract_rules.main([str(src), str(out)]) == 0
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == {
        "groups": [{"name": "g", "rules": [{"record": "a:b", "expr": "sum(up)"}]}]
    }


def test_refuses_other_kinds(tmp_path):
    src = tmp_path / "rules.yaml"
    src.write_text(RULE.replace("PrometheusRule", "ConfigMap"), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a PrometheusRule"):
        extract_rules.main([str(src), str(tmp_path / "gen.yaml")])


def test_repo_rules_have_the_recording_group():
    text = (ROOT / "deploy/monitoring/prometheus-rules.yaml").read_text(encoding="utf-8")
    groups = {g["name"]: g for g in yaml.safe_load(text)["spec"]["groups"]}
    records = {r["record"] for r in groups["modelport.recording"]["rules"]}
    wanted = {"modelport:confidence_mean:1h", "modelport:class_share:1h", "modelport:serve_error_ratio:5m"}
    assert wanted <= records


def test_runbook_anchors():
    assert extract_rules.anchors("# Runbook\n\n## `ModelDriftWarning`\n\n## Pods stay `Pending`\n") == {
        "runbook",
        "modeldriftwarning",
        "pods-stay-pending",
    }
    groups = [
        {"name": "a", "rules": [{"record": "x", "expr": "1"}]},
        {
            "name": "b",
            "rules": [
                {"alert": "A", "expr": "1", "annotations": {"runbook_url": "https://h/docs/runbook.md#a"}},
                {"alert": "B", "expr": "1", "annotations": {"runbook_url": "https://h/docs/runbook.md#gone"}},
                {"alert": "C", "expr": "1"},
            ],
        },
    ]
    assert extract_rules.runbook_problems(groups, {"a"}) == [
        "alert B: runbook anchor #gone not in docs/runbook.md",
        "alert C: no runbook_url",
    ]


def test_repo_alerts_have_runbook_anchors():
    text = (ROOT / "deploy/monitoring/prometheus-rules.yaml").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/runbook.md").read_text(encoding="utf-8")
    groups = yaml.safe_load(text)["spec"]["groups"]
    assert extract_rules.runbook_problems(groups, extract_rules.anchors(runbook)) == []
