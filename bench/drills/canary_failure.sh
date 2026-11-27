#!/usr/bin/env bash
# FR-10 drill: a version that passed the gate fails half its live requests during the canary; it must be
# rolled back and rejected while the champion keeps serving.
# Usage: bench/drills/canary_failure.sh [MODEL]   (needs make up, make ui and a served champion;
# checked by verify.py --checks V-10)
set -euo pipefail
MODEL=${1:-covertype-classifier}
SRC=examples/models/covertype-classifier
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
status() { uv run modelport --json status "$MODEL" | python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

# the spec copy is not deployed by itself (the drill starts the faulty canary), and its gate limits are wide
# so training noise cannot stop the drill before the canary: this drill tests the canary, V-07 the gate
cp -r "$SRC" "$work/model"
sed -e "s/^serving:$/serving:\n  auto_deploy: false/" \
  -e "s/^  min_improvement: .*/  min_improvement: -0.1/" \
  -e "s/^  max_drop_per_class: .*/  max_drop_per_class: 0.2/" \
  -e "s/^evaluation:$/evaluation:\n  max_latency_increase: 1.0/" \
  "$SRC/model.yaml" > "$work/model/model.drill.yaml"
grep -q "^  auto_deploy: false$" "$work/model/model.drill.yaml"
grep -q "^  max_latency_increase: 1.0$" "$work/model/model.drill.yaml"
grep -q "^name: $MODEL$" "$work/model/model.drill.yaml"
before=$(status 'd["aliases"]["champion"]')
[[ $before != None ]] || { echo "drill: $MODEL has no champion"; exit 1; }
uv run modelport submit "$work/model" --spec model.drill.yaml --wait --timeout 3600
version=$(status 'd["newest"]["version"]')
uv run python scripts/ops.py wait-dag-run evaluate_and_promote "evaluate__${MODEL}__v${version}" --timeout 1800
gate=$(status 'd["newest"]["gate"]["result"]')
[[ $gate == passed ]] || { echo "drill: v$version did not pass the gate ($gate)"; exit 1; }

run_id="drill__${MODEL}__v${version}"
uv run python scripts/ops.py trigger deploy_candidate "$run_id" --wait --timeout 1800 \
  --conf "{\"model\": \"$MODEL\", \"version\": \"$version\", \"observation_minutes\": 3, \"fault_error_rate\": 0.5}"

mkdir -p .data
printf '{"model": "%s", "version": "%s", "champion_before": "%s", "run_id": "%s"}\n' \
  "$MODEL" "$version" "$before" "$run_id" > .data/drill-canary-failure.json
cat .data/drill-canary-failure.json
uv run modelport status "$MODEL"

# the latest submission must be the example folder again (retrains use its spec)
uv run modelport submit "$SRC" > /dev/null
