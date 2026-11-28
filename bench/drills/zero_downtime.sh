#!/usr/bin/env bash
# FR-11 drill: promote a new version through deploy_candidate while constant load runs; no request may fail.
# Usage: bench/drills/zero_downtime.sh [MODEL]   (needs make up, make ui and a served champion;
# checked by verify.py --checks V-11)
set -euo pipefail
MODEL=${1:-fashion-classifier}
SRC=examples/models/image-classifier
EPOCHS=6
RPS=5
DURATION=420 # a deploy with a 2-minute window took about 3.5 minutes on the CI runner
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
status() { uv run modelport --json status "$MODEL" | python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

# a new version that is not deployed by itself (the drill starts its deploy under load); its gate limits are
# wide because training noise can put a retrain below the champion: this drill tests the swap, V-07 the gate
cp -r "$SRC" "$work/model"
sed -e "s/^  epochs: .*/  epochs: $EPOCHS/" -e "s/^serving:$/serving:\n  auto_deploy: false/" \
  -e "s/^  min_improvement: .*/  min_improvement: -0.1/" \
  -e "s/^  max_drop_per_class: .*/  max_drop_per_class: 0.3/" \
  -e "s/^  max_latency_increase: .*/  max_latency_increase: 1.0/" \
  "$SRC/model.fashion.yaml" > "$work/model/model.fashion-zd.yaml"
grep -q "^  auto_deploy: false$" "$work/model/model.fashion-zd.yaml"
grep -q "^  min_improvement: -0.1$" "$work/model/model.fashion-zd.yaml"
grep -q "^name: $MODEL$" "$work/model/model.fashion-zd.yaml"
uv run modelport submit "$work/model" --spec model.fashion-zd.yaml --wait --timeout 3600
version=$(status 'd["newest"]["version"]')
uv run python scripts/ops.py wait-dag-run evaluate_and_promote "evaluate__${MODEL}__v${version}" --timeout 1800
gate=$(status 'd["newest"]["gate"]["result"]')
[[ $gate == passed ]] || { echo "drill: v$version did not pass the gate ($gate)"; exit 1; }
before=$(status 'd["aliases"]["champion"]')

load="$work/load.json"
uv run python -m modelport_platform.tools.loadgen --model "$MODEL" --rows "$SRC/example/input.parquet" \
  --rps "$RPS" --duration "$DURATION" --out "$load" > /dev/null &
loadgen=$!
sleep 10 # the load is running before the canary starts
uv run python scripts/ops.py trigger deploy_candidate "drill__${MODEL}__v${version}" \
  --conf "{\"model\": \"$MODEL\", \"version\": \"$version\", \"observation_minutes\": 2}" --wait --timeout 1200
during=false
kill -0 "$loadgen" 2> /dev/null && during=true
after=$(status 'd["aliases"]["champion"]')
wait "$loadgen" || true # V-11 judges the failures from the result file

mkdir -p .data
python3 - "$load" "$before" "$after" "$during" > .data/drill-zero-downtime.json << 'EOF'
import json, sys
result = json.load(open(sys.argv[1]))
result.update(champion_before=sys.argv[2], champion_after=sys.argv[3], promoted_during_load=sys.argv[4] == "true")
print(json.dumps(result))
EOF
cat .data/drill-zero-downtime.json

# the latest submission must be the example folder again (code.zip check, retrains)
uv run modelport submit "$SRC" --spec model.fashion.yaml > /dev/null
