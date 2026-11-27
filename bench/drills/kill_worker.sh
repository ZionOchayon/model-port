#!/usr/bin/env bash
# FR-5 drill: kill a Ray Train worker after epoch 0's checkpoint is registered; training must resume, not restart.
# Usage: bench/drills/kill_worker.sh [MODEL]   (needs make up and make ui; checked by verify.py --checks V-05d)
set -euo pipefail
MODEL=${1:-fashion-classifier}
SRC=examples/models/image-classifier
EPOCHS=4
k="kubectl --context kind-modelport"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# a copy of the model folder with a longer training, so there is time to kill a worker mid-run
cp -r "$SRC" "$work/model"
sed "s/^  epochs: .*/  epochs: $EPOCHS/" "$SRC/model.fashion.yaml" > "$work/model/model.fashion-drill.yaml"
grep -q "^  epochs: $EPOCHS$" "$work/model/model.fashion-drill.yaml"
grep -q "^name: $MODEL$" "$work/model/model.fashion-drill.yaml"

out=$(uv run modelport --json submit "$work/model" --spec model.fashion-drill.yaml)
run_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["train_run_id"])' "$out")
rayjob=$(uv run python -c 'import sys; from modelport.naming import rayjob_name; print(rayjob_name(sys.argv[1], sys.argv[2]))' "$MODEL" "$run_id")
echo "drill: train_model run $run_id, RayJob $rayjob"

uv run python scripts/ops.py wait-checkpoint "$MODEL" "$rayjob" --timeout 1800
pod=$($k get pods -n ray -l "modelport.dev/model=$MODEL,modelport.dev/kind=train,ray.io/node-type=worker" \
  -o jsonpath='{.items[0].metadata.name}')
test -n "$pod"
killed_at_ms=$(python3 -c 'import time; print(int(time.time() * 1000))')
# SIGKILL the Train worker actor process (psutil ships with Ray; the image may lack pkill)
$k exec -n ray "$pod" -- python -c '
import os, psutil
# the command line of this script contains the marker too: skip it
workers = [p for p in psutil.process_iter(["cmdline"]) if p.pid != os.getpid() and any("RayTrainWorker" in c for c in (p.info["cmdline"] or []))]
killed = [p.kill() or p.pid for p in workers]
assert killed, "no RayTrainWorker process found"
print("drill: killed", killed)
'
uv run python scripts/ops.py wait-dag-run train_model "$run_id" --state success --timeout 3600
# train_model then evaluates the drill version on the shared cluster; wait so later checks see a settled stack
version=$(uv run modelport --json status "$MODEL" | python3 -c 'import json,sys; print(json.load(sys.stdin)["newest"]["version"])')
uv run python scripts/ops.py wait-dag-run evaluate_and_promote "evaluate__${MODEL}__v${version}" --state success --timeout 1800

mkdir -p .data
printf '{"run_id": "%s", "rayjob": "%s", "killed_at_ms": %s, "epochs": %s, "pod": "%s"}\n' \
  "$run_id" "$rayjob" "$killed_at_ms" "$EPOCHS" "$pod" > .data/drill-kill-worker.json
cat .data/drill-kill-worker.json

# the latest submission must be the example folder again (code.zip check, retrains)
uv run modelport submit "$SRC" --spec model.fashion.yaml > /dev/null
