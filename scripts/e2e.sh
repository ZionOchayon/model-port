#!/usr/bin/env bash
# CI end-to-end flow on the smoke profile, cumulative by stage. Usage: scripts/e2e.sh v0.1|v0.2|v0.3
# Needs a running stack (make up PROFILE=smoke). Each step's result is checked by scripts/verify.py at the end.
set -euo pipefail
STAGE=${1:?usage: scripts/e2e.sh v0.1|v0.2|v0.3}
P=smoke
IMG=examples/models/image-classifier
group() { echo "::group::$*"; "$@"; echo "::endgroup::"; }
mp() { uv run modelport "$@"; }
ops() { uv run python scripts/ops.py "$@"; }

group make ui
set -a; . .modelport/credentials.env; set +a
mkdir -p .data

# ---- v0.1: ingest (FR-8), train (FR-4, FR-5, FR-6), gate (FR-7) ----
group uv run python examples/datasets/prepare.py --dataset fashion-mnist --rows 6000 --upload
group uv run python examples/datasets/prepare.py --dataset fashion-mnist --corrupt --upload
group ops wait-dataset fashion-mnist 1 --timeout 900
group ops wait-quarantine fashion-mnist --timeout 900
group mp submit "$IMG" --spec model.fashion.yaml --wait --timeout 3600
if mp submit "$IMG" --spec model.fashion-broken.yaml --wait --timeout 3600; then
  echo "the broken version was not rejected"; exit 1
fi
# retrains use the latest submission's spec file: point it back at the good spec (same run, reused)
group mp submit "$IMG" --spec model.fashion.yaml
group uv run python examples/datasets/prepare.py --dataset fashion-mnist --rows 6000 --offset 6000 --upload
group ops wait-dag-run train_model retrain__fashion-classifier__fashion-mnist__v2 --state success --timeout 3600
# the retrain is version 3 (1 good, 2 broken); its evaluation runs on the shared cluster after train_model ends
group ops wait-dag-run evaluate_and_promote evaluate__fashion-classifier__v3 --state success --timeout 1800
group bash bench/drills/kill_worker.sh fashion-classifier
checks=V-01,V-02,V-04,V-05,V-05d,V-06,V-07a,V-07b,V-08

# ---- v0.2: serving (FR-9, FR-10, FR-11) ----
if [[ $STAGE != v0.1 ]]; then
  TAB=examples/models/covertype-classifier
  # the v3 retrain passes the gate and its canary, so fashion-classifier has a champion and a previous
  group ops wait-alias fashion-classifier previous --timeout 2400
  group uv run python examples/datasets/prepare.py --dataset covertype --rows 20000 --upload
  group ops wait-dataset covertype 1 --timeout 900
  group mp submit "$TAB" --wait --timeout 3600
  group ops wait-alias covertype-classifier champion --timeout 2400
  group make ui # the Serve Service exists only once a model is served
  group mp predict fashion-classifier "$IMG/example/input.parquet"
  group mp predict covertype-classifier "$TAB/example/input.parquet"
  checks=$checks,V-09
fi
if [[ $STAGE == v0.3 ]]; then
  echo "e2e: stage $STAGE is not implemented yet" >&2
  exit 1
fi

group uv run python scripts/verify.py --profile "$P" --model fashion-classifier --checks "$checks"
