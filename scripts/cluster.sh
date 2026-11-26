#!/usr/bin/env bash
# Create the kind cluster `modelport` if it does not exist, then point kubectl at it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
. ./versions.env

if ! kind get clusters | grep -qx modelport; then
  kind create cluster --name modelport --image "$KIND_NODE_IMAGE" --config deploy/kind/kind-config.yaml --wait 180s
fi
bash scripts/kubeconfig.sh
