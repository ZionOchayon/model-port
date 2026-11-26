#!/usr/bin/env bash
# Delete the kind cluster and local state. Local Docker images stay, so the next `make up` is fast (NFR-3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
kind delete cluster --name modelport
rm -rf .modelport/
echo "down: cluster modelport and .modelport/ removed"
