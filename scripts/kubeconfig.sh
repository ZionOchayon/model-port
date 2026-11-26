#!/usr/bin/env bash
# Point kubectl at the kind cluster. Inside the dev toolbox (ADR-0019) the API server is reached by its
# container name on the Docker network `kind`; on a host, kind's own kubeconfig entry is used.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ${MODELPORT_TOOLBOX:-} == 1 ]]; then
  mkdir -p .modelport
  kind get kubeconfig --internal --name modelport > .modelport/kubeconfig
  chmod 600 .modelport/kubeconfig
else
  kubectl config use-context kind-modelport > /dev/null
fi
kubectl --context kind-modelport cluster-info > /dev/null
echo "kubeconfig: context kind-modelport is reachable"
