#!/usr/bin/env bash
# `make ui`: port-forward the modelport UIs and print their URLs and logins.
# Inside the dev toolbox the forwards bind 0.0.0.0 (published ports, ADR-0019) and, in an interactive
# terminal, stay in the foreground until Ctrl+C; elsewhere they run in the background (PIDs in .modelport/ui.pids).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
k="kubectl --context kind-modelport"
pids=.modelport/ui.pids
mkdir -p .modelport
if [[ -f $pids ]]; then xargs -r kill < "$pids" 2> /dev/null || true; fi
: > "$pids"
address=127.0.0.1
[[ ${MODELPORT_TOOLBOX:-} == 1 ]] && address=0.0.0.0

# name namespace service local:remote
forwards=(
  "Airflow platform airflow-api-server 8080:8080"
  "MLflow platform mlflow 5000:5000"
  "MinIO-S3 platform minio 9000:9000"
  "Ray-dashboard ray modelport-shared-head-svc 8265:8265"
  "Serve ray modelport-serve-serve-svc 8000:8000"
  "Prometheus monitoring kps-prometheus 9090:9090"
  "Grafana monitoring grafana 3000:80"
  "Alertmanager monitoring kps-alertmanager 9093:9093"
)
for forward in "${forwards[@]}"; do
  read -r name ns svc ports <<< "$forward"
  if $k get svc -n "$ns" "$svc" > /dev/null 2>&1; then
    $k port-forward --address "$address" -n "$ns" "svc/$svc" "$ports" > "/tmp/modelport-ui-$name.log" 2>&1 &
    echo $! >> "$pids"
    echo "ui: ${name//-/ } -> http://localhost:${ports%%:*}"
  fi
done
sleep 3
if [[ -f .modelport/credentials.env ]]; then
  # shellcheck source=/dev/null
  . .modelport/credentials.env
  echo "ui: Airflow login modelport / ${MODELPORT_AIRFLOW_PASSWORD}"
  [[ -n ${MODELPORT_GRAFANA_PASSWORD:-} ]] && $k get svc -n monitoring grafana > /dev/null 2>&1 \n    && echo "ui: Grafana login admin / ${MODELPORT_GRAFANA_PASSWORD}"
fi
if [[ ${MODELPORT_TOOLBOX:-} == 1 && -t 0 ]]; then
  echo "ui: forwarding until Ctrl+C"
  wait
fi
