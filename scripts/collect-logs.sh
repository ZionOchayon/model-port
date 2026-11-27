#!/usr/bin/env bash
# Collect cluster diagnostics into ./artifacts (CI, after a failed job). Every step is best effort.
set -u
mkdir -p artifacts
k="kubectl --context kind-modelport"
$k get all -A -o wide > artifacts/get-all.txt 2>&1
$k get rayjobs,rayclusters,rayservices -A -o yaml > artifacts/ray-crs.yaml 2>&1
$k describe pods -A > artifacts/describe-pods.txt 2>&1
$k get events -A --sort-by=.lastTimestamp > artifacts/events.txt 2>&1
$k logs -n ray deploy/kuberay-operator > artifacts/kuberay-operator.log 2>&1
for d in mlflow minio; do $k logs -n platform "deploy/$d" > "artifacts/$d.log" 2>&1; done
for c in scheduler triggerer dag-processor api-server; do
  $k logs -n platform "deploy/airflow-$c" -c "$c" > "artifacts/airflow-$c.log" 2>&1
done
sched=$($k get pod -n platform -l component=scheduler -o name | head -1)
run() { $k exec -n platform "${sched#pod/}" -c scheduler -- "$@"; }
for dag in ingest_dataset retrain_on_dataset train_model evaluate_and_promote; do
  echo "== $dag"; run airflow dags list-runs "$dag" 2>&1
done > artifacts/airflow-runs.txt
# task logs are files whose names contain ":", which the artifact upload rejects
run tar cf - -C /opt/airflow logs > artifacts/airflow-task-logs.tar 2> artifacts/airflow-task-logs.err  # no gzip in the image
for p in $($k get pods -n ray -l ray.io/node-type=head -o name); do
  $k logs -n ray "$p" --all-containers > "artifacts/$(basename "$p").log" 2>&1
done
cp -r .data artifacts/data 2>/dev/null
cp /tmp/modelport-ui-*.log artifacts/ 2>/dev/null
exit 0
