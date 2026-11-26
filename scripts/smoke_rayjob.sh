#!/usr/bin/env bash
# Prove KubeRay and the modelport Ray image work: install the operator (idempotent), run a one-line
# RayJob, wait for SUCCEEDED, and print how long it took. Needs the cluster and `scripts/images.sh ray`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
. ./versions.env
# shellcheck source=/dev/null
. .modelport/images.env
k="kubectl --context kind-modelport"

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update > /dev/null
helm --kube-context kind-modelport upgrade --install kuberay-operator kuberay/kuberay-operator \
  --version "$KUBERAY_VERSION" -n ray -f deploy/helm/kuberay-operator.values.yaml --wait --timeout 5m

name="smoke-$(date +%s)"
start=$(date +%s)
$k apply -f - <<YAML
apiVersion: ray.io/v1
kind: RayJob
metadata: {name: $name, namespace: ray, labels: {app.kubernetes.io/part-of: modelport, modelport.dev/kind: smoke}}
spec:
  entrypoint: python -c "import ray; ray.init(); print(ray.cluster_resources())"
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 30
  activeDeadlineSeconds: 900
  submitterPodTemplate:
    spec:
      restartPolicy: Never
      containers:
        - name: ray-job-submitter
          image: $RAY_IMAGE
          imagePullPolicy: IfNotPresent
          resources: {requests: {cpu: 100m, memory: 200Mi}, limits: {cpu: "1", memory: 1Gi}}
  rayClusterSpec:
    rayVersion: "$RAY_VERSION"
    headGroupSpec:
      rayStartParams: {num-cpus: "1", dashboard-host: 0.0.0.0}
      template:
        spec:
          containers:
            - name: ray-head
              image: $RAY_IMAGE
              imagePullPolicy: IfNotPresent
              resources: {requests: {cpu: 200m, memory: 1536Mi}, limits: {cpu: "1", memory: 3Gi}}
YAML

for _ in $(seq 1 180); do
  status=$($k get rayjob -n ray "$name" -o jsonpath='{.status.jobDeploymentStatus}/{.status.jobStatus}')
  case $status in
    Complete/SUCCEEDED) echo "smoke_rayjob: $name SUCCEEDED in $(( $(date +%s) - start )) s"; exit 0 ;;
    Failed/* | ValidationFailed/* | */FAILED | */STOPPED)
      echo "smoke_rayjob: $name ended $status" >&2
      $k describe rayjob -n ray "$name" >&2
      exit 1 ;;
  esac
  sleep 5
done
echo "smoke_rayjob: $name not done after 15 min (last: $status)" >&2
exit 1
