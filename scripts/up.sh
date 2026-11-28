#!/usr/bin/env bash
# `make up`: create the kind cluster and install everything, then wait until it is ready.
# PROFILE=full|smoke. Exit 0 ready, 1 preflight or install failure, 124 not ready within UP_TIMEOUT.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
. ./versions.env
PROFILE=${PROFILE:-full}
UP_TIMEOUT=${UP_TIMEOUT:-1500}
k="kubectl --context kind-modelport"
start=$(date +%s)
step() { echo "up: [$(( $(date +%s) - start ))s] $*"; }
diagnose() { # on any failure: what is not running, and why
  echo "up: failed; pods and recent warnings:" >&2
  $k get pods -A -o wide >&2 2>/dev/null || true
  $k get events -A --field-selector type!=Normal --sort-by=.lastTimestamp 2>/dev/null | tail -20 >&2 || true
}
trap 'code=$?; (( code != 0 )) && diagnose; exit $code' EXIT
case $PROFILE in smoke | full) ;; *) echo "up: PROFILE must be smoke or full, got '$PROFILE'" >&2; exit 1 ;; esac

# 1 Preflight
docker info > /dev/null 2>&1 || { echo "preflight: Docker engine not reachable" >&2; exit 1; }
min_gib=${UP_MIN_MEM_GIB:-$([[ $PROFILE == smoke ]] && echo 12 || echo 14)}
mem_gib=$(( $(docker info --format '{{.MemTotal}}') / 1073741824 ))
if (( mem_gib < min_gib )); then
  echo "preflight: engine memory ${mem_gib} GiB < required ${min_gib} GiB (profile $PROFILE)" >&2
  exit 1
fi
cpus=$(docker info --format '{{.NCPU}}')
(( cpus >= $([[ $PROFILE == smoke ]] && echo 4 || echo 8) )) || echo "preflight: warning: only $cpus CPUs" >&2
[[ $(docker info --format '{{.CgroupVersion}}') == 2 ]] || { echo "preflight: cgroup v2 required (Kubernetes 1.37)" >&2; exit 1; }

step "2 cluster"; bash scripts/cluster.sh
step "3 images"; bash scripts/images.sh all
# shellcheck source=/dev/null
. .modelport/images.env
# kind cannot load an image referenced by digest (it is stored as `import-<date>@sha256`, which the kubelet
# cannot resolve), so each digest-pinned image is re-tagged locally with its digest prefix and loaded by tag.
mirror() { # pinned image ref -> prints the local tag that pods use
  local ref=$1 name=${1%%@*} digest=${1##*@sha256:}
  local local_ref="modelport/mirror-${name##*/}-${digest:0:12}"
  local_ref=${local_ref/:/-}:local
  docker image inspect "$local_ref" > /dev/null 2>&1 || { docker pull -q "$ref" > /dev/null && docker tag "$ref" "$local_ref"; }
  kind load docker-image "$local_ref" --name modelport > /dev/null
  echo "$local_ref"
}
POSTGRES_POD_IMAGE=$(mirror "$POSTGRES_IMAGE")
MLFLOW_POD_IMAGE=$(mirror "$MLFLOW_IMAGE")

step "4 namespaces"; $k apply -f deploy/manifests/base/namespaces.yaml > /dev/null
step "5 secrets"; bash scripts/secrets.sh

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update > /dev/null
helm repo add apache-airflow https://airflow.apache.org --force-update > /dev/null
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update > /dev/null

step "6 monitoring"
kps_values=(-f deploy/helm/kube-prometheus-stack.values.yaml)
[[ $PROFILE == smoke ]] && kps_values+=(-f deploy/helm/kube-prometheus-stack.smoke.values.yaml)
[[ -n ${SLACK_WEBHOOK_URL:-} ]] && kps_values+=(-f deploy/helm/kube-prometheus-stack.slack.values.yaml) # secrets.sh made alertmanager-slack
helm --kube-context kind-modelport upgrade --install kps prometheus-community/kube-prometheus-stack \
  --version "$KPS_CHART_VERSION" -n monitoring "${kps_values[@]}" --wait --timeout 10m > /dev/null
helm --kube-context kind-modelport upgrade --install pushgateway prometheus-community/prometheus-pushgateway \
  --version "$PUSHGATEWAY_CHART_VERSION" -n monitoring -f deploy/helm/pushgateway.values.yaml --wait > /dev/null

step "7 KubeRay"
helm --kube-context kind-modelport upgrade --install kuberay-operator kuberay/kuberay-operator --version "$KUBERAY_VERSION" \
  -n ray -f deploy/helm/kuberay-operator.values.yaml --wait --timeout 5m > /dev/null

step "8 platform env"
env_file=.modelport/platform.env
cat > "$env_file" <<ENV
MODELPORT_PROFILE=$PROFILE
MODELPORT_S3_ENDPOINT=http://minio.platform.svc.cluster.local:9000
AWS_ENDPOINT_URL=http://minio.platform.svc.cluster.local:9000
AWS_REGION=us-east-1
MLFLOW_TRACKING_URI=http://mlflow.platform.svc.cluster.local:5000
MODELPORT_MLFLOW_UI_URL=http://localhost:5000
MODELPORT_RAY_NAMESPACE=ray
MODELPORT_RAY_IMAGE=$RAY_IMAGE
MODELPORT_SHARED_RAY_ADDRESS=http://modelport-shared-head-svc.ray.svc.cluster.local:8265
MODELPORT_RAYSERVICE_NAME=modelport-serve
MODELPORT_SERVE_URL=http://modelport-serve-serve-svc.ray.svc.cluster.local:8000
MODELPORT_CANARY_SYNTHETIC_TRAFFIC=true
MODELPORT_PROMETHEUS_URL=http://kps-prometheus.monitoring.svc.cluster.local:9090
MODELPORT_PUSHGATEWAY_URL=http://pushgateway.monitoring.svc.cluster.local:9091
MODELPORT_DATASETS=fashion-mnist,eurosat,covertype
ENV
for ns in platform ray; do
  $k create configmap modelport-platform-env -n "$ns" --from-env-file="$env_file" --dry-run=client -o yaml | $k apply -f - > /dev/null
done

step "9 manifests"
$k delete job minio-buckets -n platform --ignore-not-found > /dev/null
kubectl kustomize "deploy/manifests/overlays/$PROFILE" \
  | sed "s|__RAY_IMAGE__|$RAY_IMAGE|g; s|__MINIO_IMAGE__|$MINIO_IMAGE|g; s|__POSTGRES_IMAGE__|$POSTGRES_POD_IMAGE|g; s|__MLFLOW_IMAGE__|$MLFLOW_POD_IMAGE|g" \
  | $k apply -f - > /dev/null

step "10 wait storage"
$k rollout status -n platform statefulset/postgres --timeout=300s > /dev/null
for d in minio mlflow; do $k rollout status -n platform "deploy/$d" --timeout=300s > /dev/null; done
$k wait -n platform --for=condition=complete job/minio-buckets --timeout=300s > /dev/null

step "11 Airflow"
# The migration Job is a plain resource (useHelmHooks: false) and Job templates are immutable.
$k delete job airflow-run-airflow-migrations -n platform --ignore-not-found > /dev/null
smoke_values=(); [[ $PROFILE == smoke ]] && smoke_values=(-f deploy/helm/airflow.smoke.values.yaml)
helm --kube-context kind-modelport upgrade --install airflow apache-airflow/airflow --version "$AIRFLOW_CHART_VERSION" \
  -n platform -f deploy/helm/airflow.values.yaml "${smoke_values[@]}" --set images.airflow.tag="$IMAGE_TAG" \
  --wait --timeout 15m > /dev/null

step "12 pools"
slots=$([[ $PROFILE == smoke ]] && echo 1 || echo 2)
scheduler=$($k get pod -n platform -l component=scheduler -o jsonpath='{.items[0].metadata.name}')
for pool in "ray_training $slots training RayJobs" "shared_cluster $slots evaluation and batch jobs on the shared cluster" \
            "serve_config 1 every change to the RayService"; do
  read -r name n description <<< "$pool"
  $k exec -n platform "$scheduler" -c scheduler -- airflow pools set "$name" "$n" "$description" > /dev/null
done

step "13 monitoring objects"
$k apply -k deploy/monitoring > /dev/null

step "14 wait all"
remaining=$(( UP_TIMEOUT - ($(date +%s) - start) ))
(( remaining > 0 )) || { echo "up: not ready within ${UP_TIMEOUT}s" >&2; exit 124; }
uv run python scripts/wait_ready.py --profile "$PROFILE" --timeout "$remaining"

step "15 credentials"
secret() { $k get secret -n "$1" "$2" -o "jsonpath={.data.$3}" | base64 -d; }
umask 077
cat > .modelport/credentials.env <<ENV
MODELPORT_AIRFLOW_USERNAME=modelport
MODELPORT_AIRFLOW_PASSWORD=$(secret platform airflow-auth 'passwords\.json' | python3 -c 'import json,sys; print(json.load(sys.stdin)["modelport"])')
MODELPORT_GRAFANA_PASSWORD=$(secret monitoring grafana-admin admin-password)
# Platform-engineer tooling only (examples/datasets/prepare.py, scripts/*); the modelport CLI never reads these.
MODELPORT_S3_ENDPOINT=http://localhost:9000
MODELPORT_S3_ACCESS_KEY=$(secret platform minio-credentials root-user)
MODELPORT_S3_SECRET_KEY=$(secret platform minio-credentials root-password)
ENV

echo "up: ready in $(( $(date +%s) - start )) s (profile $PROFILE)"
echo "Next: make ui"
