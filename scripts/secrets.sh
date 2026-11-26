#!/usr/bin/env bash
# Create every modelport Secret once, with random values (NFR-4). An existing Secret is never changed,
# so running this twice is a no-op. Nothing secret is stored in the repo, images or Helm values.
set -euo pipefail
k="kubectl --context kind-modelport"

exists() { $k get secret -n "$1" "$2" > /dev/null 2>&1; }
create() { # namespace name --from-literal=... (only when absent)
  local ns=$1 name=$2; shift 2
  if exists "$ns" "$name"; then echo "secrets: $ns/$name exists"; return; fi
  $k create secret generic "$name" -n "$ns" "$@" > /dev/null
  echo "secrets: $ns/$name created"
}
read_key() { $k get secret -n "$1" "$2" -o "jsonpath={.data.$3}" | base64 -d; }
hex() { openssl rand -hex "$1"; }

create platform postgres-credentials \
  --from-literal=postgres-password="$(hex 24)" --from-literal=airflow-password="$(hex 24)" \
  --from-literal=mlflow-password="$(hex 24)"
create platform minio-credentials --from-literal=root-user=modelport-admin --from-literal=root-password="$(hex 24)"

# Copies and derived values are read back from the source Secrets, so they always agree.
s3_user=$(read_key platform minio-credentials root-user)
s3_password=$(read_key platform minio-credentials root-password)
af_db_password=$(read_key platform postgres-credentials airflow-password)
for target in "ray s3-credentials" "platform modelport-s3"; do
  read -r ns name <<< "$target"
  create "$ns" "$name" --from-literal=AWS_ACCESS_KEY_ID="$s3_user" --from-literal=AWS_SECRET_ACCESS_KEY="$s3_password"
done
create platform airflow-metadata \
  --from-literal=connection="postgresql://airflow:${af_db_password}@postgres.platform.svc.cluster.local:5432/airflow"
create platform airflow-fernet-key --from-literal=fernet-key="$(openssl rand -base64 32 | tr '+/' '-_')"
create platform airflow-api-secret-key --from-literal=api-secret-key="$(hex 32)"
create platform airflow-jwt-secret --from-literal=jwt-secret="$(hex 32)"
create platform airflow-auth --from-literal=passwords.json="{\"modelport\": \"$(hex 16)\"}"
create platform airflow-connections --from-literal=AIRFLOW_CONN_MINIO_S3="{\"conn_type\": \"aws\", \"login\": \"${s3_user}\", \"password\": \"${s3_password}\", \"extra\": {\"endpoint_url\": \"http://minio.platform.svc.cluster.local:9000\", \"region_name\": \"us-east-1\"}}"
create monitoring grafana-admin --from-literal=admin-user=admin --from-literal=admin-password="$(hex 16)"
if [[ -n ${SLACK_WEBHOOK_URL:-} ]]; then
  create monitoring alertmanager-slack --from-literal=url="$SLACK_WEBHOOK_URL"
fi
