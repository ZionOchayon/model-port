#!/usr/bin/env bash
# Build the modelport images and load them into kind. Usage: scripts/images.sh [ray|airflow|minio|all]
# Writes .modelport/images.env (IMAGE_TAG, RAY_IMAGE, AIRFLOW_IMAGE, MINIO_IMAGE) for the other scripts. A partial
# build updates only the images it built; IMAGE_TAG is always the Airflow image's tag (the chart uses it).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
. ./versions.env
what=${1:-all}

tag=$(git rev-parse --short=12 HEAD)
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
  tag="$tag-dirty-$(git diff HEAD | sha256sum | cut -c1-8)"
fi
mkdir -p .modelport
RAY_IMAGE="" AIRFLOW_IMAGE=""
# shellcheck source=/dev/null
[[ -f .modelport/images.env ]] && . .modelport/images.env

build() { # dockerfile image build-args...
  local dockerfile=$1 image=$2; shift 2
  docker build -f "$dockerfile" -t "$image" "$@" .
  kind load docker-image "$image" --name modelport
}

rm -rf dist && uv build --all-packages --wheel -o dist/ -q
case $what in
  ray | airflow | minio | all) ;;
  *) echo "usage: scripts/images.sh [ray|airflow|minio|all]" >&2; exit 2 ;;
esac
if [[ $what == ray || $what == all ]]; then
  RAY_IMAGE=modelport/ray:$tag
  build deploy/images/ray/Dockerfile "$RAY_IMAGE" \
    --build-arg RAY_BASE_IMAGE="$RAY_BASE_IMAGE" --build-arg RAY_VERSION="$RAY_VERSION" \
    --build-arg TORCH_VERSION="$TORCH_VERSION" --build-arg XGBOOST_VERSION="$XGBOOST_VERSION"
fi
if [[ $what == airflow || $what == all ]]; then
  AIRFLOW_IMAGE=modelport/airflow:$tag
  build deploy/images/airflow/Dockerfile "$AIRFLOW_IMAGE" \
    --build-arg AIRFLOW_BASE_IMAGE="$AIRFLOW_BASE_IMAGE" --build-arg AIRFLOW_VERSION="$AIRFLOW_VERSION"
fi
# MinIO is built from source at a fixed release (ADR-0003), so it is tagged by release and built only once.
MINIO_IMAGE=modelport/minio:$MINIO_RELEASE
if [[ $what == minio || $what == all ]]; then
  if ! docker image inspect "$MINIO_IMAGE" > /dev/null 2>&1; then
    docker build -f deploy/images/minio/Dockerfile -t "$MINIO_IMAGE" --build-arg MINIO_RELEASE="$MINIO_RELEASE" \
      --build-arg GO_BUILDER_IMAGE="$GO_BUILDER_IMAGE" --build-arg ALPINE_IMAGE="$ALPINE_IMAGE" .
  fi
  kind load docker-image "$MINIO_IMAGE" --name modelport
fi

cat > .modelport/images.env <<ENV
IMAGE_TAG=${AIRFLOW_IMAGE##*:}
RAY_IMAGE=$RAY_IMAGE
AIRFLOW_IMAGE=$AIRFLOW_IMAGE
MINIO_IMAGE=$MINIO_IMAGE
ENV
echo "images: $what built and loaded (tag $tag)"
