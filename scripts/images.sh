#!/usr/bin/env bash
# Build the modelport images and load them into kind. Usage: scripts/images.sh [ray|all]
# Writes .modelport/images.env (IMAGE_TAG, RAY_IMAGE) for the other scripts.
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
cat > .modelport/images.env <<ENV
IMAGE_TAG=$tag
RAY_IMAGE=modelport/ray:$tag
ENV
# shellcheck source=/dev/null
. .modelport/images.env

rm -rf dist && uv build --all-packages --wheel -o dist/ -q

build_ray() {
  docker build -f deploy/images/ray/Dockerfile -t "$RAY_IMAGE" \
    --build-arg RAY_BASE_IMAGE="$RAY_BASE_IMAGE" --build-arg RAY_VERSION="$RAY_VERSION" \
    --build-arg TORCH_VERSION="$TORCH_VERSION" --build-arg XGBOOST_VERSION="$XGBOOST_VERSION" .
  kind load docker-image "$RAY_IMAGE" --name modelport
}

case $what in
  ray | all) build_ray ;;
  *) echo "usage: scripts/images.sh [ray|all]" >&2; exit 2 ;;
esac
echo "images: $what built and loaded (tag $IMAGE_TAG)"
