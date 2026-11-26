#!/usr/bin/env bash
# Install the pinned kind, kubectl and helm (linux-amd64) with sha256 checks.
# Used by CI and the dev toolbox image. Usage: DEST=/usr/local/bin bash scripts/install-tools.sh
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
versions="${VERSIONS_FILE:-$here/../versions.env}"
[[ -f $versions ]] || versions="$here/versions.env"   # the toolbox build copies both files side by side
# shellcheck source=/dev/null
. "$versions"
dest="${DEST:-/usr/local/bin}"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fetch() { # url, file, sha256
  curl -fsSLo "$work/$2" "$1"
  echo "$3  $work/$2" | sha256sum -c --quiet - || { echo "install-tools: sha256 mismatch for $2" >&2; exit 1; }
}

fetch "https://github.com/kubernetes-sigs/kind/releases/download/${KIND_VERSION}/kind-linux-amd64" kind "$KIND_SHA256_LINUX_AMD64"
fetch "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" kubectl "$KUBECTL_SHA256_LINUX_AMD64"
fetch "https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz" helm.tgz "$HELM_SHA256_LINUX_AMD64"
tar -xzf "$work/helm.tgz" -C "$work" linux-amd64/helm

install -m 0755 "$work/kind" "$dest/kind"
install -m 0755 "$work/kubectl" "$dest/kubectl"
install -m 0755 "$work/linux-amd64/helm" "$dest/helm"
echo "install-tools: kind ${KIND_VERSION}, kubectl ${KUBECTL_VERSION}, helm ${HELM_VERSION} -> $dest"
