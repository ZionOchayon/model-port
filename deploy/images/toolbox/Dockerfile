# modelport dev toolbox: every tool `make` needs, pinned from versions.env (ADR-0019).
# The image args come from docker-compose.yml and have no defaults on purpose: a missing pin fails the build.
ARG TOOLBOX_BASE_IMAGE
ARG DOCKER_CLI_IMAGE
ARG PROMETHEUS_IMAGE
ARG ALERTMANAGER_IMAGE

FROM ${DOCKER_CLI_IMAGE} AS dockercli
FROM ${PROMETHEUS_IMAGE} AS prometheus
FROM ${ALERTMANAGER_IMAGE} AS alertmanager

FROM ${TOOLBOX_BASE_IMAGE}

# Python 3.12 is Ubuntu 24.04's python3; uv uses it instead of downloading one.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         bash ca-certificates curl git jq make openssl python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# kind, kubectl, helm (the same sha256-checked script CI uses), then uv and gh, all from versions.env.
COPY versions.env scripts/install-tools.sh /tmp/tools/
RUN set -eu; . /tmp/tools/versions.env; cd /tmp \
    && DEST=/usr/local/bin bash /tmp/tools/install-tools.sh \
    && curl -fsSLo uv.tgz "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${UV_SHA256_LINUX_AMD64}  uv.tgz" | sha256sum -c - \
    && tar -xzf uv.tgz && install -m 0755 uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/ \
    && curl -fsSLo gh.tgz "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
    && echo "${GH_SHA256_LINUX_AMD64}  gh.tgz" | sha256sum -c - \
    && tar -xzf gh.tgz && install -m 0755 "gh_${GH_VERSION}_linux_amd64/bin/gh" /usr/local/bin/gh \
    && rm -rf /tmp/*

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
# buildx makes `docker build` use BuildKit (the images use RUN --mount).
COPY --from=dockercli /usr/local/libexec/docker/cli-plugins/docker-buildx /usr/local/libexec/docker/cli-plugins/docker-buildx
COPY --from=prometheus /bin/promtool /usr/local/bin/promtool
COPY --from=alertmanager /bin/amtool /usr/local/bin/amtool

# The venv lives outside the bind-mounted repo so it never clashes with a host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    MODELPORT_TOOLBOX=1 \
    KUBECONFIG=/workspace/.modelport/kubeconfig

RUN git config --system --add safe.directory /workspace
WORKDIR /workspace
CMD ["bash"]
