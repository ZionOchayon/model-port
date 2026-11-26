# modelport Ray image: the pinned Ray CPU image + hash-locked ML deps + the modelport wheels.
# Build args have no defaults on purpose: a missing pin fails the build.
ARG RAY_BASE_IMAGE
FROM ${RAY_BASE_IMAGE}

ARG RAY_VERSION
ARG TORCH_VERSION
ARG XGBOOST_VERSION

# uv writes `ray==<v>` (extras resolved into the lock), so the pins are checked without extras.
# The lock and the wheels are bind-mounted (BuildKit): they never enter a layer and need no cleanup,
# which also avoids deleting root-owned files as the image's `ray` user.
RUN --mount=type=bind,source=deploy/images/ray/requirements.lock,target=/tmp/requirements.lock \
    test -n "${RAY_VERSION}" && test -n "${TORCH_VERSION}" && test -n "${XGBOOST_VERSION}" \
    && grep -q "^ray==${RAY_VERSION} " /tmp/requirements.lock \
    && grep -q "^torch==${TORCH_VERSION}+cpu " /tmp/requirements.lock \
    && grep -q "^xgboost==${XGBOOST_VERSION} " /tmp/requirements.lock \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
         --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

RUN --mount=type=bind,source=dist,target=/tmp/wheels \
    pip install --no-cache-dir --no-deps /tmp/wheels/modelport-*.whl /tmp/wheels/modelport_platform-*.whl
