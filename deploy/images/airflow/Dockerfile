# modelport Airflow image: official Airflow + hash-locked extras (constraints-based) + modelport wheels + DAGs.
ARG AIRFLOW_BASE_IMAGE
FROM ${AIRFLOW_BASE_IMAGE}

ARG AIRFLOW_VERSION
# Lock and wheels are bind-mounted (BuildKit), so they never enter a layer.
RUN --mount=type=bind,source=deploy/images/airflow/requirements.lock,target=/tmp/requirements.lock \
    test -n "${AIRFLOW_VERSION}" \
    && grep -q "^apache-airflow==${AIRFLOW_VERSION} " /tmp/requirements.lock \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

RUN --mount=type=bind,source=dist,target=/tmp/wheels \
    pip install --no-cache-dir --no-deps /tmp/wheels/modelport-*.whl /tmp/wheels/modelport_platform-*.whl

COPY --chown=airflow:root platform/dags/ /opt/airflow/dags/
