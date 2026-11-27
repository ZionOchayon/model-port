"""DAG tests run with the Airflow image's locked environment; DAG files read platform settings at parse."""

import os

os.environ.update(
    {
        "MODELPORT_PROFILE": "smoke",
        "MODELPORT_S3_ENDPOINT": "http://minio.platform.svc.cluster.local:9000",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "MLFLOW_TRACKING_URI": "http://mlflow.platform.svc.cluster.local:5000",
        "MODELPORT_RAY_IMAGE": "modelport/ray:test",
        "AIRFLOW__CORE__LOAD_EXAMPLES": "False",  # Airflow 3 DagBag has no include_examples argument
    }
)
