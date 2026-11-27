from pathlib import Path

import pytest
import yaml

from modelport.naming import DNS_1123, rayjob_name
from modelport.spec import load_spec
from modelport_platform.ray.manifests import build_train_rayjob
from modelport_platform.settings import PlatformSettings

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS = Path(__file__).parent / "snapshots"
SPEC = load_spec(ROOT / "examples" / "models" / "image-classifier" / "model.fashion.yaml")
ENV = {
    "MODELPORT_S3_ENDPOINT": "http://minio.platform.svc.cluster.local:9000",
    "AWS_ACCESS_KEY_ID": "k",
    "AWS_SECRET_ACCESS_KEY": "s",
    "MLFLOW_TRACKING_URI": "http://mlflow.platform.svc.cluster.local:5000",
    "MODELPORT_RAY_IMAGE": "modelport/ray:0123456789ab",
}


def rayjob(profile="smoke", dataset_version=None, spec=SPEC):
    settings = PlatformSettings.from_env({**ENV, "MODELPORT_PROFILE": profile})
    return build_train_rayjob(
        name=rayjob_name(spec.name, "submit__fashion-classifier__0123456789ab"),
        spec=spec,
        submission_run_id="9f2c1d",
        spec_file="model.fashion.yaml",
        dataset_version=dataset_version,
        working_dir_uri="s3://mlflow/1/9f2c1d/artifacts/code/code.zip",
        settings=settings,
    )


@pytest.mark.parametrize("profile", ["smoke", "full"])
def test_train_rayjob_snapshot(profile):
    rendered = yaml.safe_dump(rayjob(profile), sort_keys=False)
    snapshot = SNAPSHOTS / f"rayjob_train_{profile}.yaml"
    assert rendered == snapshot.read_text(encoding="utf-8")


def test_rayjob_name_is_label_safe():
    name = rayjob()["metadata"]["name"]
    assert DNS_1123.fullmatch(name)
    assert len(name) <= 63
    assert rayjob()["metadata"]["labels"]["modelport.dev/model"] == "fashion-classifier"


def test_head_has_zero_cpus():
    assert rayjob()["spec"]["rayClusterSpec"]["headGroupSpec"]["rayStartParams"]["num-cpus"] == "0"


def test_worker_count_matches_spec():
    group = rayjob()["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]
    workers = SPEC.resources.train.workers
    assert (group["replicas"], group["minReplicas"], group["maxReplicas"]) == (workers, workers, workers)
    assert group["rayStartParams"]["num-cpus"] == "1"


def test_runtime_env_has_pinned_pip_and_working_dir():
    env = yaml.safe_load(rayjob()["spec"]["runtimeEnvYAML"])
    assert env["working_dir"] == "s3://mlflow/1/9f2c1d/artifacts/code/code.zip"
    assert env["pip"] == {
        "packages": ["torch==2.15.0", "torchvision==0.29.2", "pillow==13.0.0"],
        "pip_check": False,
    }


def test_entrypoint_and_dataset_version():
    entry = rayjob(dataset_version=3)["spec"]["entrypoint"]
    assert entry == (
        "python -m modelport_platform.train.entrypoint --model fashion-classifier --submission-run-id 9f2c1d"
        f" --spec-file model.fashion.yaml --rayjob {rayjob()['metadata']['name']} --dataset-version 3"
    )
    assert "--dataset-version" not in rayjob()["spec"]["entrypoint"]


def test_pods_get_platform_env_and_sizing():
    spec = rayjob("full")["spec"]["rayClusterSpec"]
    worker = spec["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]
    assert worker["envFrom"] == [
        {"configMapRef": {"name": "modelport-platform-env"}},
        {"secretRef": {"name": "s3-credentials"}},
    ]
    assert worker["env"] == [{"name": "RAY_TRAIN_V2_ENABLED", "value": "1"}]
    assert worker["resources"] == {
        "requests": {"cpu": "1", "memory": "2Gi"},
        "limits": {"cpu": "1", "memory": "2Gi"},
    }
    smoke = rayjob("smoke")["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"][
        "containers"
    ][0]
    assert smoke["resources"]["requests"]["cpu"] == "250m"
