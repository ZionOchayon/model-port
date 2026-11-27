import pytest
from kubernetes.client import ApiException

from modelport.errors import PlatformUnavailableError
from modelport_platform.kube import KubeClient, Plural, RayJobPhase, rayjob_status, rayservice_apps_ready


class FakeApi:
    def __init__(self, create_status=None, delete_status=None, get_status=None):
        self.objects = {}
        self.create_status, self.delete_status, self.get_status = create_status, delete_status, get_status
        self.calls = []

    def create_namespaced_custom_object(self, group, version, namespace, plural, body):
        self.calls.append(("create", plural, body["metadata"]["name"]))
        if self.create_status:
            raise ApiException(status=self.create_status)
        self.objects[(plural, body["metadata"]["name"])] = dict(body)
        return dict(body)

    def get_namespaced_custom_object(self, group, version, namespace, plural, name):
        if self.get_status:
            raise ApiException(status=self.get_status)
        if (plural, name) not in self.objects:
            raise ApiException(status=404)
        return self.objects[(plural, name)]

    def patch_namespaced_custom_object(self, group, version, namespace, plural, name, body):
        self.objects[(plural, name)] = {**self.objects[(plural, name)], **body}
        return self.objects[(plural, name)]

    def delete_namespaced_custom_object(self, group, version, namespace, plural, name):
        if self.delete_status:
            raise ApiException(status=self.delete_status)
        self.objects.pop((plural, name), None)


BODY = {"apiVersion": "ray.io/v1", "kind": "RayJob", "metadata": {"name": "train-x-1", "namespace": "ray"}}


def test_ensure_creates():
    api = FakeApi()
    obj, created = KubeClient("ray", api=api).ensure(Plural.RAYJOBS, BODY)
    assert created
    assert obj["metadata"]["name"] == "train-x-1"


def test_ensure_adopts_on_409():
    api = FakeApi(create_status=409)
    api.objects[("rayjobs", "train-x-1")] = {**BODY, "status": {"jobStatus": "RUNNING"}}
    obj, created = KubeClient("ray", api=api).ensure(Plural.RAYJOBS, BODY)
    assert not created
    assert obj["status"]["jobStatus"] == "RUNNING"


def test_get_missing_returns_none_and_patch_merges():
    api = FakeApi()
    kube = KubeClient("ray", api=api)
    assert kube.get(Plural.RAYJOBS, "nope") is None
    kube.ensure(Plural.RAYSERVICES, {**BODY, "kind": "RayService"})
    assert kube.merge_patch(Plural.RAYSERVICES, "train-x-1", {"spec": {"a": 1}})["spec"] == {"a": 1}


def test_delete_ignores_404():
    KubeClient("ray", api=FakeApi(delete_status=404)).delete(Plural.RAYJOBS, "gone")
    with pytest.raises(ApiException):
        KubeClient("ray", api=FakeApi(delete_status=500)).delete(Plural.RAYJOBS, "x")


def test_forbidden_is_a_platform_error_with_fix():
    with pytest.raises(PlatformUnavailableError) as info:
        KubeClient("ray", api=FakeApi(get_status=403)).get(Plural.RAYJOBS, "x")
    assert "airflow-rbac.yaml" in info.value.fix


DEPLOYMENT = [
    "",
    "Initializing",
    "Running",
    "Complete",
    "Failed",
    "ValidationFailed",
    "Suspending",
    "Suspended",
    "Retrying",
    "Waiting",
]
JOB = ["", "PENDING", "RUNNING", "STOPPED", "SUCCEEDED", "FAILED"]


def expected_phase(deployment, job):
    if deployment == "Complete" and job == "SUCCEEDED":
        return RayJobPhase.SUCCEEDED
    if deployment in {"Failed", "ValidationFailed"} or job in {"FAILED", "STOPPED"}:
        return RayJobPhase.FAILED
    if deployment == "Running":
        return RayJobPhase.RUNNING
    return RayJobPhase.PENDING


@pytest.mark.parametrize("deployment", DEPLOYMENT)
@pytest.mark.parametrize("job", JOB)
def test_rayjob_status_mapping(deployment, job):
    status = rayjob_status({"status": {"jobDeploymentStatus": deployment, "jobStatus": job}})
    assert status.phase is expected_phase(deployment, job)


def test_rayjob_status_fields():
    status = rayjob_status(
        {
            "status": {
                "jobDeploymentStatus": "Failed",
                "jobStatus": "FAILED",
                "reason": "AppFailed",
                "message": "train_loop raised",
                "startTime": "2026-11-27T10:00:00Z",
                "endTime": "2026-11-27T10:05:30Z",
                "rayClusterName": "train-x-1-raycluster-abcde",
            }
        }
    )
    assert (status.reason, status.message, status.cluster_name) == (
        "AppFailed",
        "train_loop raised",
        "train-x-1-raycluster-abcde",
    )
    assert (status.end_time - status.start_time).total_seconds() == 330
    assert rayjob_status({}).phase is RayJobPhase.PENDING


def service(generation=2, observed=2, service_status="Running", apps=None):
    return {
        "metadata": {"generation": generation},
        "status": {
            "observedGeneration": observed,
            "serviceStatus": service_status,
            "activeServiceStatus": {"applicationStatuses": apps or {}},
        },
    }


def test_rayservice_ready_requires_observed_generation():
    apps = {"m": {"status": "RUNNING"}, "m--champion": {"status": "RUNNING"}}
    assert rayservice_apps_ready(service(apps=apps), ["m", "m--champion"]) == (True, None)
    assert rayservice_apps_ready(service(generation=3, apps=apps), ["m"]) == (False, None)
    assert rayservice_apps_ready(service(service_status="", apps=apps), ["m"]) == (False, None)
    assert rayservice_apps_ready(service(apps={"m": {"status": "DEPLOYING"}}), ["m"]) == (False, None)
    assert rayservice_apps_ready(service(apps=apps), ["m", "other"]) == (False, None)


def test_rayservice_deploy_failed_reported():
    apps = {
        "m": {"status": "RUNNING"},
        "m--candidate": {"status": "DEPLOY_FAILED", "message": "self-test failed"},
    }
    ready, failure = rayservice_apps_ready(service(apps=apps), ["m", "m--candidate"])
    assert not ready
    assert failure == "m--candidate DEPLOY_FAILED: self-test failed"


def test_rayservice_ready_from_ready_condition():
    # KubeRay 1.7 reports readiness as a Ready condition; serviceStatus may be absent
    apps = {"m": {"status": "RUNNING"}}
    obj = service(service_status=None, apps=apps)
    del obj["status"]["serviceStatus"]
    assert rayservice_apps_ready(obj, ["m"]) == (False, None)
    obj["status"]["conditions"] = [{"type": "Ready", "status": "True"}]
    assert rayservice_apps_ready(obj, ["m"]) == (True, None)


def test_rayservice_deploy_failed_in_pending_status_reported():
    # the first deployment of a RayService reports its apps under pendingServiceStatus
    obj = service(service_status=None, apps={})
    obj["status"]["pendingServiceStatus"] = {
        "applicationStatuses": {"m--candidate": {"status": "DEPLOY_FAILED", "message": "self-test failed"}}
    }
    assert rayservice_apps_ready(obj, ["m--candidate"]) == (
        False,
        "m--candidate DEPLOY_FAILED: self-test failed",
    )
