import pytest
from botocore.exceptions import ClientError

from modelport.naming import Bucket
from modelport_platform.tools import bootstrap_buckets


class FakeS3:
    def __init__(self, existing=(), head_fails=()):
        self.buckets = set(existing)
        self.head_fails = set(head_fails)

    def create_bucket(self, Bucket):
        if Bucket in self.buckets:
            raise ClientError({"Error": {"Code": "BucketAlreadyOwnedByYou"}}, "CreateBucket")
        self.buckets.add(Bucket)

    def head_bucket(self, Bucket):
        if Bucket in self.head_fails or Bucket not in self.buckets:
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")


def test_idempotent_existing_buckets():
    client = FakeS3(existing={"incoming", "mlflow"})
    created = bootstrap_buckets.bootstrap(client)
    assert sorted(created) == sorted(b.value for b in Bucket if b.value not in {"incoming", "mlflow"})
    assert bootstrap_buckets.bootstrap(client) == []
    assert client.buckets == {b.value for b in Bucket}


def test_unexpected_create_error_raises():
    class Denied(FakeS3):
        def create_bucket(self, Bucket):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "CreateBucket")

    with pytest.raises(ClientError):
        bootstrap_buckets.bootstrap(Denied())


def test_main_exits_1_when_a_bucket_is_missing(monkeypatch):
    monkeypatch.setattr(bootstrap_buckets, "s3_client", lambda: FakeS3(head_fails={"datasets"}))
    assert bootstrap_buckets.main() == 1
