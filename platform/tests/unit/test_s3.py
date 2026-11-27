import pytest
from botocore.exceptions import ClientError

from modelport.errors import NotFoundError
from modelport_platform.s3 import S3DatasetStore, split_uri


class FakeS3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        class Body:
            def __init__(self, data):
                self.data = data

            def read(self):
                return self.data

        return {"Body": Body(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def get_paginator(self, name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for b, k in objects if b == Bucket and k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys[:2]]}
                yield {"Contents": [{"Key": k} for k in keys[2:]]}

        return Paginator()

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


def test_split_uri():
    assert split_uri("s3://datasets/fashion-mnist/v1/train.parquet") == (
        "datasets",
        "fashion-mnist/v1/train.parquet",
    )
    assert split_uri("s3://incoming/") == ("incoming", "")
    with pytest.raises(ValueError, match="s3://"):
        split_uri("/tmp/x")


def test_store_missing_key_not_found():
    with pytest.raises(NotFoundError) as info:
        S3DatasetStore(FakeS3()).read_bytes("s3://datasets/x/contract.json")
    assert "s3://datasets/x/contract.json" in info.value.what


def test_store_roundtrip_list_copy_delete():
    store = S3DatasetStore(FakeS3())
    for name in ("a", "b", "c"):
        store.write_bytes(f"s3://incoming/ds/{name}.parquet", name.encode())
    assert store.list("s3://incoming/ds/") == [f"s3://incoming/ds/{n}.parquet" for n in "abc"]
    store.copy("s3://incoming/ds/a.parquet", "s3://quarantine/ds/1/a.parquet")
    assert store.read_bytes("s3://quarantine/ds/1/a.parquet") == b"a"
    store.delete("s3://incoming/ds/a.parquet")
    assert not store.exists("s3://incoming/ds/a.parquet")
    assert store.exists("s3://incoming/ds/b.parquet")
