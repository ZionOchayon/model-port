"""In-memory fakes of the platform ports."""

from modelport.errors import NotFoundError


class InMemoryDatasetStore:
    """DatasetStore over a dict; `writes` records the order of writes (manifest-last checks)."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []

    def read_bytes(self, uri):
        try:
            return self.objects[uri]
        except KeyError:
            raise NotFoundError(what=f"{uri} does not exist", fix="-") from None

    def write_bytes(self, uri, data, *, content_type="application/octet-stream"):
        self.objects[uri] = bytes(data)
        self.writes.append(uri)

    def exists(self, uri):
        return uri in self.objects

    def list(self, prefix_uri):
        return sorted(u for u in self.objects if u.startswith(prefix_uri))

    def copy(self, src_uri, dst_uri):
        self.write_bytes(dst_uri, self.read_bytes(src_uri))

    def delete(self, uri):
        self.objects.pop(uri, None)
