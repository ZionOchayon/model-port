"""In-memory fakes of the SDK ports, for unit tests that must not touch MLflow."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelport.errors import NotFoundError
from modelport.lifecycle import AliasOpKind, AliasSnapshot, ModelState, Plan
from modelport.naming import Alias, Tag, submission_key
from modelport.registry import VersionInfo


class InMemoryRegistry:
    """Implements ModelRegistry and ArtifactStore with dicts."""

    def __init__(self) -> None:
        self.models: dict[str, dict[str, str]] = {}
        self.versions: dict[str, dict[str, VersionInfo]] = {}
        self.aliases: dict[str, dict[str, str]] = {}
        self.runs: dict[str, dict[str, Any]] = {}

    # ModelRegistry
    def ensure_registered_model(self, name: str, tags: Mapping[str, str]) -> None:
        self.models.setdefault(name, {}).update(tags)
        self.versions.setdefault(name, {})
        self.aliases.setdefault(name, {})

    def add_version(self, name: str, tags: Mapping[str, str] | None = None, run_id: str = "") -> VersionInfo:
        self.ensure_registered_model(name, {})
        number = str(len(self.versions[name]) + 1)
        info = VersionInfo(
            name, number, run_id, f"models:/{name}/{number}", dict(tags or {}), (), datetime.now(UTC)
        )
        self.versions[name][number] = info
        return info

    def get_version(self, name: str, version: str) -> VersionInfo:
        try:
            info = self.versions[name][version]
        except KeyError:
            raise NotFoundError(what=f"{name} has no version {version}", fix="Check the version.") from None
        aliases = tuple(a for a, v in self.aliases.get(name, {}).items() if v == version)
        return VersionInfo(
            info.name, info.version, info.run_id, info.source, info.tags, aliases, info.created_at
        )

    def resolve(self, name: str, alias: Alias) -> VersionInfo | None:
        version = self.aliases.get(name, {}).get(alias)
        return self.get_version(name, version) if version else None

    def snapshot(self, name: str) -> AliasSnapshot:
        if name not in self.models:
            raise NotFoundError(what=f"No model named '{name}' is registered", fix="Check the name.")
        a = self.aliases[name]
        named = set(a.values())
        states = {v: ModelState(self.versions[name][v].tags.get(Tag.STATE, "registered")) for v in named}
        return AliasSnapshot(
            name, a.get(Alias.CHAMPION), a.get(Alias.PREVIOUS), a.get(Alias.CANDIDATE), states
        )

    def find_version_by_tag(self, name: str, key: Tag, value: str) -> VersionInfo | None:
        found = [v for v in self.versions.get(name, {}).values() if v.tags.get(key) == value]
        return self.get_version(name, found[-1].version) if found else None

    def set_version_tags(self, name: str, version: str, tags: Mapping[Tag, str]) -> None:
        info = self.versions[name][version]
        self.versions[name][version] = VersionInfo(
            info.name,
            info.version,
            info.run_id,
            info.source,
            {**info.tags, **tags},
            info.aliases,
            info.created_at,
        )

    def set_model_tags(self, name: str, tags: Mapping[Tag, str]) -> None:
        self.models[name].update(tags)

    def apply_plan(self, name: str, plan: Plan) -> None:
        for op in plan.alias_ops:
            if op.kind is AliasOpKind.SET and op.version is not None:
                self.aliases[name][op.alias] = op.version
            else:
                self.aliases[name].pop(op.alias, None)
        for version, state in plan.states.items():
            self.set_version_tags(name, version, {Tag.STATE: state.value})

    def list_models(self, *, tag_equals: Mapping[Tag, str] | None = None) -> list[str]:
        wanted = tag_equals or {}
        return sorted(
            m for m, tags in self.models.items() if all(tags.get(k) == v for k, v in wanted.items())
        )

    def model_uri(self, name: str, version: str) -> str:
        return f"models:/{name}/{version}"

    # ArtifactStore
    def find_submission(self, model: str, key: str) -> str | None:
        found = [r for r, run in self.runs.items() if run["tags"] == {"model": model, "key": key}]
        return found[-1] if found else None

    def create_submission(self, model: str, package: Any, spec_path: Path, spec_sha256: str) -> str:
        run_id = f"run-{len(self.runs) + 1}"
        key = submission_key(package.sha256, spec_path.name, spec_sha256)
        self.runs[run_id] = {"tags": {"model": model, "key": key}, "artifacts": {}}
        return run_id

    def download(self, run_id: str, artifact_path: str, dst: Path) -> Path:
        target = dst / Path(artifact_path).name
        dst.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.runs[run_id]["artifacts"][artifact_path])
        return target

    def log_json(self, run_id: str, artifact_path: str, payload: Mapping[str, Any]) -> None:
        self.runs.setdefault(run_id, {"tags": {}, "artifacts": {}})["artifacts"][artifact_path] = json.dumps(
            dict(payload)
        ).encode()

    def read_json(self, run_id: str, artifact_path: str) -> dict[str, Any]:
        return json.loads(self.runs[run_id]["artifacts"][artifact_path])

    def artifact_s3_uri(self, run_id: str, artifact_path: str) -> str:
        return f"s3://mlflow/0/{run_id}/artifacts/{artifact_path}"

    def run_url(self, run_id: str) -> str:
        return f"http://localhost:5000/#/runs/{run_id}"
