"""Client settings for the modelport CLI, from the environment and the file `make up` writes."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from modelport.errors import ConfigError

CREDENTIALS_FILE = Path(".modelport") / "credentials.env"


def _read_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines; blank lines and # comments are ignored."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _find_credentials_file(start: Path) -> Path | None:
    """`.modelport/credentials.env` in `start` or the nearest parent (so it works from any repo folder)."""
    for folder in (start, *start.parents):
        if (folder / CREDENTIALS_FILE).is_file():
            return folder / CREDENTIALS_FILE
    return None


@dataclass(frozen=True)
class ClientSettings:
    """Where the CLI finds Airflow, MLflow, Serve and the Ray dashboard, and how long it waits."""

    airflow_url: str
    airflow_username: str
    airflow_password: str
    mlflow_url: str
    serve_url: str
    ray_dashboard_url: str
    http_timeout_s: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, credentials_file: Path | None = None) -> Self:
        """Read settings: environment first, then the credentials file, then the local defaults.

        Args:
            env: environment to read; defaults to `os.environ`.
            credentials_file: path of a KEY=value file; defaults to `.modelport/credentials.env` found
                in the current folder or a parent.

        Raises:
            ConfigError: no Airflow password is available, or a value is malformed.
        """
        environ = os.environ if env is None else env
        path = credentials_file or _find_credentials_file(Path.cwd())
        values = {**(_read_env_file(path) if path and path.is_file() else {}), **environ}
        password = values.get("MODELPORT_AIRFLOW_PASSWORD")
        if not password:
            raise ConfigError(
                what="No Airflow password found",
                fix="Run `make up` (it writes .modelport/credentials.env) or set MODELPORT_AIRFLOW_PASSWORD.",
            )
        timeout = values.get("MODELPORT_HTTP_TIMEOUT", "30")
        try:
            timeout_s = float(timeout)
        except ValueError:
            raise ConfigError(
                what=f"MODELPORT_HTTP_TIMEOUT is '{timeout}', not a number of seconds",
                fix="Set MODELPORT_HTTP_TIMEOUT to a number, e.g. 30.",
            ) from None
        return cls(
            airflow_url=values.get("MODELPORT_AIRFLOW_URL", "http://localhost:8080"),
            airflow_username=values.get("MODELPORT_AIRFLOW_USERNAME", "modelport"),
            airflow_password=password,
            mlflow_url=values.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            serve_url=values.get("MODELPORT_SERVE_URL", "http://localhost:8000"),
            ray_dashboard_url=values.get("MODELPORT_RAY_DASHBOARD_URL", "http://localhost:8265"),
            http_timeout_s=timeout_s,
        )
