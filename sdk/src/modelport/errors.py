"""The modelport error hierarchy: every error says what failed and how to fix it (NFR-6).

Each subclass carries the CLI exit code for its kind of failure; codes 2-12 are part of the CLI contract.
"""

from collections.abc import Mapping
from typing import ClassVar


class ModelportError(Exception):
    """Base error: `what` failed, `fix` says what to do, `details` holds machine-readable context."""

    exit_code: ClassVar[int] = 1

    def __init__(self, what: str, fix: str, *, details: Mapping[str, object] | None = None) -> None:
        super().__init__(what)
        self.what = what
        self.fix = fix
        self.details: Mapping[str, object] = dict(details or {})

    def __str__(self) -> str:
        return f"{self.what}\n  How to fix: {self.fix}"


class SpecError(ModelportError):
    """model.yaml is missing or invalid, or a name does not follow the naming rules."""

    exit_code = 2


class ContractError(ModelportError):
    """The model class breaks the BaseModel contract."""

    exit_code = 3


class SignatureMismatchError(ModelportError):
    """Input or output data does not match the model signature."""

    exit_code = 4


class PackagingError(ModelportError):
    """The model folder cannot be packaged."""

    exit_code = 5


class PlatformUnavailableError(ModelportError):
    """Airflow, MLflow or Serve cannot be reached."""

    exit_code = 6


class NotFoundError(ModelportError):
    """A model, version, alias or object does not exist."""

    exit_code = 7


class PromotionRefusedError(ModelportError):
    """A promotion or rollback is not allowed (gate not passed, nothing to roll back to)."""

    exit_code = 8


class IllegalTransitionError(ModelportError):
    """A lifecycle transition is not allowed from the version's current state."""

    exit_code = 9


class RemoteOperationError(ModelportError):
    """A remote operation failed (a pipeline run failed, predict returned a server error)."""

    exit_code = 10


class ConfigError(ModelportError):
    """Settings or credentials are missing or invalid."""

    exit_code = 11


class WaitTimeoutError(ModelportError):
    """Waiting for a result took longer than the timeout."""

    exit_code = 12
