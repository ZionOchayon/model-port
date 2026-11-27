"""Client for the Serve endpoint (`POST /<model>/predict`), used by `modelport predict` and load tests."""

import time
from typing import Any

import requests

from modelport.errors import (
    ModelportError,
    NotFoundError,
    PlatformUnavailableError,
    RemoteOperationError,
    SignatureMismatchError,
)
from modelport.logs import get_logger
from modelport.serving import PredictRequest, PredictResponse

_log = get_logger(__name__)


def _message(response: Any) -> str:  # noqa: ANN401 - requests.Response
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return str(response.text)[:300]


class ServeClient:
    """Errors carry the HTTP status in `details["status"]`."""

    def __init__(
        self, base_url: str, *, timeout_s: float = 30.0, session: requests.Session | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._session = session or requests.Session()

    def predict(self, model: str, request: PredictRequest) -> PredictResponse:
        """One prediction call.

        Raises:
            NotFoundError: no endpoint for the model (404).
            SignatureMismatchError: the rows do not fit the model's signature (422).
            RemoteOperationError: the server failed (5xx).
            PlatformUnavailableError: Serve cannot be reached.
        """
        url = f"{self.base_url}/{model}/predict"
        start = time.perf_counter()
        try:
            response = self._session.post(url, json=request.model_dump(), timeout=self._timeout_s)
        except requests.RequestException as err:
            raise PlatformUnavailableError(
                what=f"Cannot reach Serve at {self.base_url} ({type(err).__name__})",
                fix="Run `make ui` to open the port-forwards (after `make up`).",
            ) from err
        status = response.status_code
        if status == 200:
            parsed = PredictResponse.model_validate(response.json())
            seconds = round(time.perf_counter() - start, 3)
            _log.info("serve.predicted", model=model, rows=len(request.instances), seconds=seconds)
            return parsed
        error: type[ModelportError]
        if status == 404:
            error, what, fix = (
                NotFoundError,
                f"Serve has no endpoint for '{model}'",
                f"The model has no champion yet; check `modelport status {model}`.",
            )
        elif status == 422:
            error, what, fix = (
                SignatureMismatchError,
                _message(response),
                "Send the columns and types of the model's signature (`modelport status` links the version).",
            )
        else:
            error, what, fix = (
                RemoteOperationError,
                f"Serve answered HTTP {status}: {_message(response)}",
                f"Retry; `modelport status {model}` shows which version is serving.",
            )
        raise error(what=what, fix=fix, details={"status": status})
