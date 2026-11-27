import pytest
import requests

from modelport.clients.serve import ServeClient
from modelport.errors import (
    NotFoundError,
    PlatformUnavailableError,
    RemoteOperationError,
    SignatureMismatchError,
)
from modelport.serving import PredictRequest

REQUEST = PredictRequest(instances=[{"x": 1.0}])
OK = {
    "model": "m",
    "version": "3",
    "variant": "champion",
    "predictions": [{"prediction": "a", "confidence": 0.9}],
}


class Response:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload
        self.text = str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class Session:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def post(self, url, *, json=None, timeout=None):
        self.calls.append((url, json))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def client(answer):
    return ServeClient("http://serve:8000/", session=Session(answer))


def test_predict_returns_response():
    c = client(Response(200, OK))
    response = c.predict("m", REQUEST)
    assert (response.version, response.variant, response.predictions[0].prediction) == ("3", "champion", "a")
    assert c._session.calls == [("http://serve:8000/m/predict", {"instances": [{"x": 1.0}]})]


def test_predict_404_message():
    with pytest.raises(NotFoundError, match="Serve has no endpoint for 'm'") as err:
        client(Response(404, {"error": "no route"})).predict("m", REQUEST)
    assert "no champion yet" in err.value.fix
    assert err.value.details["status"] == 404


def test_predict_422_signature_error():
    with pytest.raises(SignatureMismatchError, match="row 0 has no column 'x'") as err:
        client(Response(422, {"error": "row 0 has no column 'x'"})).predict("m", REQUEST)
    assert err.value.details["status"] == 422


def test_predict_5xx_and_connection_errors():
    with pytest.raises(RemoteOperationError, match="candidate failed") as err:
        client(Response(500, {"error": "candidate failed: RuntimeError"})).predict("m", REQUEST)
    assert err.value.details["status"] == 500
    with pytest.raises(PlatformUnavailableError, match="make ui"):
        client(requests.ConnectionError("refused")).predict("m", REQUEST)
