import pytest

from modelport.config import ClientSettings
from modelport.errors import ConfigError


def test_settings_defaults_and_overrides(isolated_cwd):
    s = ClientSettings.from_env({"MODELPORT_AIRFLOW_PASSWORD": "pw"})
    assert (s.airflow_url, s.airflow_username, s.airflow_password) == (
        "http://localhost:8080",
        "modelport",
        "pw",
    )
    assert (s.mlflow_url, s.serve_url, s.ray_dashboard_url) == (
        "http://localhost:5000",
        "http://localhost:8000",
        "http://localhost:8265",
    )
    assert s.http_timeout_s == 30.0
    s = ClientSettings.from_env(
        {
            "MODELPORT_AIRFLOW_PASSWORD": "pw",
            "MODELPORT_AIRFLOW_URL": "http://af:1",
            "MODELPORT_HTTP_TIMEOUT": "5",
        }
    )
    assert (s.airflow_url, s.http_timeout_s) == ("http://af:1", 5.0)


def test_settings_reads_credentials_file(isolated_cwd):
    creds = isolated_cwd / ".modelport" / "credentials.env"
    creds.parent.mkdir()
    creds.write_text(
        "# comment\nMODELPORT_AIRFLOW_USERNAME=alice\nMODELPORT_AIRFLOW_PASSWORD=s3cret\n", encoding="utf-8"
    )
    nested = isolated_cwd / "examples" / "models"
    nested.mkdir(parents=True)
    s = ClientSettings.from_env({}, credentials_file=creds)
    assert (s.airflow_username, s.airflow_password) == ("alice", "s3cret")
    import os

    os.chdir(nested)
    assert ClientSettings.from_env({}).airflow_password == "s3cret"
    assert ClientSettings.from_env({"MODELPORT_AIRFLOW_PASSWORD": "env"}).airflow_password == "env"


def test_settings_missing_password_raises_config_error(isolated_cwd):
    with pytest.raises(ConfigError) as info:
        ClientSettings.from_env({})
    assert "make up" in info.value.fix


def test_bad_timeout_raises_config_error(isolated_cwd):
    with pytest.raises(ConfigError):
        ClientSettings.from_env({"MODELPORT_AIRFLOW_PASSWORD": "pw", "MODELPORT_HTTP_TIMEOUT": "soon"})
