import json

from modelport.logs import configure_logging, get_logger


def test_json_logs_have_fixed_keys(capsys):
    configure_logging(json=True, level="INFO")
    get_logger("t").info("registry.alias_set", model="m-1")
    get_logger("t").debug("hidden")
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert {"timestamp", "level", "event", "model", "logger"} <= record.keys()
    assert record["event"] == "registry.alias_set"
    assert record["level"] == "info"
