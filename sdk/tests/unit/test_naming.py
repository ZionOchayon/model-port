from datetime import date

import pytest

from modelport import naming
from modelport.errors import SpecError

DAY = date(2026, 1, 2)


def test_paths_match_contract():
    assert naming.dataset_root("fashion-mnist") == "s3://datasets/fashion-mnist"
    assert naming.dataset_version_uri("fashion-mnist", 3) == "s3://datasets/fashion-mnist/v3"
    assert naming.incoming_prefix("covertype") == "s3://incoming/covertype/"
    assert naming.quarantine_uri("covertype", "ab12") == "s3://quarantine/covertype/ab12/"
    assert naming.checkpoint_root("image-classifier") == "s3://checkpoints/image-classifier"
    assert naming.prediction_uri("image-classifier", DAY) == "s3://predictions/image-classifier/2026-01-02/"
    assert (
        naming.inference_log_prefix("image-classifier", DAY)
        == "s3://inference-logs/image-classifier/2026-01-02/"
    )
    assert naming.batch_input_uri("covertype", DAY) == "s3://batch-input/covertype/2026-01-02/"
    assert naming.asset_uri("covertype") == "s3://datasets/covertype"
    assert naming.experiment_training("image-classifier") == "modelport-train/image-classifier"
    assert naming.serve_app_names("m-1") == ("m-1", "m-1--champion", "m-1--candidate")
    assert naming.EXPERIMENT_SUBMISSIONS == "modelport-submissions"
    assert naming.EXPERIMENT_EVALUATION == "modelport-evaluation"
    assert [a.value for a in naming.Alias] == ["candidate", "champion", "previous"]
    assert naming.Tag.GATE_RESULT == "modelport.gate.result"
    assert naming.Tag.SUBMISSION_KEY == "modelport.submission_key"


def test_run_ids_match_contract():
    key = naming.submission_key("c0de", "model.yaml", "5bec")
    assert len(key) == 64
    assert naming.submit_run_id("image-classifier", key) == f"submit__image-classifier__{key[:12]}"
    assert naming.retrain_run_id("m-1", "covertype", 2) == "retrain__m-1__covertype__v2"
    assert naming.evaluate_run_id("m-1", "7") == "evaluate__m-1__v7"
    assert naming.deploy_run_id("m-1", "7") == "deploy__m-1__v7"
    assert naming.serve_sync_run_id("promote", "m-1", "20260102T030405Z") == "promote__m-1__20260102T030405Z"
    assert naming.shared_job_id("eval", "m-1", "v7-ab12cd34") == "eval__m-1__v7-ab12cd34"


def test_rayjob_name_length():
    name = naming.rayjob_name("a" * 39 + "b", "submit__x__0123456789ab")
    assert len(name) <= 63
    assert name.startswith("train-" + "a" * 39 + "b-")
    assert naming.DNS_1123.fullmatch(name)


def test_run_ids_deterministic():
    assert naming.rayjob_name("m-1", "run") == naming.rayjob_name("m-1", "run")
    assert naming.submission_key("a", "b", "c") == naming.submission_key("a", "b", "c")
    assert naming.submission_key("a", "model.yaml", "c") != naming.submission_key(
        "a", "model.fashion.yaml", "c"
    )


@pytest.mark.parametrize("bad", ["Image_Classifier", "a", "a" * 41, "-abc", "abc-"])
def test_invalid_model_name_rejected(bad):
    with pytest.raises(SpecError):
        naming.checkpoint_root(bad)


@pytest.mark.parametrize(
    ("call", "args"),
    [
        (naming.dataset_root, ("Fashion",)),
        (naming.dataset_version_uri, ("covertype", 0)),
        (naming.evaluate_run_id, ("m-1", "v7")),
        (naming.shared_job_id, ("eval", "m-1", "bad key!")),
        (naming.serve_sync_run_id, ("delete", "m-1", "20260102T030405Z")),
    ],
)
def test_invalid_inputs_rejected(call, args):
    with pytest.raises(SpecError):
        call(*args)
