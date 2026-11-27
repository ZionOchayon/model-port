"""retrain_on_dataset: when a dataset gets a new version, start training for every subscribed model (FR-8).

A model subscribes through its registered-model tags (`modelport.dataset`, `modelport.auto_retrain`), which
`modelport submit` sets; it retrains with its latest submission and spec file.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Asset, AssetAny, dag, task

from modelport.naming import Tag, asset_uri, retrain_run_id
from modelport.registry import ModelRegistry
from modelport_platform import dagkit


def subscribers(events: Iterable[Mapping[str, Any]], registry: ModelRegistry) -> list[dict[str, Any]]:
    """One train_model trigger per (subscribed model, new dataset version)."""
    triggers = []
    for extra in events:
        dataset, version = str(extra["dataset"]), int(extra["version"])
        for model in registry.list_models(tag_equals={Tag.DATASET: dataset, Tag.AUTO_RETRAIN: "true"}):
            tags = registry.model_tags(model)
            triggers.append(
                {
                    "trigger_run_id": retrain_run_id(model, dataset, version),
                    "conf": {
                        "model": model,
                        "submission_run_id": tags[Tag.LATEST_SUBMISSION_RUN_ID],
                        "dataset_version": version,
                        "spec_file": tags.get(Tag.LATEST_SPEC_FILE, "model.yaml"),
                    },
                }
            )
    return triggers


@dag(
    dag_id="retrain_on_dataset",
    schedule=AssetAny(*[Asset(asset_uri(d)) for d in dagkit.settings().datasets]),
    max_active_runs=1,
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "ingest"],
    doc_md=__doc__,
)
def retrain_on_dataset() -> None:
    """Trigger train_model for each model subscribed to the dataset that changed."""

    @task
    def find_subscribers(triggering_asset_events: Any = None) -> list[dict[str, Any]]:  # noqa: ANN401
        events = [e.extra for batch in (triggering_asset_events or {}).values() for e in batch]
        return subscribers(events, dagkit.registry())

    TriggerDagRunOperator.partial(
        task_id="trigger_training", trigger_dag_id="train_model", skip_when_already_exists=True
    ).expand_kwargs(find_subscribers())


retrain_on_dataset()
