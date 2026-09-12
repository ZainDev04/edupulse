"""MLflow experiment tracking behind the registry."""

import json

import pandas as pd
import pytest

from edupulse.models.tracking import ExperimentTracker, _scalars, mlflow_available, search_runs
from edupulse.tasks import get_task

needs_mlflow = pytest.mark.skipif(not mlflow_available(), reason="mlflow not installed")


def test_scalars_keep_finite_numbers_only():
    out = _scalars({"a": 1, "b": 2.5, "c": float("nan"), "d": "x", "e": True, "f": [1], "g": float("inf")})
    assert out == {"a": 1.0, "b": 2.5}


def test_disabled_tracker_is_a_noop(fast_settings):
    tracker = ExperimentTracker(fast_settings, enabled=False)
    assert not tracker.enabled
    with tracker.run(get_task("at_risk"), "v1") as t:
        t.log_params({"x": 1})
        t.log_metrics({"m": 0.5})
        t.set_tags({"k": "v"})
        t.log_figures({"f": "does/not/exist.png"})
        t.log_model(None)
    assert tracker.info == {}


def test_bare_path_becomes_sqlite_uri(fast_settings, tmp_path):
    s = fast_settings.model_copy(update={"tracking_uri": str(tmp_path / "store.db")})
    uri = ExperimentTracker(s, enabled=False).tracking_uri
    assert uri.startswith("sqlite:///") and uri.endswith("store.db")
    s = fast_settings.model_copy(update={"tracking_uri": "http://mlflow.internal:5000"})
    assert ExperimentTracker(s, enabled=False).tracking_uri == "http://mlflow.internal:5000"


@needs_mlflow
def test_pipeline_run_is_recorded(trained_at_risk, fast_settings):
    assert trained_at_risk.run_id, "run_pipeline should return the MLflow run id"

    meta = json.loads((trained_at_risk.artefact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["tracking"]["run_id"] == trained_at_risk.run_id
    assert "MLflow run" in (trained_at_risk.artefact_dir / "model_card.md").read_text(encoding="utf-8")

    runs = search_runs("at_risk", fast_settings)
    assert len(runs) >= 1
    row = runs[runs["run_id"] == trained_at_risk.run_id].iloc[0]
    assert row["task"] == "at_risk"
    assert row["model"] == trained_at_risk.training.best_candidate
    assert row["cv_metric"] == "roc_auc"
    assert row["cv_score"] == pytest.approx(trained_at_risk.training.cv_score)
    assert row["metrics.test_roc_auc"] == pytest.approx(trained_at_risk.test_metrics["roc_auc"])
    assert row["metrics.threshold"] == pytest.approx(trained_at_risk.training.threshold)


@needs_mlflow
def test_run_holds_params_artefacts_and_a_loadable_model(trained_at_risk, fast_settings):
    import mlflow

    ExperimentTracker(fast_settings)  # points the fluent API at the test store
    run = mlflow.get_run(trained_at_risk.run_id)
    params = run.data.params
    assert params["cv_folds"] == "3" and params["tune"] == "False"
    assert int(params["n_train"]) + int(params["n_test"]) == 400
    assert any(k.startswith("leaderboard.") for k in run.data.metrics)
    assert run.data.tags["registry_version"] == trained_at_risk.artefact_dir.name

    paths = {a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id)}
    assert {"registry", "figures"} <= paths
    registry_files = {a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id, artifact_path="registry")}
    assert "registry/model_card.md" in registry_files
    assert not any(p.endswith(".joblib") for p in registry_files)

    model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    sample = pd.DataFrame(
        [
            {
                "gender": "female",
                "race_ethnicity": "group B",
                "parental_level_of_education": "high school",
                "lunch": "free/reduced",
                "test_preparation_course": "none",
            }
        ]
    )
    proba = model.predict_proba(sample)[:, 1]
    assert 0.0 <= proba[0] <= 1.0


@needs_mlflow
def test_search_runs_filters_by_task(trained_at_risk, fast_settings):
    assert search_runs("performance_level", fast_settings).empty
    assert not search_runs(None, fast_settings).empty
