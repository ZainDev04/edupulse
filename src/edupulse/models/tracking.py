"""MLflow experiment tracking, layered behind the model registry.

The registry on disk stays the source of truth for serving. MLflow records
the history of every training run (params, CV and hold-out metrics, the
Optuna trial curve, figures, model card, and the fitted pipeline) so that
runs can be compared in the MLflow UI::

    edupulse ui        # opens http://localhost:5000 on ./mlruns

Runs are stored in a SQLite database (``mlruns/mlflow.db``) with artefacts
under ``mlruns/artifacts``; point ``EDUPULSE_TRACKING_URI`` at a remote
server to share them.

Tracking is optional. If ``mlflow`` is not installed, or
``EDUPULSE_TRACKING=false``, :class:`ExperimentTracker` turns into a no-op
and the pipeline behaves exactly as before.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from edupulse.config import Settings, get_settings
from edupulse.logging_utils import get_logger
from edupulse.tasks import Task

log = get_logger(__name__)

_SETTING_PARAMS = (
    "random_state",
    "test_size",
    "cv_folds",
    "cv_repeats",
    "tune",
    "n_trials",
    "target_recall",
    "at_risk_threshold",
    "pass_mark",
)


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def _scalars(d: dict[str, Any]) -> dict[str, float]:
    """Keep the numeric, finite entries of ``d`` (MLflow metrics must be floats)."""
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if math.isnan(v) or math.isinf(v):
            continue
        out[k] = float(v)
    return out


class ExperimentTracker:
    """Thin wrapper over the MLflow fluent API with a no-op mode.

    Usage inside the pipeline::

        tracker = ExperimentTracker(settings)
        with tracker.run(task, version):
            ...
            tracker.log_training(result)
            tracker.log_metrics(test_metrics)
            tracker.log_artefact_dir(artefact_dir)
        tracker.info  # -> {"run_id": ..., "experiment_id": ..., "tracking_uri": ...}
    """

    def __init__(self, settings: Settings | None = None, enabled: bool | None = None):
        self.settings = settings or get_settings()
        wanted = self.settings.tracking if enabled is None else enabled
        self.enabled = bool(wanted and mlflow_available())
        if wanted and not self.enabled:
            log.warning("MLflow tracking requested but mlflow is not installed; continuing without it.")
        self.info: dict[str, str] = {}
        self._mlflow = None
        if self.enabled:
            import mlflow

            self._mlflow = mlflow
            self._connect()

    @property
    def tracking_uri(self) -> str:
        """The backend store URI. A bare path is treated as a SQLite database file."""
        uri = self.settings.tracking_uri
        if "://" in uri:
            return uri
        return "sqlite:///" + Path(uri).resolve().as_posix()

    def _connect(self) -> None:
        os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
        uri = self.tracking_uri
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self._mlflow.set_tracking_uri(uri)
        name = self.settings.tracking_experiment
        if self._mlflow.get_experiment_by_name(name) is None:
            artifacts = Path(self.settings.tracking_artifacts).resolve()
            artifacts.mkdir(parents=True, exist_ok=True)
            self._mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
        self._mlflow.set_experiment(name)

    # ------------------------------------------------------------------ run
    @contextmanager
    def run(self, task: Task, version: str | None = None, tags: dict[str, str] | None = None):
        """Open an MLflow run for ``task``; yields ``self``."""
        if not self.enabled:
            yield self
            return
        from edupulse import __version__

        name = f"{task.name}-{version}" if version else task.name
        run_tags = {"task": task.name, "task_kind": task.kind, "edupulse_version": __version__, **(tags or {})}
        with self._mlflow.start_run(run_name=name, tags=run_tags) as active:
            self.info = {
                "run_id": active.info.run_id,
                "experiment_id": active.info.experiment_id,
                "tracking_uri": self.tracking_uri,
            }
            self._mlflow.log_params({k: getattr(self.settings, k) for k in _SETTING_PARAMS})
            yield self

    # ------------------------------------------------------------- logging
    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled and params:
            self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        if self.enabled:
            scalars = _scalars(metrics)
            if scalars:
                self._mlflow.log_metrics({f"{prefix}{k}": v for k, v in scalars.items()})

    def set_tags(self, tags: dict[str, str]) -> None:
        if self.enabled and tags:
            self._mlflow.set_tags(tags)

    def log_training(self, result, *, task: Task) -> None:
        """Log the tuned params, the CV leaderboard and the Optuna trial curve from a ``TrainingResult``."""
        if not self.enabled:
            return
        self.set_tags({"model": result.best_candidate, "model_class": type(result.pipeline[-1]).__name__})
        self.log_params({f"model.{k}": v for k, v in result.best_params.items()})
        self.log_metrics({f"cv_{task.primary_metric}": result.cv_score, "train_seconds": result.train_seconds})
        if result.threshold is not None:
            self.log_metrics({"threshold": result.threshold})

        # one metric per zoo candidate so the leaderboard is comparable across runs in the UI
        board: pd.DataFrame = result.leaderboard
        col = f"{task.primary_metric}_mean"
        if col in board:
            self.log_metrics({f"leaderboard.{row.model}": row[col] for _, row in board.iterrows()})

        # the Optuna objective per trial, as a step series
        hist = result.tuning_history
        if hist is not None and not hist.empty and "value" in hist:
            for step, value in enumerate(hist["value"].tolist()):
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    self._mlflow.log_metric("tuning_objective", float(value), step=step)

    def log_artefact_dir(self, path: Path, artifact_path: str = "registry") -> None:
        """Copy the registry files (model card, metadata, CSVs) into the run.

        The joblib pipeline is skipped: :meth:`log_model` stores it in MLflow's own model format.
        """
        if not self.enabled:
            return
        for f in sorted(Path(path).glob("*")):
            if f.is_file() and f.suffix != ".joblib":
                self._mlflow.log_artifact(str(f), artifact_path=artifact_path)

    def log_figures(self, figures: dict[str, str]) -> None:
        if not self.enabled:
            return
        for p in figures.values():
            if Path(p).exists():
                self._mlflow.log_artifact(str(p), artifact_path="figures")

    def log_model(self, pipeline, input_example: pd.DataFrame | None = None) -> None:
        """Log the fitted sklearn pipeline in MLflow's model format.

        cloudpickle is used because the pipeline contains project classes
        (``FeatureEngineer``) that the default skops format refuses to load.
        """
        if not self.enabled:
            return
        import inspect

        import mlflow.sklearn

        kwargs: dict[str, Any] = {"serialization_format": "cloudpickle"}
        if input_example is not None:
            example = input_example.head(5).copy()
            # integer columns cannot hold NaN; declare them as floats so the signature accepts missing scores
            int_cols = example.select_dtypes("integer").columns
            kwargs["input_example"] = example.astype(dict.fromkeys(int_cols, "float64"))
        # MLflow 3 renamed ``artifact_path`` to ``name``
        if "name" in inspect.signature(mlflow.sklearn.log_model).parameters:
            kwargs["name"] = "model"
        else:  # pragma: no cover
            kwargs["artifact_path"] = "model"
        sk_log = logging.getLogger("mlflow.sklearn")
        level = sk_log.level
        sk_log.setLevel(logging.ERROR)  # silence the pickle caution; the registry already stores joblib pickles
        try:
            with warnings.catch_warnings():
                # classifiers predict int labels; MLflow's hint about NaN in integer outputs does not apply
                warnings.filterwarnings(
                    "ignore", message="Hint: Inferred schema contains integer", category=UserWarning
                )
                mlflow.sklearn.log_model(pipeline, **kwargs)
        finally:
            sk_log.setLevel(level)


# --------------------------------------------------------------------------- #
# Querying
# --------------------------------------------------------------------------- #
def search_runs(task_name: str | None = None, settings: Settings | None = None, max_results: int = 50) -> pd.DataFrame:
    """Return recent runs (newest first) as a tidy frame for the CLI and API."""
    settings = settings or get_settings()
    if not mlflow_available():
        return pd.DataFrame()
    import mlflow

    tracker = ExperimentTracker(settings, enabled=True)
    exp = mlflow.get_experiment_by_name(settings.tracking_experiment)
    if exp is None:  # pragma: no cover - the tracker creates it on connect
        return pd.DataFrame()
    filt = f"tags.task = '{task_name}'" if task_name else ""
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filt,
        max_results=max_results,
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        return runs
    keep = ["run_id", "start_time", "status", "tags.mlflow.runName", "tags.task", "tags.model"]
    keep += [c for c in runs.columns if c.startswith("metrics.") and not c.startswith("metrics.leaderboard.")]
    keep = [c for c in keep if c in runs.columns]
    out = runs[keep].rename(columns=lambda c: c.replace("tags.mlflow.runName", "run_name").replace("tags.", ""))

    # each task has its own primary CV metric; surface it under one name so mixed-task listings line up
    cv_cols = [c for c in out.columns if c.startswith("metrics.cv_")]
    if cv_cols:
        cv = out[cv_cols]
        out["cv_metric"] = cv.notna().idxmax(axis=1).str.replace("metrics.cv_", "", regex=False)
        out["cv_score"] = cv.bfill(axis=1).iloc[:, 0]
    out["tracking_uri"] = tracker.tracking_uri
    return out
