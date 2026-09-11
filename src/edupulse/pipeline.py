"""End-to-end pipeline: data -> features -> train/tune -> evaluate -> explain -> audit -> register."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from edupulse.config import Settings, get_settings
from edupulse.data.loader import load_clean
from edupulse.features.engineering import DomainRules, engineer_features
from edupulse.logging_utils import get_logger
from edupulse.models.evaluate import evaluate_task, make_figures
from edupulse.models.explain import explain_task
from edupulse.models.fairness import audit_task
from edupulse.models.registry import ModelRegistry, hash_dataframe
from edupulse.models.train import TrainingResult, train_task
from edupulse.tasks import TASKS, Task, get_task

log = get_logger(__name__)


@dataclass
class PipelineOutput:
    task: Task
    training: TrainingResult
    test_metrics: dict[str, Any]
    artefact_dir: Path
    explainability: dict[str, Any]
    fairness: dict[str, Any]

    @property
    def headline(self) -> str:
        m = self.test_metrics
        if self.task.kind == "binary":
            return f"ROC-AUC {m['roc_auc']:.3f} · PR-AUC {m['average_precision']:.3f} · recall {m['recall']:.3f} @ thr {m['threshold']:.2f}"
        if self.task.kind == "regression":
            return f"R² {m['r2']:.3f} · RMSE {m['rmse']:.2f} · MAE {m['mae']:.2f}"
        return f"accuracy {m['accuracy']:.3f} · macro-F1 {m['f1_macro']:.3f}"


def rules_from_settings(settings: Settings) -> DomainRules:
    return DomainRules(
        pass_mark=settings.pass_mark,
        at_risk_threshold=settings.at_risk_threshold,
        medium_cutoff=settings.medium_cutoff,
        high_cutoff=settings.high_cutoff,
    )


def prepare_data(task: Task, settings: Settings | None = None, data_path: str | Path | None = None):
    """Load, engineer and split the data for ``task`` -> ``(X_train, X_test, y_train, y_test, raw)``."""
    settings = settings or get_settings()
    raw = load_clean(data_path)
    engineered = engineer_features(raw, rules_from_settings(settings))
    X, y = task.build_xy(engineered)
    stratify = y if task.is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state, stratify=stratify
    )
    log.info("Task %s: %d train / %d test rows, %d features", task.name, len(X_train), len(X_test), X.shape[1])
    return X_train, X_test, y_train, y_test, raw


def run_pipeline(
    task_name: str,
    *,
    settings: Settings | None = None,
    data_path: str | Path | None = None,
    registry: ModelRegistry | None = None,
    explain: bool = True,
    audit: bool = True,
    version: str | None = None,
) -> PipelineOutput:
    """Train, evaluate and register a model for ``task_name``."""
    settings = settings or get_settings()
    task = get_task(task_name)
    registry = registry or ModelRegistry(settings.models_dir)
    rules = rules_from_settings(settings)

    X_train, X_test, y_train, y_test, raw = prepare_data(task, settings, data_path)
    result = train_task(task, X_train, y_train, settings=settings, rules=rules)

    metrics = evaluate_task(task, result.pipeline, X_test, y_test, threshold=result.threshold)
    log.info(
        "Hold-out metrics for %s: %s", task.name, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
    )

    figures = make_figures(
        task,
        result.pipeline,
        X_test,
        y_test,
        result.leaderboard,
        settings.figures_dir,
        threshold=result.threshold,
        tuning_history=result.tuning_history,
    )
    explainability = (
        explain_task(
            task, result.pipeline, X_train, X_test, y_test, settings.figures_dir, random_state=settings.random_state
        )
        if explain
        else {}
    )
    fairness = (
        audit_task(task, result.pipeline, X_test, y_test, settings.figures_dir, threshold=result.threshold)
        if audit
        else {}
    )
    figures.update(explainability.get("figures", {}))
    if fairness.get("figure"):
        figures["fairness"] = fairness["figure"]

    artefact_dir = registry.save(
        task,
        result.pipeline,
        model_name=result.best_candidate,
        best_params=result.best_params,
        cv_score=result.cv_score,
        test_metrics=metrics,
        threshold=result.threshold,
        encoded_features=result.feature_names,
        n_train=len(X_train),
        n_test=len(X_test),
        data_sha256=hash_dataframe(raw),
        train_seconds=result.train_seconds,
        leaderboard=result.leaderboard,
        explainability={k: v for k, v in explainability.items() if k != "figures"},
        fairness=fairness,
        figures=figures,
        tuning_history=result.tuning_history,
        version=version,
    )

    # Persist a processed snapshot for notebooks / dashboard
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    engineer_features(raw, rules).to_csv(settings.processed_dir / "students_engineered.csv", index=False)

    return PipelineOutput(task, result, metrics, artefact_dir, explainability, fairness)


def run_all(settings: Settings | None = None, **kwargs) -> dict[str, PipelineOutput]:
    """Train every registered task and write a consolidated summary."""
    settings = settings or get_settings()
    outputs = {name: run_pipeline(name, settings=settings, **kwargs) for name in TASKS}
    summary = {
        name: {
            "model": o.training.best_candidate,
            "version": o.artefact_dir.name,
            "cv_metric": o.task.primary_metric,
            "cv_score": o.training.cv_score,
            "headline": o.headline,
            "test_metrics": {k: v for k, v in o.test_metrics.items() if isinstance(v, (int, float))},
        }
        for name, o in outputs.items()
    }
    (settings.reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outputs


def load_engineered(settings: Settings | None = None) -> pd.DataFrame:
    """Convenience loader for notebooks/dashboard (always recomputes from raw for freshness)."""
    settings = settings or get_settings()
    return engineer_features(load_clean(), rules_from_settings(settings))
