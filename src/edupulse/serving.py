"""Framework-agnostic prediction service used by the CLI, REST API and dashboard.

Loads registered pipelines lazily (one per task), validates incoming rows and
returns JSON-ready dictionaries with calibrated probabilities, risk bands and
optional SHAP explanations.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edupulse.data.schema import SCORE_COLUMNS, validate_dataframe
from edupulse.logging_utils import get_logger
from edupulse.models.explain import Explainer
from edupulse.models.registry import LoadedModel, ModelRegistry
from edupulse.pipeline import load_engineered
from edupulse.tasks import TASKS, get_task

log = get_logger(__name__)

RISK_BANDS = [(0.0, "low"), (0.35, "moderate"), (0.6, "high"), (0.8, "critical")]


def risk_band(p: float) -> str:
    band = RISK_BANDS[0][1]
    for lo, name in RISK_BANDS:
        if p >= lo:
            band = name
    return band


class PredictionService:
    """Thread-safe-enough (read-only) façade over the model registry."""

    def __init__(self, models_dir: Path | None = None):
        self.registry = ModelRegistry(models_dir)
        self._models: dict[str, LoadedModel] = {}
        self._explainers: dict[str, Explainer] = {}

    # ------------------------------------------------------------------ #
    def available_tasks(self) -> list[str]:
        return [t for t in TASKS if self.registry.latest_version(t) is not None]

    def model(self, task_name: str) -> LoadedModel:
        name = get_task(task_name).name
        if name not in self._models:
            self._models[name] = self.registry.load(name)
            log.info("Loaded %s v%s", name, self._models[name].metadata.version)
        return self._models[name]

    def explainer(self, task_name: str) -> Explainer:
        name = get_task(task_name).name
        if name not in self._explainers:
            m = self.model(name)
            background = _background_frame()[m.task.features]
            self._explainers[name] = Explainer(m.pipeline, m.task, background, max_background=100)
        return self._explainers[name]

    def info(self, task_name: str) -> dict[str, Any]:
        m = self.model(task_name)
        md = m.metadata
        return {
            "task": m.task.name,
            "kind": m.task.kind,
            "description": m.task.description,
            "version": md.version,
            "model": md.model_name,
            "model_class": md.model_class,
            "features": md.features,
            "threshold": md.threshold,
            "cv_metric": md.cv_metric,
            "cv_score": md.cv_score,
            "test_metrics": {k: v for k, v in md.test_metrics.items() if isinstance(v, (int, float))},
            "created_at": md.created_at,
            "leakage_note": m.task.leakage_note,
        }

    # ------------------------------------------------------------------ #
    def _frame(self, task_name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
        m = self.model(task_name)
        df = pd.DataFrame(rows).dropna(axis=1, how="all")  # optional fields sent as null
        missing = [f for f in m.task.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required fields for task '{m.task.name}': {missing}")
        for c in df.columns:
            if c in SCORE_COLUMNS:
                df[c] = pd.to_numeric(df[c], errors="raise")
            else:
                df[c] = df[c].astype(str).str.strip()
        validate_dataframe(df, require_scores=False)
        return df[m.task.features]

    def predict(
        self, task_name: str, rows: list[dict[str, Any]], *, explain: bool = False, top_k: int = 5
    ) -> list[dict[str, Any]]:
        m = self.model(task_name)
        X = self._frame(task_name, rows)
        out: list[dict[str, Any]] = []

        if m.task.kind == "binary":
            proba = m.pipeline.predict_proba(X)[:, 1]
            for p in proba:
                out.append(
                    {
                        "probability": float(p),
                        "at_risk": bool(p >= m.threshold),
                        "threshold": float(m.threshold),
                        "risk_band": risk_band(float(p)),
                    }
                )
        elif m.task.kind == "multiclass":
            proba = m.pipeline.predict_proba(X)
            for row in proba:
                idx = int(np.argmax(row))
                out.append(
                    {
                        "label": m.task.classes[idx],
                        "probabilities": {c: float(v) for c, v in zip(m.task.classes, row, strict=False)},
                    }
                )
        else:
            pred = m.pipeline.predict(X)
            for v in pred:
                out.append({"prediction": float(np.clip(v, 0, 100))})

        if explain:
            ex = self.explainer(task_name)
            for i, rec in enumerate(out):
                rec["explanation"] = ex.local_explanation(X.iloc[[i]], top_k=top_k)
        for rec in out:
            rec["model_version"] = m.metadata.version
        return out


@lru_cache(maxsize=1)
def _background_frame() -> pd.DataFrame:
    return load_engineered().sample(200, random_state=0)


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    return PredictionService()
