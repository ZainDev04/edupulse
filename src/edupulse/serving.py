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
            "primary_metric": m.task.primary_metric,
        }

    def leaderboard(self, task_name: str) -> list[dict[str, Any]]:
        """Cross-validation leaderboard rows for the latest model of ``task_name``."""
        m = self.model(task_name)
        board = pd.read_csv(m.path / "leaderboard.csv")
        return board.to_dict(orient="records")

    def fairness(self, task_name: str) -> dict[str, Any]:
        """Subgroup metrics and gap summary recorded at training time."""
        f = self.model(task_name).metadata.fairness
        return {"groups": f.get("groups", []), "summary": f.get("summary", {})}

    def importance(self, task_name: str) -> dict[str, Any]:
        """Global SHAP and permutation importance recorded at training time."""
        e = self.model(task_name).metadata.explainability
        return {
            "shap": e.get("shap_importance", []),
            "permutation": e.get("permutation_importance", []),
            "shap_kind": e.get("shap_kind"),
        }

    def dataset_stats(self) -> dict[str, Any]:
        """Headline statistics of the training data plus group breakdowns."""
        df = _full_frame()
        scores = ["math_score", "reading_score", "writing_score"]
        by_attr = {}
        for attr in ("lunch", "test_preparation_course", "parental_level_of_education", "race_ethnicity", "gender"):
            g = df.groupby(attr, observed=True)
            by_attr[attr] = [
                {
                    "group": str(k),
                    "n": int(len(v)),
                    "at_risk_rate": float(v["at_risk"].mean()),
                    "avg_score": float(v["average_score"].mean()),
                }
                for k, v in g
            ]
        return {
            "n_students": int(len(df)),
            "at_risk_rate": float(df["at_risk"].mean()),
            "average_score": float(df["average_score"].mean()),
            "score_means": {c: float(df[c].mean()) for c in scores},
            "performance_level_share": {
                k: float(v) for k, v in df["performance_level"].value_counts(normalize=True).items()
            },
            "by_attribute": by_attr,
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
def _full_frame() -> pd.DataFrame:
    return load_engineered()


@lru_cache(maxsize=1)
def _background_frame() -> pd.DataFrame:
    return _full_frame().sample(200, random_state=0)


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    return PredictionService()
