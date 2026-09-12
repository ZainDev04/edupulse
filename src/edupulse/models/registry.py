"""Model registry: versioned persistence of pipelines + metadata + model cards.

Layout on disk::

    models/
      at_risk/
        latest.json            -> {"version": "20260912-031500"}
        20260912-031500/
          pipeline.joblib
          metadata.json        (metrics, params, threshold, data hash, ...)
          leaderboard.csv
          model_card.md
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from edupulse import __version__
from edupulse.config import get_settings
from edupulse.logging_utils import get_logger
from edupulse.tasks import Task, get_task

log = get_logger(__name__)


@dataclass
class ModelMetadata:
    task: str
    version: str
    created_at: str
    model_name: str
    model_class: str
    best_params: dict[str, Any]
    cv_metric: str
    cv_score: float
    test_metrics: dict[str, Any]
    threshold: float | None
    features: list[str]
    encoded_features: list[str]
    n_train: int
    n_test: int
    data_sha256: str
    train_seconds: float
    explainability: dict[str, Any] = field(default_factory=dict)
    fairness: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    tracking: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@dataclass
class LoadedModel:
    task: Task
    pipeline: Pipeline
    metadata: ModelMetadata
    path: Path

    @property
    def threshold(self) -> float:
        return self.metadata.threshold if self.metadata.threshold is not None else 0.5


def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "edupulse": __version__,
        "platform": platform.platform(),
    }


class ModelRegistry:
    """Save/load versioned model artefacts."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else get_settings().models_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- save
    def save(
        self,
        task: Task,
        pipeline: Pipeline,
        *,
        model_name: str,
        best_params: dict[str, Any],
        cv_score: float,
        test_metrics: dict[str, Any],
        threshold: float | None,
        encoded_features: list[str],
        n_train: int,
        n_test: int,
        data_sha256: str,
        train_seconds: float,
        leaderboard: pd.DataFrame,
        explainability: dict[str, Any] | None = None,
        fairness: dict[str, Any] | None = None,
        figures: dict[str, str] | None = None,
        tuning_history: pd.DataFrame | None = None,
        tracking: dict[str, str] | None = None,
        version: str | None = None,
    ) -> Path:
        version = version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        d = self.root / task.name / version
        d.mkdir(parents=True, exist_ok=True)

        meta = ModelMetadata(
            task=task.name,
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_name=model_name,
            model_class=type(pipeline.named_steps["model"]).__name__,
            best_params=best_params,
            cv_metric=task.primary_metric,
            cv_score=cv_score,
            test_metrics=test_metrics,
            threshold=threshold,
            features=list(task.features),
            encoded_features=list(encoded_features),
            n_train=n_train,
            n_test=n_test,
            data_sha256=data_sha256,
            train_seconds=train_seconds,
            explainability=explainability or {},
            fairness=fairness or {},
            figures=figures or {},
            environment=_environment(),
            tracking=tracking or {},
        )
        joblib.dump(pipeline, d / "pipeline.joblib", compress=3)
        (d / "metadata.json").write_text(meta.to_json(), encoding="utf-8")
        leaderboard.to_csv(d / "leaderboard.csv", index=False)
        if tuning_history is not None and not tuning_history.empty:
            tuning_history.to_csv(d / "tuning_history.csv", index=False)
        (d / "model_card.md").write_text(render_model_card(task, meta, leaderboard), encoding="utf-8")
        (self.root / task.name / "latest.json").write_text(json.dumps({"version": version}), encoding="utf-8")
        log.info("Saved %s model v%s -> %s", task.name, version, d)
        return d

    # ----------------------------------------------------------------- load
    def versions(self, task_name: str) -> list[str]:
        d = self.root / task_name
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "pipeline.joblib").exists())

    def latest_version(self, task_name: str) -> str | None:
        f = self.root / task_name / "latest.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))["version"]
        v = self.versions(task_name)
        return v[-1] if v else None

    def load(self, task_name: str, version: str | None = None) -> LoadedModel:
        task = get_task(task_name)
        version = version or self.latest_version(task.name)
        if version is None:
            raise FileNotFoundError(
                f"No trained model for task '{task.name}' in {self.root}. Run `edupulse train --task {task.name}`."
            )
        d = self.root / task.name / version
        pipeline = joblib.load(d / "pipeline.joblib")
        meta = ModelMetadata(**json.loads((d / "metadata.json").read_text(encoding="utf-8")))
        return LoadedModel(task=task, pipeline=pipeline, metadata=meta, path=d)

    def available(self) -> dict[str, str | None]:
        return {p.name: self.latest_version(p.name) for p in self.root.iterdir() if p.is_dir()}


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
def _fmt_metric(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def render_model_card(task: Task, meta: ModelMetadata, leaderboard: pd.DataFrame) -> str:
    scalar_metrics = {k: v for k, v in meta.test_metrics.items() if isinstance(v, (int, float))}
    board = leaderboard[["rank", "model", f"{task.primary_metric}_mean", f"{task.primary_metric}_std"]].copy()
    board.columns = ["Rank", "Model", f"CV {task.primary_metric} (mean)", "std"]
    board_md = (
        board.to_markdown(index=False, floatfmt=".4f")
        if hasattr(board, "to_markdown")
        else board.to_string(index=False)
    )

    lines = [
        f"# Model card: `{task.name}` (v{meta.version})",
        "",
        f"**Task type:** {task.kind}  ",
        f"**Description:** {task.description}  ",
        f"**Selected model:** `{meta.model_name}` ({meta.model_class})  ",
        f"**Created:** {meta.created_at}  ",
        f"**Training data hash:** `{meta.data_sha256}` · {meta.n_train} train / {meta.n_test} test rows  ",
    ]
    if meta.tracking.get("run_id"):
        lines.append(f"**MLflow run:** `{meta.tracking['run_id']}` (experiment {meta.tracking.get('experiment_id')})  ")
    lines.append("")
    if task.leakage_note:
        lines += ["> Leakage note: " + task.leakage_note, ""]
    lines += ["## Intended use", "", _intended_use(task), ""]
    lines += ["## Inputs", "", *[f"- `{f}`" for f in meta.features], ""]
    lines += ["## Hold-out performance", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt_metric(v)} |" for k, v in scalar_metrics.items()]
    lines += ["", f"Cross-validated {task.primary_metric} of the selected configuration: **{meta.cv_score:.4f}**", ""]
    if meta.threshold is not None:
        lines += [
            f"Decision threshold: **{meta.threshold:.3f}** (highest threshold that keeps out-of-fold recall ≥ target recall).",
            "",
        ]
    lines += ["## Model leaderboard (repeated stratified CV)", "", board_md, ""]
    if meta.best_params:
        lines += [
            "## Tuned hyper-parameters (Optuna)",
            "",
            "```json",
            json.dumps(meta.best_params, indent=2),
            "```",
            "",
        ]
    if meta.explainability.get("shap_importance"):
        lines += ["## Top features (mean |SHAP|)", "", "| Feature | mean abs SHAP |", "|---|---|"]
        lines += [f"| {r['feature']} | {r['mean_abs_shap']:.4f} |" for r in meta.explainability["shap_importance"][:10]]
        lines += [""]
    if meta.fairness.get("summary"):
        lines += ["## Fairness audit (hold-out subgroup gaps)", "", "| Attribute | Metric | Gap |", "|---|---|---|"]
        for attr, gaps in meta.fairness["summary"].items():
            for m, g in gaps.items():
                lines.append(f"| {attr} | {m} | {g:.3f} |")
        lines += [""]
    lines += [
        "## Limitations",
        "",
        "- Trained on 1,000 anonymised records from one cohort. Results may not transfer to other institutions.",
        "- Background attributes are proxies, not causes. Predictions are for prioritising support and must never be used to penalise a student.",
        "- Sensitive attributes are model inputs. The fairness section measures subgroup disparities; it does not remove them.",
        "",
        "## Environment",
        "",
        *[f"- {k}: {v}" for k, v in meta.environment.items()],
        "",
    ]
    return "\n".join(lines)


def _intended_use(task: Task) -> str:
    return {
        "at_risk": (
            "Rank incoming students by probability of under-performing so that academic advisors can "
            "offer test preparation or tutoring before exams. The model is a triage tool. It is not a verdict on the student."
        ),
        "math_score": (
            "Estimate an expected math score from the reading and writing scores and background. Useful for "
            "checking data entry, filling in a missing exam, or spotting a student whose math result is far below expectation."
        ),
        "performance_level": (
            "Reproduce the original coursework tiering task. Its main use is as a demonstration of "
            "target leakage in model evaluation."
        ),
    }.get(task.name, task.description)
