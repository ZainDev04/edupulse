"""Task registry.

A *task* bundles everything that differs between prediction problems: which
columns are legal inputs, how the target is built, and which metric drives
model selection.  Adding a new problem is a matter of registering a new
:class:`Task` here - the training pipeline, API and dashboard pick it up
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from edupulse.data.schema import CATEGORICAL_COLUMNS
from edupulse.features.engineering import PERFORMANCE_LEVELS

TaskKind = Literal["binary", "multiclass", "regression"]

BACKGROUND_FEATURES = list(CATEGORICAL_COLUMNS)


@dataclass(frozen=True)
class Task:
    name: str
    kind: TaskKind
    target: str
    features: list[str]
    primary_metric: str
    description: str
    classes: tuple[str, ...] | None = None
    positive_label: int | None = None
    leakage_note: str | None = None
    extra_metrics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_classification(self) -> bool:
        return self.kind in ("binary", "multiclass")

    @property
    def slug(self) -> str:
        return self.name.replace("_", "-")

    def build_xy(self, engineered: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split an *engineered* frame into ``(X, y)`` for this task."""
        X = engineered[self.features].copy()
        y = engineered[self.target].copy()
        if self.kind == "multiclass" and self.classes:
            codes = pd.Categorical(y, categories=self.classes, ordered=True).codes
            y = pd.Series(codes, index=X.index, name=self.target)
        return X, y

    def decode(self, y_codes) -> list:
        """Map integer class codes back to labels (classification only)."""
        if self.kind == "multiclass" and self.classes:
            return [self.classes[int(c)] for c in y_codes]
        if self.kind == "binary":
            return [int(c) for c in y_codes]
        return [float(v) for v in y_codes]


TASKS: dict[str, Task] = {
    "at_risk": Task(
        name="at_risk",
        kind="binary",
        target="at_risk",
        features=BACKGROUND_FEATURES,
        primary_metric="roc_auc",
        extra_metrics=("average_precision", "f1", "recall", "precision", "balanced_accuracy"),
        positive_label=1,
        description=(
            "Early-warning system: flag students likely to average below 60 using ONLY "
            "background information available before any exam is taken."
        ),
    ),
    "math_score": Task(
        name="math_score",
        kind="regression",
        target="math_score",
        features=BACKGROUND_FEATURES + ["reading_score", "writing_score"],
        primary_metric="r2",
        extra_metrics=("neg_root_mean_squared_error", "neg_mean_absolute_error"),
        description=(
            "Cross-subject score prediction: estimate a student's math score from their "
            "background and literacy (reading/writing) scores."
        ),
    ),
    "performance_level": Task(
        name="performance_level",
        kind="multiclass",
        target="performance_level",
        features=BACKGROUND_FEATURES + ["math_score", "reading_score", "writing_score"],
        primary_metric="f1_macro",
        extra_metrics=("accuracy", "balanced_accuracy"),
        classes=PERFORMANCE_LEVELS,
        description=(
            "Performance tiering (low, medium, high) from background and subject scores: "
            "the original coursework formulation."
        ),
        leakage_note=(
            "The target is a deterministic function of the three score inputs "
            "(average < 60 -> low, < 80 -> medium, else high). Near-perfect accuracy is "
            "therefore expected and says nothing about generalisation. The task is kept for "
            "continuity with the original report and as a leakage case study. See the "
            "'at_risk' task for the background-only formulation."
        ),
    ),
}


def get_task(name: str) -> Task:
    """Look up a task by name (``at_risk`` and ``at-risk`` both work)."""
    try:
        return TASKS[name.replace("-", "_")]
    except KeyError as exc:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASKS)}") from exc
