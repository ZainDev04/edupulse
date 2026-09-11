"""Fairness / subgroup audit.

For an early-warning system it matters *who* the model fails on.  This module
slices hold-out performance by sensitive attributes (gender, ethnicity, lunch
programme - a proxy for socio-economic status) and reports common group
fairness metrics:

- **Selection rate** (share flagged) and *demographic parity difference*
- **True-positive rate** (recall) and *equal-opportunity difference*
- **False-positive rate** and *predictive parity* (precision)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from edupulse.tasks import Task  # noqa: E402

SENSITIVE_ATTRIBUTES = ["gender", "race_ethnicity", "lunch", "parental_level_of_education"]


def _safe_auc(y, p) -> float | None:
    try:
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    except ValueError:
        return None


def subgroup_metrics(
    task: Task,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float | None = None,
    attributes: list[str] | None = None,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Return one row per (attribute, group) with size-aware metrics."""
    attributes = [a for a in (attributes or SENSITIVE_ATTRIBUTES) if a in X_test.columns]
    y = np.asarray(y_test)
    rows: list[dict[str, Any]] = []

    if task.kind == "binary":
        proba = pipeline.predict_proba(X_test)[:, 1]
        thr = 0.5 if threshold is None else threshold
        pred = (proba >= thr).astype(int)
    elif task.kind == "regression":
        pred = pipeline.predict(X_test)
        proba = None
    else:
        pred = pipeline.predict(X_test)
        proba = None

    for attr in attributes:
        for group, idx in X_test.groupby(attr, observed=True).indices.items():
            if len(idx) < min_group_size:
                continue
            yg, pg = y[idx], pred[idx]
            row: dict[str, Any] = {"attribute": attr, "group": str(group), "n": int(len(idx))}
            if task.kind == "binary":
                row.update(
                    prevalence=float(yg.mean()),
                    selection_rate=float(pg.mean()),
                    tpr=float(recall_score(yg, pg, zero_division=0)),
                    fpr=float(((pg == 1) & (yg == 0)).sum() / max((yg == 0).sum(), 1)),
                    precision=float(precision_score(yg, pg, zero_division=0)),
                    roc_auc=_safe_auc(yg, proba[idx]),
                )
            elif task.kind == "regression":
                row.update(mae=float(mean_absolute_error(yg, pg)), mean_residual=float(np.mean(yg - pg)))
            else:
                row.update(accuracy=float(np.mean(yg == pg)))
            rows.append(row)
    return pd.DataFrame(rows)


def fairness_summary(groups: pd.DataFrame, task: Task) -> dict[str, Any]:
    """Max-minus-min gaps per attribute for the headline fairness metrics."""
    if groups.empty:
        return {}
    metric_cols = {
        "binary": ["selection_rate", "tpr", "fpr", "precision"],
        "regression": ["mae"],
        "multiclass": ["accuracy"],
    }[task.kind]
    summary: dict[str, Any] = {}
    for attr, g in groups.groupby("attribute"):
        summary[attr] = {f"{m}_gap": float(g[m].max() - g[m].min()) for m in metric_cols if m in g}
        if task.kind == "binary" and "selection_rate" in g and g["selection_rate"].max() > 0:
            summary[attr]["disparate_impact_ratio"] = float(g["selection_rate"].min() / g["selection_rate"].max())
    return summary


def plot_subgroups(groups: pd.DataFrame, task: Task, path: Path, title: str) -> Path | None:
    if groups.empty:
        return None
    metric = {"binary": "tpr", "regression": "mae", "multiclass": "accuracy"}[task.kind]
    attrs = groups["attribute"].unique()
    fig, axes = plt.subplots(1, len(attrs), figsize=(5.5 * len(attrs), 5), squeeze=False)
    for ax, attr in zip(axes[0], attrs, strict=False):
        g = groups[groups["attribute"] == attr]
        ax.bar(g["group"], g[metric], color="#4F46E5")
        ax.set_title(attr.replace("_", " "))
        ax.set_ylabel(metric.upper() if len(metric) <= 3 else metric)
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def audit_task(
    task: Task,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    groups = subgroup_metrics(task, pipeline, X_test, y_test, threshold=threshold)
    fig = plot_subgroups(
        groups, task, figures_dir / task.name / "fairness_subgroups.png", f"{task.name}: subgroup performance"
    )
    return {
        "groups": groups.to_dict(orient="records"),
        "summary": fairness_summary(groups, task),
        "figure": str(fig) if fig else None,
    }
