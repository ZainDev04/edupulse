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

from edupulse.models.mitigation import GroupThresholds, overall_metrics  # noqa: E402
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
    group_thresholds: GroupThresholds | None = None,
) -> pd.DataFrame:
    """Return one row per (attribute, group) with size-aware metrics.

    For binary tasks ``group_thresholds`` replaces the single ``threshold`` with the
    per-group cut-offs, which is how the mitigated audit is produced.
    """
    attributes = [a for a in (attributes or SENSITIVE_ATTRIBUTES) if a in X_test.columns]
    y = np.asarray(y_test)
    rows: list[dict[str, Any]] = []

    if task.kind == "binary":
        proba = pipeline.predict_proba(X_test)[:, 1]
        if group_thresholds is not None and group_thresholds.attribute in X_test.columns:
            pred = group_thresholds.predict(proba, X_test[group_thresholds.attribute])
        else:
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


def plot_before_after(before: pd.DataFrame, after: pd.DataFrame, attribute: str, path: Path, title: str) -> Path | None:
    """Recall and selection rate per group of ``attribute``, global threshold next to per-group thresholds."""
    b = before[before["attribute"] == attribute].set_index("group")
    a = after[after["attribute"] == attribute].set_index("group")
    if b.empty or a.empty:
        return None
    groups = list(b.index)
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ("tpr", "selection_rate"), ("Recall (TPR)", "Selection rate"), strict=True):
        ax.bar(x - 0.2, b.loc[groups, metric], width=0.4, color="#94A3B8", label="global threshold")
        ax.bar(x + 0.2, a.loc[groups, metric], width=0.4, color="#4F46E5", label="per-group thresholds")
        ax.set_xticks(x, groups)
        ax.set_ylim(0, 1.08)
        ax.set_title(label)
        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
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
    group_thresholds: GroupThresholds | None = None,
) -> dict[str, Any]:
    """Subgroup audit under the global threshold, plus a before/after comparison when
    per-group thresholds are supplied."""
    groups = subgroup_metrics(task, pipeline, X_test, y_test, threshold=threshold)
    fig = plot_subgroups(
        groups, task, figures_dir / task.name / "fairness_subgroups.png", f"{task.name}: subgroup performance"
    )
    out: dict[str, Any] = {
        "groups": groups.to_dict(orient="records"),
        "summary": fairness_summary(groups, task),
        "figure": str(fig) if fig else None,
    }
    if task.kind == "binary" and group_thresholds is not None and group_thresholds.attribute in X_test.columns:
        after = subgroup_metrics(task, pipeline, X_test, y_test, group_thresholds=group_thresholds)
        y = np.asarray(y_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        thr = 0.5 if threshold is None else threshold
        fig2 = plot_before_after(
            groups,
            after,
            group_thresholds.attribute,
            figures_dir / task.name / "fairness_mitigation.png",
            f"{task.name}: recall equalised across {group_thresholds.attribute}",
        )
        out["mitigated"] = {
            "attribute": group_thresholds.attribute,
            "thresholds": group_thresholds.thresholds,
            "groups": after.to_dict(orient="records"),
            "summary": fairness_summary(after, task),
            "overall_before": overall_metrics(y, (proba >= thr).astype(int)),
            "overall_after": overall_metrics(y, group_thresholds.predict(proba, X_test[group_thresholds.attribute])),
            "figure": str(fig2) if fig2 else None,
        }
    return out
