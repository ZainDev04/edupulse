"""Explainability: SHAP values + permutation importance.

SHAP values are computed on the *pre-processed* matrix (one-hot columns) and
then aggregated back to the original human-readable feature names so that
"race_ethnicity" appears as one bar instead of five.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from edupulse.logging_utils import get_logger  # noqa: E402
from edupulse.tasks import Task  # noqa: E402

log = get_logger(__name__)
_TREE_TYPES = (
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "XGB",
    "LGBM",
    "DecisionTree",
)


def _is_tree(model) -> bool:
    return any(type(model).__name__.startswith(t) for t in _TREE_TYPES)


def _transform(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    Z = pipeline[:-1].transform(X)
    if not isinstance(Z, pd.DataFrame):
        Z = pd.DataFrame(Z, columns=pipeline.named_steps["preprocess"].get_feature_names_out())
    return Z


def group_map(encoded_columns: list[str], original_features: list[str]) -> dict[str, str]:
    """Map each encoded column to the original feature it came from."""
    mapping = {}
    for col in encoded_columns:
        parent = next(
            (f for f in sorted(original_features, key=len, reverse=True) if col == f or col.startswith(f + "_")), col
        )
        mapping[col] = parent
    return mapping


class Explainer:
    """Compute SHAP values for a fitted EduPulse pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        task: Task,
        background: pd.DataFrame,
        *,
        max_background: int = 200,
        random_state: int = 42,
    ):
        self.pipeline = pipeline
        self.task = task
        self.model = pipeline.named_steps["model"]
        Zb = _transform(pipeline, background)
        if len(Zb) > max_background:
            Zb = Zb.sample(max_background, random_state=random_state)
        self.background = Zb
        self.encoded_columns = list(Zb.columns)
        self.mapping = group_map(self.encoded_columns, list(task.features))

        if _is_tree(self.model):
            self.kind = "tree"
            self._explainer = shap.TreeExplainer(self.model)
        elif hasattr(self.model, "coef_") and task.kind != "multiclass":
            self.kind = "linear"
            self._explainer = shap.LinearExplainer(self.model, Zb)
        else:
            self.kind = "kernel"
            f = self.model.predict_proba if task.is_classification else self.model.predict
            self._explainer = shap.KernelExplainer(f, shap.sample(Zb, min(50, len(Zb)), random_state=random_state))

    # ------------------------------------------------------------------ #
    def shap_values(self, X: pd.DataFrame, class_index: int | None = None) -> np.ndarray:
        """Return SHAP values as ``(n_samples, n_encoded_features)``.

        For classification the values of one class are returned (``class_index``
        defaults to the positive class for binary tasks and to *all classes*
        stacked as ``(n, f, c)`` for multi-class tasks when ``class_index`` is
        ``None``).
        """
        Z = _transform(self.pipeline, X)
        if self.kind == "tree":
            sv = self._explainer.shap_values(Z, check_additivity=False)
        elif self.kind == "linear":
            sv = self._explainer.shap_values(Z)
        else:
            sv = self._explainer.shap_values(Z, nsamples=200, silent=True)
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[0] != len(Z):  # legacy (c, n, f) -> (n, f, c)
            sv = np.moveaxis(sv, 0, -1)
        if sv.ndim == 3:
            if self.task.kind == "binary":
                sv = sv[:, :, 1 if class_index is None else class_index]
            elif class_index is not None:
                sv = sv[:, :, class_index]
        return sv

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| aggregated to original feature names."""
        sv = self.shap_values(X)
        abs_mean = np.abs(sv).mean(axis=0)
        if abs_mean.ndim == 2:  # multiclass: average over classes
            abs_mean = abs_mean.mean(axis=1)
        enc = pd.Series(abs_mean, index=self.encoded_columns)
        grouped = enc.groupby(enc.index.map(self.mapping)).sum().sort_values(ascending=False)
        return grouped.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})

    def local_explanation(self, X_row: pd.DataFrame, top_k: int = 10) -> list[dict[str, Any]]:
        """Per-feature contribution for a single row (aggregated, signed)."""
        if self.task.kind == "multiclass":
            cls = int(self.pipeline.predict(X_row)[0])
            sv = self.shap_values(X_row, class_index=cls)[0]
        else:
            sv = self.shap_values(X_row)[0]
        contrib = pd.Series(sv, index=self.encoded_columns)
        grouped = contrib.groupby(contrib.index.map(self.mapping)).sum()
        rows = []
        for feat, val in grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(top_k).items():
            rows.append({"feature": feat, "value": _fmt(X_row.iloc[0].get(feat)), "contribution": float(val)})
        return rows


def _fmt(v):
    if v is None:
        return None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return str(v)


# --------------------------------------------------------------------------- #
# Permutation importance (model-agnostic, on raw features)
# --------------------------------------------------------------------------- #
def permutation_report(
    pipeline: Pipeline, task: Task, X: pd.DataFrame, y: pd.Series, *, n_repeats: int = 10, random_state: int = 42
) -> pd.DataFrame:
    scoring = {"binary": "roc_auc", "multiclass": "f1_macro", "regression": "r2"}[task.kind]
    r = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    return (
        pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_importance(df: pd.DataFrame, value_col: str, path: Path, title: str, err_col: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 2))
    d = df.sort_values(value_col)
    ax.barh(d["feature"], d[value_col], xerr=d[err_col] if err_col else None, color="#4F46E5", capsize=4)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " "))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_beeswarm(explainer: Explainer, X: pd.DataFrame, path: Path, title: str, max_display: int = 15) -> Path:
    Z = _transform(explainer.pipeline, X)
    sv = explainer.shap_values(X, class_index=None if explainer.task.kind != "multiclass" else 0)
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        sv, Z, feature_names=explainer.encoded_columns, max_display=max_display, show=False, plot_size=None
    )
    plt.title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def explain_task(
    task: Task,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the full explainability suite and return a JSON-serialisable summary."""
    d = figures_dir / task.name
    out: dict[str, Any] = {}

    perm = permutation_report(pipeline, task, X_test, y_test, random_state=random_state)
    out["permutation_importance"] = perm.to_dict(orient="records")
    out["figures"] = {
        "permutation": str(
            plot_importance(
                perm,
                "importance_mean",
                d / "permutation_importance.png",
                f"{task.name}: permutation importance (hold-out)",
                "importance_std",
            )
        )
    }

    try:
        expl = Explainer(pipeline, task, X_train, random_state=random_state)
        sample = X_test if len(X_test) <= 300 else X_test.sample(300, random_state=random_state)
        gi = expl.global_importance(sample)
        out["shap_importance"] = gi.to_dict(orient="records")
        out["shap_kind"] = expl.kind
        out["figures"]["shap_bar"] = str(
            plot_importance(gi, "mean_abs_shap", d / "shap_importance.png", f"{task.name}: mean |SHAP| by feature")
        )
        out["figures"]["shap_beeswarm"] = str(
            plot_shap_beeswarm(expl, sample, d / "shap_beeswarm.png", f"{task.name}: SHAP beeswarm")
        )
    except Exception as exc:  # pragma: no cover - SHAP is best-effort
        log.warning("SHAP explanation skipped: %s", exc)
        out["shap_error"] = str(exc)
    return out
