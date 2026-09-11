"""Hold-out evaluation: metrics + publication-quality figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from edupulse.tasks import Task  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", palette="deep")
PALETTE = {"primary": "#4F46E5", "accent": "#F59E0B", "muted": "#94A3B8", "good": "#10B981", "bad": "#EF4444"}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate_task(
    task: Task,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute a task-appropriate metric dictionary on the hold-out set."""
    y_true = np.asarray(y_test)
    out: dict[str, Any] = {"n_test": int(len(y_true))}

    if task.kind == "regression":
        pred = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
        out.update(
            r2=float(r2_score(y_true, pred)),
            rmse=rmse,
            mae=float(mean_absolute_error(y_true, pred)),
            mape=float(np.mean(np.abs((y_true - pred) / np.clip(y_true, 1, None))) * 100),
            residual_std=float(np.std(y_true - pred)),
        )
        return out

    proba = pipeline.predict_proba(X_test)
    if task.kind == "binary":
        p1 = proba[:, 1]
        thr = 0.5 if threshold is None else threshold
        pred = (p1 >= thr).astype(int)
        out.update(
            threshold=float(thr),
            roc_auc=float(roc_auc_score(y_true, p1)),
            average_precision=float(average_precision_score(y_true, p1)),
            brier=float(brier_score_loss(y_true, p1)),
            log_loss=float(log_loss(y_true, p1)),
            accuracy=float(accuracy_score(y_true, pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
            precision=float(precision_score(y_true, pred, zero_division=0)),
            recall=float(recall_score(y_true, pred, zero_division=0)),
            f1=float(f1_score(y_true, pred, zero_division=0)),
            f2=float(
                (5 * precision_score(y_true, pred, zero_division=0) * recall_score(y_true, pred, zero_division=0))
                / max(
                    4 * precision_score(y_true, pred, zero_division=0) + recall_score(y_true, pred, zero_division=0),
                    1e-12,
                )
            ),
            positive_rate=float(np.mean(y_true)),
            flagged_rate=float(np.mean(pred)),
        )
    else:
        pred = np.argmax(proba, axis=1)
        out.update(
            accuracy=float(accuracy_score(y_true, pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
            f1_macro=float(f1_score(y_true, pred, average="macro")),
            f1_weighted=float(f1_score(y_true, pred, average="weighted")),
            roc_auc_ovr=float(roc_auc_score(y_true, proba, multi_class="ovr")),
            log_loss=float(log_loss(y_true, proba, labels=list(range(proba.shape[1])))),
        )
    labels = list(task.classes) if task.classes else ["0", "1"]
    out["confusion_matrix"] = confusion_matrix(y_true, pred).tolist()
    out["classification_report"] = classification_report(
        y_true, pred, target_names=labels, output_dict=True, zero_division=0
    )
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_leaderboard(board: pd.DataFrame, metric: str, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(board) + 2))
    b = board.sort_values(f"{metric}_mean")
    colors = [PALETTE["muted"] if f == "dummy" else PALETTE["primary"] for f in b["family"]]
    ax.barh(b["model"], b[f"{metric}_mean"], xerr=b[f"{metric}_std"], color=colors, capsize=4)
    for i, v in enumerate(b[f"{metric}_mean"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=11)
    ax.set_xlabel(f"CV {metric} (mean ± std)")
    ax.set_title(title)
    lo = max(0.0, b[f"{metric}_mean"].min() - 0.1) if b[f"{metric}_mean"].min() > 0 else None
    if lo is not None:
        ax.set_xlim(left=lo)
    return _save(fig, path)


def plot_confusion(y_true, y_pred, labels: list[str], path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    ax.grid(False)
    return _save(fig, path)


def plot_roc_pr(y_true, proba, path: Path, title: str) -> Path:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(fpr, tpr, color=PALETTE["primary"], lw=2.5, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["muted"])
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, color=PALETTE["accent"], lw=2.5, label=f"AP = {average_precision_score(y_true, proba):.3f}")
    axes[1].axhline(np.mean(y_true), ls="--", color=PALETTE["muted"], label=f"prevalence = {np.mean(y_true):.2f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    axes[1].legend(loc="upper right")
    fig.suptitle(title)
    return _save(fig, path)


def plot_calibration(y_true, proba, path: Path, title: str) -> Path:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(mean_pred, frac_pos, "o-", color=PALETTE["primary"], lw=2.5, label="model")
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["muted"], label="perfect")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Reliability diagram")
    axes[0].legend()
    axes[1].hist(proba, bins=25, color=PALETTE["accent"], alpha=0.85)
    axes[1].set(xlabel="Predicted probability", ylabel="Students", title="Score distribution")
    fig.suptitle(title)
    return _save(fig, path)


def plot_threshold_sweep(y_true, proba, threshold: float, path: Path, title: str) -> Path:
    ts = np.linspace(0.05, 0.95, 91)
    P = [precision_score(y_true, proba >= t, zero_division=0) for t in ts]
    R = [recall_score(y_true, proba >= t, zero_division=0) for t in ts]
    F = [f1_score(y_true, proba >= t, zero_division=0) for t in ts]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ts, P, label="precision", color=PALETTE["primary"], lw=2.5)
    ax.plot(ts, R, label="recall", color=PALETTE["accent"], lw=2.5)
    ax.plot(ts, F, label="F1", color=PALETTE["good"], lw=2.5)
    ax.axvline(threshold, ls="--", color=PALETTE["bad"], label=f"chosen threshold = {threshold:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title=title)
    ax.legend()
    return _save(fig, path)


def plot_regression_diagnostics(y_true, pred, path: Path, title: str) -> Path:
    resid = np.asarray(y_true) - np.asarray(pred)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].scatter(y_true, pred, alpha=0.6, color=PALETTE["primary"], s=28)
    lim = [min(np.min(y_true), np.min(pred)), max(np.max(y_true), np.max(pred))]
    axes[0].plot(lim, lim, "--", color=PALETTE["muted"])
    axes[0].set(xlabel="Actual", ylabel="Predicted", title=f"Actual vs predicted (R² = {r2_score(y_true, pred):.3f})")
    axes[1].scatter(pred, resid, alpha=0.6, color=PALETTE["accent"], s=28)
    axes[1].axhline(0, ls="--", color=PALETTE["muted"])
    axes[1].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs predicted")
    sns.histplot(resid, kde=True, ax=axes[2], color=PALETTE["good"])
    axes[2].set(
        xlabel="Residual", title=f"Residual distribution (RMSE = {np.sqrt(mean_squared_error(y_true, pred)):.2f})"
    )
    fig.suptitle(title)
    return _save(fig, path)


def plot_tuning_history(history: pd.DataFrame, metric: str, path: Path, title: str) -> Path | None:
    if history is None or history.empty or "value" not in history:
        return None
    h = history.dropna(subset=["value"]).copy()
    if h.empty:
        return None
    h["best_so_far"] = h["value"].cummax()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(h["number"], h["value"], color=PALETTE["muted"], s=30, label="trial")
    ax.plot(h["number"], h["best_so_far"], color=PALETTE["primary"], lw=2.5, label="best so far")
    ax.set(xlabel="Optuna trial", ylabel=f"CV {metric}", title=title)
    ax.legend()
    return _save(fig, path)


def make_figures(
    task: Task,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    leaderboard: pd.DataFrame,
    figures_dir: Path,
    *,
    threshold: float | None = None,
    tuning_history: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Generate every figure relevant to ``task`` and return ``{name: path}``."""
    d = figures_dir / task.name
    d.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Path | None] = {}
    y_true = np.asarray(y_test)
    figs["leaderboard"] = plot_leaderboard(
        leaderboard, task.primary_metric, d / "leaderboard.png", f"{task.name}: model leaderboard"
    )
    figs["tuning"] = plot_tuning_history(
        tuning_history, task.primary_metric, d / "tuning_history.png", f"{task.name}: Optuna tuning"
    )

    if task.kind == "regression":
        pred = pipeline.predict(X_test)
        figs["diagnostics"] = plot_regression_diagnostics(
            y_true, pred, d / "regression_diagnostics.png", f"{task.name}: hold-out diagnostics"
        )
    else:
        proba = pipeline.predict_proba(X_test)
        if task.kind == "binary":
            p1 = proba[:, 1]
            thr = 0.5 if threshold is None else threshold
            pred = (p1 >= thr).astype(int)
            figs["roc_pr"] = plot_roc_pr(y_true, p1, d / "roc_pr.png", f"{task.name}: ROC & PR (hold-out)")
            figs["calibration"] = plot_calibration(y_true, p1, d / "calibration.png", f"{task.name}: calibration")
            figs["threshold"] = plot_threshold_sweep(
                y_true, p1, thr, d / "threshold_sweep.png", f"{task.name}: threshold trade-off"
            )
            labels = ["not at risk", "at risk"]
        else:
            pred = np.argmax(proba, axis=1)
            labels = list(task.classes or [])
        figs["confusion"] = plot_confusion(
            y_true, pred, labels, d / "confusion_matrix.png", f"{task.name}: confusion matrix"
        )
    return {k: str(v) for k, v in figs.items() if v is not None}
