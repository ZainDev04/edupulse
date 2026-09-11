"""Automated exploratory data analysis (EDA) report.

Generates a set of figures + a JSON profile of the dataset that the README,
notebooks and dashboard all reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from edupulse.data.schema import CATEGORICAL_COLUMNS, SCORE_COLUMNS  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", palette="deep")


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def profile(df: pd.DataFrame) -> dict[str, Any]:
    """Numeric + categorical profile and simple statistical tests."""
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "scores": df[SCORE_COLUMNS].describe().round(2).to_dict(),
        "score_correlation": df[SCORE_COLUMNS].corr().round(3).to_dict(),
        "categorical": {c: df[c].value_counts().to_dict() for c in CATEGORICAL_COLUMNS},
        "at_risk_rate": float(df["at_risk"].mean()) if "at_risk" in df else None,
        "performance_level_share": df["performance_level"].value_counts(normalize=True).round(3).to_dict()
        if "performance_level" in df
        else None,
        "tests": {},
    }
    # Effect of each background attribute on average score (one-way ANOVA / t-test)
    target = "average_score" if "average_score" in df else SCORE_COLUMNS[0]
    for c in CATEGORICAL_COLUMNS:
        groups = [g[target].to_numpy() for _, g in df.groupby(c, observed=True)]
        if len(groups) == 2:
            stat, p = stats.ttest_ind(*groups, equal_var=False)
            test = "welch_t"
        else:
            stat, p = stats.f_oneway(*groups)
            test = "anova_f"
        eta_sq = _eta_squared(df, c, target)
        out["tests"][c] = {"test": test, "statistic": float(stat), "p_value": float(p), "eta_squared": eta_sq}
    return out


def _eta_squared(df: pd.DataFrame, group: str, target: str) -> float:
    grand = df[target].mean()
    ss_between = sum(len(g) * (g[target].mean() - grand) ** 2 for _, g in df.groupby(group, observed=True))
    ss_total = ((df[target] - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total else 0.0


def make_eda_figures(df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    figs: dict[str, str] = {}
    d = out_dir / "eda"

    # 1. Score distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, color in zip(axes, SCORE_COLUMNS, ["#4F46E5", "#F59E0B", "#10B981"], strict=False):
        sns.histplot(df[col], bins=25, kde=True, ax=ax, color=color)
        ax.axvline(df[col].mean(), ls="--", color="black", label=f"mean = {df[col].mean():.1f}")
        ax.set_title(col.replace("_", " ").title())
        ax.legend()
    figs["score_distributions"] = _save(fig, d / "score_distributions.png")

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(df[SCORE_COLUMNS].corr(), annot=True, fmt=".2f", cmap="Blues", vmin=0.5, vmax=1, ax=ax, cbar=False)
    ax.set_title("Score correlations")
    figs["correlation"] = _save(fig, d / "score_correlation.png")

    # 3. Average score by each background attribute
    target = "average_score" if "average_score" in df else SCORE_COLUMNS[0]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, col in zip(axes.flat, CATEGORICAL_COLUMNS, strict=False):
        order = df.groupby(col, observed=True)[target].median().sort_values().index
        sns.boxplot(data=df, x=col, y=target, order=order, ax=ax, hue=col, legend=False, palette="crest")
        ax.set_title(f"{target.replace('_', ' ')} by {col.replace('_', ' ')}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    axes.flat[-1].axis("off")
    figs["scores_by_background"] = _save(fig, d / "scores_by_background.png")

    # 4. At-risk rate by attribute
    if "at_risk" in df:
        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
        for ax, col in zip(axes, CATEGORICAL_COLUMNS, strict=False):
            rate = df.groupby(col, observed=True)["at_risk"].mean().sort_values()
            ax.barh(rate.index, rate.values, color="#EF4444")
            ax.axvline(df["at_risk"].mean(), ls="--", color="black")
            ax.set_title(col.replace("_", " "))
            ax.set_xlabel("at-risk rate")
        figs["at_risk_by_background"] = _save(fig, d / "at_risk_by_background.png")

    # 5. Pairwise scores coloured by test prep
    g = sns.pairplot(
        df[SCORE_COLUMNS + ["test_preparation_course"]],
        hue="test_preparation_course",
        corner=True,
        height=3,
        plot_kws={"alpha": 0.5, "s": 18},
    )
    g.figure.suptitle("Score relationships by test-preparation status", y=1.02)
    figs["pairplot"] = _save(g.figure, d / "score_pairplot.png")

    # 6. Performance-level share
    if "performance_level" in df:
        fig, ax = plt.subplots(figsize=(6, 5))
        share = df["performance_level"].value_counts().reindex(["low", "medium", "high"])
        ax.bar(share.index, share.values, color=["#EF4444", "#F59E0B", "#10B981"])
        for i, v in enumerate(share.values):
            ax.text(i, v, f"{v} ({v / len(df):.0%})", ha="center", va="bottom")
        ax.set_title("Performance-level distribution")
        figs["performance_levels"] = _save(fig, d / "performance_levels.png")

    return figs


def run_eda(df: pd.DataFrame, figures_dir: Path, reports_dir: Path) -> dict[str, Any]:
    prof = profile(df)
    prof["figures"] = make_eda_figures(df, figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "eda_profile.json").write_text(json.dumps(prof, indent=2, default=_json_default), encoding="utf-8")
    return prof


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)
