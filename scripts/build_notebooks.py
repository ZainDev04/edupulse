"""Generate the analysis notebooks from source (keeps them reproducible + diff-friendly).

Run ``python scripts/build_notebooks.py`` then execute with
``jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb``
(or simply ``make notebooks``).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = """import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Image, Markdown, display
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)

from edupulse.config import get_settings
S = get_settings()
print("EduPulse project root:", S.project_root)"""


def nb(title: str, cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}")]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return n


# --------------------------------------------------------------------------- #
NB1 = nb(
    "01. Exploratory data analysis",
    [
        (
            "md",
            "Understand the *Students Performance* dataset before modelling: distributions, background effects, "
            "statistical significance and the class balance of the targets we will predict.\n\n"
            "All logic lives in the `edupulse` package. This notebook only calls it, so the results here are the same "
            "ones the CLI, API and dashboard use.",
        ),
        ("code", SETUP),
        (
            "md",
            "## 1. Load, clean and validate\nThe loader normalises column names, strips whitespace, drops duplicates and "
            "validates every column against a declared schema (allowed categories, score ranges).",
        ),
        ("code", "from edupulse.data import load_clean\nraw = load_clean()\nraw.head()"),
        ("code", "raw.describe(include='all').T"),
        (
            "md",
            "## 2. Feature engineering\n`engineer_features` adds totals, averages, pass flags, the ordinal parental-education rank "
            "and the two targets: `performance_level` (low/medium/high) and `at_risk` (average < 60).",
        ),
        (
            "code",
            "from edupulse.features import engineer_features\ndf = engineer_features(raw)\n"
            "print(df.shape)\ndf[['average_score','performance_level','at_risk','n_passed','parental_education_rank']].head(10)",
        ),
        (
            "code",
            "print('At-risk rate: {:.1%}'.format(df.at_risk.mean()))\ndf.performance_level.value_counts(normalize=True).round(3)",
        ),
        ("md", "## 3. Score distributions and correlations"),
        (
            "code",
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
            "for ax, c in zip(axes, ['math_score','reading_score','writing_score']):\n"
            "    sns.histplot(df[c], kde=True, bins=25, ax=ax); ax.axvline(df[c].mean(), ls='--', c='k'); ax.set_title(c)\n"
            "plt.tight_layout()",
        ),
        (
            "code",
            "sns.heatmap(df[['math_score','reading_score','writing_score']].corr(), annot=True, cmap='Blues', vmin=.5); plt.title('Score correlations');",
        ),
        (
            "md",
            "Reading and writing are almost interchangeable (r of about 0.95). Math is related but distinct (r of about 0.8), "
            "which is why predicting math from reading and writing is a meaningful regression task.",
        ),
        (
            "md",
            "## 4. Which background factors matter?\nWelch t-tests or one-way ANOVA on the average score, with eta squared as the effect size.",
        ),
        (
            "code",
            "from edupulse.eda import profile\nprof = profile(df)\n"
            "pd.DataFrame(prof['tests']).T.sort_values('eta_squared', ascending=False).style.format({'p_value':'{:.2e}','statistic':'{:.2f}','eta_squared':'{:.3f}'})",
        ),
        (
            "code",
            "fig, axes = plt.subplots(2, 3, figsize=(18, 9))\n"
            "for ax, c in zip(axes.flat, ['lunch','test_preparation_course','parental_level_of_education','race_ethnicity','gender']):\n"
            "    order = df.groupby(c)['average_score'].median().sort_values().index\n"
            "    sns.boxplot(data=df, x=c, y='average_score', order=order, ax=ax, hue=c, legend=False, palette='crest'); ax.tick_params(axis='x', rotation=25); ax.set_xlabel('')\n"
            "axes.flat[-1].axis('off'); plt.tight_layout()",
        ),
        (
            "md",
            "## 5. At-risk rate by background\nThis is the only signal the early-warning model gets. Scores are not allowed as inputs.",
        ),
        (
            "code",
            "fig, axes = plt.subplots(1, 5, figsize=(22, 4))\n"
            "for ax, c in zip(axes, ['lunch','test_preparation_course','parental_level_of_education','race_ethnicity','gender']):\n"
            "    df.groupby(c)['at_risk'].mean().sort_values().plot.barh(ax=ax, color='#EF4444'); ax.axvline(df.at_risk.mean(), ls='--', c='k'); ax.set_title(c); ax.set_ylabel('')\n"
            "plt.tight_layout()",
        ),
        (
            "md",
            "## Takeaways\n"
            "* Lunch type (a proxy for household income) and test preparation have the largest effects on scores. Ethnicity and parental education come next. Gender mainly changes which subject is stronger.\n"
            "* About 28.5% of students average below 60, so the binary target is moderately imbalanced.\n"
            "* Reading and writing correlate far more with each other than with math, so math is the natural regression target.",
        ),
    ],
)

NB2 = nb(
    "02. Modelling: leaderboard, tuning and evaluation",
    [
        (
            "md",
            "Compare ten model families with repeated stratified cross-validation, tune the winner with Optuna, pick an "
            "operating threshold, and evaluate on a held-out split. We run the at-risk early-warning task live here "
            "(fast settings) and load the registered artefacts for all three tasks.",
        ),
        ("code", SETUP),
        (
            "md",
            "## 1. A single call trains everything\n`run_pipeline` loads the data, engineers features, runs the leaderboard, tunes with Optuna, picks the threshold, evaluates, explains, audits fairness and registers the model.",
        ),
        (
            "code",
            "from edupulse.config import Settings\nfrom edupulse.models.registry import ModelRegistry\nfrom edupulse.pipeline import run_pipeline\n"
            "import tempfile\n"
            "tmp = Path(tempfile.mkdtemp())\n"
            "fast = S.model_copy(update=dict(models_dir=tmp/'models', figures_dir=tmp/'figures', reports_dir=tmp/'reports', processed_dir=tmp/'processed', cv_repeats=1, n_trials=15, n_jobs=1))\n"
            "out = run_pipeline('at_risk', settings=fast, registry=ModelRegistry(fast.models_dir))\n"
            "print(out.headline)",
        ),
        (
            "code",
            "board = out.training.leaderboard\nboard[['rank','model','family','roc_auc_mean','roc_auc_std','average_precision_mean','recall_mean','fit_seconds']].style.background_gradient(subset=['roc_auc_mean'], cmap='Blues').format(precision=4)",
        ),
        (
            "md",
            "Linear and kernel models win on this task. With only five categorical inputs (about 17 one-hot columns) there is little "
            "non-linear structure for trees to learn, and they over-fit the small training set.",
        ),
        ("code", "display(Image(str(fast.figures_dir/'at_risk'/'leaderboard.png')))"),
        ("md", "## 2. Hold-out evaluation"),
        (
            "code",
            "pd.Series({k: v for k, v in out.test_metrics.items() if isinstance(v, float)}).round(4).to_frame('hold-out')",
        ),
        (
            "code",
            "display(Image(str(fast.figures_dir/'at_risk'/'roc_pr.png')))\ndisplay(Image(str(fast.figures_dir/'at_risk'/'threshold_sweep.png')))\ndisplay(Image(str(fast.figures_dir/'at_risk'/'calibration.png')))",
        ),
        (
            "md",
            "### Why not accuracy?\nWith a 28% positive rate, always predicting not-at-risk scores 71% accuracy. We select models on ROC-AUC "
            "and then choose the decision threshold as the most precise operating point that still recalls at least 80% of "
            "at-risk students on out-of-fold predictions. That is a policy statement an academic advisor can understand.",
        ),
        (
            "md",
            "## 3. Registered models (full training run)\nThe artefacts under `models/` were produced by `edupulse train --all --trials 50`.",
        ),
        (
            "code",
            "reg = ModelRegistry(S.models_dir)\nrows = []\n"
            "for t in ('at_risk','math_score','performance_level'):\n"
            "    try:\n        m = reg.load(t)\n    except FileNotFoundError:\n        continue\n"
            "    md = m.metadata\n    rows.append({'task': t, 'model': md.model_name, 'version': md.version, 'cv_metric': md.cv_metric, 'cv_score': round(md.cv_score, 4),\n"
            "                 **{k: round(v, 4) for k, v in md.test_metrics.items() if k in ('roc_auc','average_precision','recall','precision','r2','rmse','mae','accuracy','f1_macro')}})\n"
            "pd.DataFrame(rows).set_index('task')",
        ),
        (
            "code",
            "for t in ('at_risk','math_score','performance_level'):\n"
            "    try: m = reg.load(t)\n    except FileNotFoundError: continue\n"
            "    display(Markdown(f'### {t}: {m.metadata.model_name}'))\n"
            "    for key in ('leaderboard','roc_pr','diagnostics','confusion'):\n"
            "        p = m.metadata.figures.get(key)\n"
            "        if p and Path(p).exists(): display(Image(p, width=800))",
        ),
        (
            "md",
            "## 4. The leakage case study\nThe original coursework predicted `performance_level` from the three scores. Because the label is "
            "defined as a threshold on their average, any model just re-learns the cut-offs, which gives close to 100% accuracy. "
            "The number is real but it tells you nothing. The at-risk task (background only) is the formulation that means something.",
        ),
        (
            "code",
            "try:\n    m = reg.load('performance_level'); print(m.task.leakage_note); print(); print(np.array(m.metadata.test_metrics['confusion_matrix']))\nexcept FileNotFoundError: print('performance_level model not trained yet')",
        ),
    ],
)

NB3 = nb(
    "03. Explainability and fairness",
    [
        (
            "md",
            "Global SHAP importance, per-student explanations, permutation importance, and a subgroup "
            "fairness audit of the at-risk model.",
        ),
        ("code", SETUP),
        (
            "code",
            "from edupulse.models.registry import ModelRegistry\nfrom edupulse.models.explain import Explainer\nfrom edupulse.pipeline import load_engineered\n"
            "reg = ModelRegistry(S.models_dir)\nm = reg.load('at_risk')\ndf = load_engineered(S)\nX = df[m.task.features]\n"
            "print(m.metadata.model_name, 'v' + m.metadata.version, '| threshold', round(m.threshold, 3))",
        ),
        ("md", "## 1. Global importance (mean absolute SHAP, aggregated back to the original features)"),
        (
            "code",
            "ex = Explainer(m.pipeline, m.task, X, max_background=200)\ngi = ex.global_importance(X.sample(300, random_state=0))\n"
            "gi.plot.barh(x='feature', y='mean_abs_shap', legend=False, color='#4F46E5', figsize=(8,4)); plt.gca().invert_yaxis(); plt.title('Mean absolute SHAP');\ngi",
        ),
        (
            "code",
            "p = m.metadata.figures.get('shap_beeswarm')\nif p and Path(p).exists(): display(Image(p, width=850))",
        ),
        (
            "md",
            "## 2. Explain one student\nContributions are in log-odds for the linear model and in probability space for tree models. Positive values push towards at risk.",
        ),
        (
            "code",
            "student = pd.DataFrame([{'gender':'male','race_ethnicity':'group A','parental_level_of_education':'high school','lunch':'free/reduced','test_preparation_course':'none'}])\n"
            "proba = m.pipeline.predict_proba(student)[0, 1]\nprint(f'P(at risk) = {proba:.3f}  ->  flagged: {proba >= m.threshold}')\n"
            "contrib = pd.DataFrame(ex.local_explanation(student, top_k=5))\n"
            "contrib.plot.barh(x='feature', y='contribution', color=np.where(contrib.contribution > 0, '#EF4444', '#10B981'), legend=False, figsize=(8,3)); plt.gca().invert_yaxis();\ncontrib",
        ),
        ("md", "## 3. Permutation importance (model-agnostic, hold-out)"),
        (
            "code",
            "pd.DataFrame(m.metadata.explainability['permutation_importance']).style.bar(subset=['importance_mean'], color='#A5B4FC').format(precision=4)",
        ),
        (
            "md",
            "## 4. Fairness audit\nWe slice hold-out performance by sensitive attributes. Because base rates genuinely differ between groups "
            "(for example, free/reduced-lunch students are at risk twice as often), some disparity in selection rate is expected. The "
            "metric to watch for an early-warning tool is the TPR gap (equal opportunity): are we equally good at catching "
            "at-risk students in every group?",
        ),
        (
            "code",
            "fair = pd.DataFrame(m.metadata.fairness['groups'])\nfair.style.format(precision=3).background_gradient(subset=['tpr'], cmap='Greens')",
        ),
        ("code", "pd.DataFrame(m.metadata.fairness['summary']).T.style.format(precision=3)"),
        ("code", "p = m.metadata.figures.get('fairness')\nif p and Path(p).exists(): display(Image(p, width=900))"),
        (
            "md",
            "## Takeaways\n* Income proxies (lunch, parental education) and test preparation dominate the risk score. Gender and ethnicity contribute less.\n"
            "* The model is a triage tool. SHAP explanations make each flag auditable by an advisor.\n"
            "* Group TPR gaps are recorded in the model card. Group-specific thresholds and reweighing are the next steps for mitigation.",
        ),
    ],
)

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, notebook in [
        ("01_eda.ipynb", NB1),
        ("02_modelling.ipynb", NB2),
        ("03_explainability_fairness.ipynb", NB3),
    ]:
        nbf.write(notebook, OUT / name)
        print("wrote", OUT / name)
