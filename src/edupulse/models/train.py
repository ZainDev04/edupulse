"""Training engine.

Workflow
--------
1. **Leaderboard** - every candidate in the model zoo is scored with repeated,
   stratified cross-validation on the training split.
2. **Tuning** - the best family (excluding baselines) is tuned with Optuna
   (TPE sampler + median pruning) on the same CV protocol.
3. **Threshold optimisation** (binary tasks) - the decision threshold is chosen
   on out-of-fold probabilities to maximise F-beta, because in an early-warning
   setting missing an at-risk student is costlier than a false alarm.
4. **Final fit** on the full training split.

Everything is wrapped in a single :class:`sklearn.pipeline.Pipeline`
(``features -> preprocess -> model``) so persisted artefacts accept raw rows.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import fbeta_score, get_scorer, precision_score, recall_score
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from edupulse.config import Settings, get_settings
from edupulse.features.engineering import DomainRules, FeatureEngineer, build_preprocessor
from edupulse.logging_utils import get_logger
from edupulse.models.zoo import Candidate, get_candidates
from edupulse.tasks import Task

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrainingResult:
    task: Task
    pipeline: Pipeline
    best_candidate: str
    best_params: dict[str, Any]
    leaderboard: pd.DataFrame
    cv_score: float
    threshold: float | None = None
    tuning_history: pd.DataFrame | None = None
    feature_names: list[str] = field(default_factory=list)
    train_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_pipeline(task: Task, estimator: BaseEstimator, rules: DomainRules | None = None) -> Pipeline:
    """Assemble ``FeatureEngineer -> ColumnTransformer -> estimator``."""
    return Pipeline(
        [
            ("features", FeatureEngineer(keep=list(task.features), rules=rules)),
            ("preprocess", build_preprocessor(list(task.features), scale_numeric=True)),
            ("model", estimator),
        ]
    )


def make_cv(task: Task, settings: Settings):
    if task.is_classification:
        return RepeatedStratifiedKFold(
            n_splits=settings.cv_folds, n_repeats=settings.cv_repeats, random_state=settings.random_state
        )
    return RepeatedKFold(n_splits=settings.cv_folds, n_repeats=settings.cv_repeats, random_state=settings.random_state)


def _scoring(task: Task) -> dict[str, str]:
    metrics = [task.primary_metric, *task.extra_metrics]
    if task.kind == "multiclass":
        # sklearn's roc_auc needs the ovr variant for multi-class
        metrics = [m if m != "roc_auc" else "roc_auc_ovr" for m in metrics]
    return {m: m for m in metrics}


# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #
def run_leaderboard(
    task: Task,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    settings: Settings | None = None,
    candidates: list[Candidate] | None = None,
    rules: DomainRules | None = None,
) -> pd.DataFrame:
    """Cross-validate every candidate and return a sorted leaderboard."""
    settings = settings or get_settings()
    candidates = candidates or get_candidates(task.kind, random_state=settings.random_state, n_jobs=settings.n_jobs)
    cv = make_cv(task, settings)
    scoring = _scoring(task)

    rows = []
    for cand in candidates:
        pipe = make_pipeline(task, cand.build(), rules)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=settings.n_jobs, error_score="raise")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Candidate %s failed: %s", cand.name, exc)
            continue
        row: dict[str, Any] = {"model": cand.name, "family": cand.family}
        for m in scoring:
            scores = res[f"test_{m}"]
            row[f"{m}_mean"] = float(np.mean(scores))
            row[f"{m}_std"] = float(np.std(scores))
        row["fit_seconds"] = float(np.mean(res["fit_time"]))
        row["elapsed_seconds"] = time.perf_counter() - t0
        rows.append(row)
        log.info(
            "%-24s %s=%.4f ± %.4f (%.1fs)",
            cand.name,
            task.primary_metric,
            row[f"{task.primary_metric}_mean"],
            row[f"{task.primary_metric}_std"],
            row["elapsed_seconds"],
        )

    board = pd.DataFrame(rows).sort_values(f"{task.primary_metric}_mean", ascending=False).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
def tune_candidate(
    task: Task,
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    settings: Settings | None = None,
    rules: DomainRules | None = None,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    """Optuna-tune ``candidate``; returns ``(best_params, best_score, history)``."""
    settings = settings or get_settings()
    if candidate.space is None:
        return {}, float("nan"), pd.DataFrame()

    cv = make_cv(task, settings)
    metric = _scoring(task)[task.primary_metric]

    def objective(trial: optuna.Trial) -> float:
        params = candidate.space(trial)
        pipe = make_pipeline(task, candidate.build(**params), rules)
        scores = []
        for i, (tr, va) in enumerate(cv.split(X, y)):
            m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            scores.append(get_scorer(metric)(m, X.iloc[va], y.iloc[va]))
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
        study_name=f"{task.name}:{candidate.name}",
    )
    study.optimize(objective, n_trials=settings.n_trials, timeout=settings.tuning_timeout_s, show_progress_bar=False)

    history = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    log.info(
        "Tuned %s: best %s=%.4f over %d trials",
        candidate.name,
        task.primary_metric,
        study.best_value,
        len(study.trials),
    )
    return study.best_params, float(study.best_value), history


# --------------------------------------------------------------------------- #
# Threshold optimisation
# --------------------------------------------------------------------------- #
def optimise_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    strategy: str = "target_recall",
    target_recall: float = 0.80,
    beta: float = 2.0,
) -> tuple[float, dict[str, float]]:
    """Choose a decision threshold from out-of-fold probabilities.

    Strategies
    ----------
    ``target_recall``
        The *highest* threshold whose recall is still >= ``target_recall``
        (i.e. the most precise operating point that still catches the required
        share of at-risk students).  This is the default for the early-warning
        use-case because it maps directly to a policy statement.
    ``fbeta``
        Maximise F-beta (recall-weighted when ``beta > 1``).
    """
    y_true = np.asarray(y_true)
    grid = np.unique(np.round(proba, 4))  # ascending
    best_t, best_val = 0.5, -1.0
    if strategy == "target_recall":
        # recall is non-increasing in the threshold, so the last grid point that
        # still satisfies the target is the most precise admissible operating point
        for t in grid:
            rec = recall_score(y_true, (proba >= t).astype(int), zero_division=0)
            if rec >= target_recall:
                best_t = float(t)
            else:
                break
    elif strategy == "fbeta":
        for t in grid:
            f = fbeta_score(y_true, (proba >= t).astype(int), beta=beta, zero_division=0)
            if f > best_val:
                best_t, best_val = float(t), float(f)
    else:
        raise ValueError(f"Unknown threshold strategy '{strategy}'")
    pred = (proba >= best_t).astype(int)
    stats = {
        "strategy": strategy,
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "flagged_rate": float(pred.mean()),
        "fbeta": float(fbeta_score(y_true, pred, beta=beta, zero_division=0)),
    }
    return best_t, stats


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_task(
    task: Task,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    settings: Settings | None = None,
    rules: DomainRules | None = None,
    candidates: list[Candidate] | None = None,
) -> TrainingResult:
    """Leaderboard -> tune best -> (threshold) -> final fit."""
    settings = settings or get_settings()
    t0 = time.perf_counter()
    candidates = candidates or get_candidates(task.kind, random_state=settings.random_state, n_jobs=settings.n_jobs)
    by_name = {c.name: c for c in candidates}

    board = run_leaderboard(task, X, y, settings=settings, candidates=candidates, rules=rules)
    non_baseline = board[board["family"] != "dummy"]
    best_name = str(non_baseline.iloc[0]["model"])
    best = by_name[best_name]
    cv_score = float(non_baseline.iloc[0][f"{task.primary_metric}_mean"])
    log.info("Best family: %s (%s=%.4f)", best_name, task.primary_metric, cv_score)

    params: dict[str, Any] = {}
    history = None
    if settings.tune and best.space is not None:
        params, tuned_score, history = tune_candidate(task, best, X, y, settings=settings, rules=rules)
        if np.isfinite(tuned_score) and tuned_score >= cv_score:
            cv_score = tuned_score
        else:  # tuned worse than defaults - keep defaults
            log.info("Tuning did not beat defaults (%.4f < %.4f); keeping defaults", tuned_score, cv_score)
            params = {}

    pipeline = make_pipeline(task, best.build(**params), rules)

    threshold = None
    if task.kind == "binary":
        cv = make_cv(task, Settings(cv_folds=settings.cv_folds, cv_repeats=1, random_state=settings.random_state))
        oof = cross_val_predict(clone(pipeline), X, y, cv=cv, method="predict_proba", n_jobs=settings.n_jobs)[:, 1]
        threshold, stats = optimise_threshold(np.asarray(y), oof, target_recall=settings.target_recall)
        log.info(
            "Decision threshold=%.3f (OOF recall=%.3f, precision=%.3f, flagged=%.1f%%)",
            threshold,
            stats["recall"],
            stats["precision"],
            100 * stats["flagged_rate"],
        )

    pipeline.fit(X, y)
    feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())

    return TrainingResult(
        task=task,
        pipeline=pipeline,
        best_candidate=best_name,
        best_params=params,
        leaderboard=board,
        cv_score=cv_score,
        threshold=threshold,
        tuning_history=history,
        feature_names=feature_names,
        train_seconds=time.perf_counter() - t0,
    )
