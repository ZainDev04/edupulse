import json

import numpy as np
import pandas as pd
import pytest

from edupulse.models.evaluate import evaluate_task
from edupulse.models.explain import Explainer, group_map
from edupulse.models.fairness import fairness_summary, subgroup_metrics
from edupulse.models.registry import ModelRegistry
from edupulse.models.train import make_pipeline, optimise_threshold
from edupulse.models.zoo import get_candidates
from edupulse.tasks import get_task


def test_zoo_has_baselines_and_tree_models():
    clf = [c.name for c in get_candidates("binary")]
    reg = [c.name for c in get_candidates("regression")]
    assert "baseline_majority" in clf and "random_forest" in clf and "logistic_regression" in clf
    assert "baseline_mean" in reg and "ridge" in reg
    with pytest.raises(ValueError):
        get_candidates("nope")


def test_optimise_threshold_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 500), 0, 1)
    t, stats = optimise_threshold(y, p, target_recall=0.8)
    assert 0 < t < 1
    assert stats["recall"] >= 0.8
    t2, stats2 = optimise_threshold(y, p, strategy="fbeta", beta=2)
    assert stats2["fbeta"] > 0
    with pytest.raises(ValueError):
        optimise_threshold(y, p, strategy="magic")


def test_make_pipeline_fits_and_predicts(engineered_df):
    task = get_task("at_risk")
    X, y = task.build_xy(engineered_df)
    cand = next(c for c in get_candidates("binary", n_jobs=1) if c.name == "logistic_regression")
    pipe = make_pipeline(task, cand.build()).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and (0 <= proba).all() and (proba <= 1).all()


def test_regression_pipeline_and_metrics(engineered_df):
    task = get_task("math_score")
    X, y = task.build_xy(engineered_df)
    cand = next(c for c in get_candidates("regression", n_jobs=1) if c.name == "ridge")
    pipe = make_pipeline(task, cand.build()).fit(X, y)
    m = evaluate_task(task, pipe, X, y)
    assert {"r2", "rmse", "mae"} <= m.keys()
    assert m["r2"] > 0.5  # reading/writing strongly predict math


def test_multiclass_pipeline_and_metrics(engineered_df):
    task = get_task("performance_level")
    X, y = task.build_xy(engineered_df)
    cand = next(c for c in get_candidates("multiclass", n_jobs=1) if c.name == "random_forest")
    pipe = make_pipeline(task, cand.build(n_estimators=50)).fit(X, y)
    m = evaluate_task(task, pipe, X, y)
    assert m["accuracy"] > 0.9  # leaky by design
    assert np.array(m["confusion_matrix"]).shape == (3, 3)


def test_group_map_aggregates_one_hot():
    enc = ["gender_female", "gender_male", "race_ethnicity_group A", "parental_level_of_education", "reading_score"]
    m = group_map(enc, ["gender", "race_ethnicity", "parental_level_of_education", "reading_score"])
    assert m["gender_female"] == "gender" and m["race_ethnicity_group A"] == "race_ethnicity"
    assert m["parental_level_of_education"] == "parental_level_of_education"


def test_explainer_tree_and_linear(engineered_df):
    task = get_task("at_risk")
    X, y = task.build_xy(engineered_df)
    for name in ("random_forest", "logistic_regression"):
        cand = next(c for c in get_candidates("binary", n_jobs=1) if c.name == name)
        pipe = make_pipeline(task, cand.build(**({"n_estimators": 30} if name == "random_forest" else {}))).fit(X, y)
        ex = Explainer(pipe, task, X, max_background=50)
        gi = ex.global_importance(X.head(40))
        assert set(gi["feature"]) == set(task.features)
        local = ex.local_explanation(X.head(1), top_k=3)
        assert len(local) == 3 and {"feature", "value", "contribution"} <= local[0].keys()


def test_fairness_subgroups(engineered_df):
    task = get_task("at_risk")
    X, y = task.build_xy(engineered_df)
    cand = next(c for c in get_candidates("binary", n_jobs=1) if c.name == "logistic_regression")
    pipe = make_pipeline(task, cand.build()).fit(X, y)
    g = subgroup_metrics(task, pipe, X, y, threshold=0.4)
    assert {"attribute", "group", "n", "tpr", "selection_rate"} <= set(g.columns)
    s = fairness_summary(g, task)
    assert "gender" in s and "tpr_gap" in s["gender"]


# --------------------------------------------------------------------------- #
# End-to-end (session-scoped fixture trains once)
# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end_registers_artifacts(trained_at_risk, fast_settings):
    out = trained_at_risk
    d = out.artefact_dir
    assert (d / "pipeline.joblib").exists() and (d / "metadata.json").exists() and (d / "model_card.md").exists()
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta["task"] == "at_risk" and meta["threshold"] is not None
    assert 0.5 < out.test_metrics["roc_auc"] <= 1.0
    assert "shap_importance" in out.explainability
    assert out.fairness["summary"]
    assert (fast_settings.figures_dir / "at_risk" / "roc_pr.png").exists()
    board = pd.read_csv(d / "leaderboard.csv")
    assert board.iloc[0]["rank"] == 1


def test_registry_load_roundtrip(trained_at_risk, fast_settings):
    reg = ModelRegistry(fast_settings.models_dir)
    assert reg.latest_version("at_risk") == trained_at_risk.artefact_dir.name
    m = reg.load("at_risk")
    assert m.task.name == "at_risk" and m.threshold == trained_at_risk.training.threshold
    with pytest.raises(FileNotFoundError):
        reg.load("math_score")
