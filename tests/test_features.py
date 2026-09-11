import numpy as np
import pandas as pd
import pytest

from edupulse.features.engineering import DomainRules, FeatureEngineer, build_preprocessor, engineer_features
from edupulse.tasks import TASKS, get_task


def test_engineer_features_adds_expected_columns(raw_df):
    out = engineer_features(raw_df)
    for col in (
        "total_score",
        "average_score",
        "pass_math",
        "n_passed",
        "performance_level",
        "at_risk",
        "parental_education_rank",
    ):
        assert col in out.columns
    assert len(out) == len(raw_df)


def test_engineered_values_are_consistent(raw_df):
    out = engineer_features(raw_df, DomainRules(pass_mark=40, at_risk_threshold=60))
    scores = raw_df[["math_score", "reading_score", "writing_score"]]
    assert (out["total_score"] == scores.sum(axis=1)).all()
    assert np.allclose(out["average_score"], scores.mean(axis=1).round(2))
    assert ((out["at_risk"] == 1) == (out["average_score"] < 60)).all()
    assert set(out["performance_level"]) <= {"low", "medium", "high"}
    assert (out.loc[out["average_score"] >= 80, "performance_level"] == "high").all()
    assert (out.loc[out["average_score"] < 60, "performance_level"] == "low").all()


def test_domain_rules_are_respected(raw_df):
    strict = engineer_features(raw_df, DomainRules(at_risk_threshold=90))
    lax = engineer_features(raw_df, DomainRules(at_risk_threshold=10))
    assert strict["at_risk"].mean() > lax["at_risk"].mean()


def test_feature_engineer_transformer_keeps_only_requested(raw_df):
    fe = FeatureEngineer(keep=["gender", "parental_education_rank", "average_score"]).fit(raw_df)
    out = fe.transform(raw_df)
    assert list(out.columns) == ["gender", "parental_education_rank", "average_score"]


def test_feature_engineer_works_without_scores(raw_df):
    background = raw_df.drop(columns=["math_score", "reading_score", "writing_score"])
    fe = FeatureEngineer(keep=list(TASKS["at_risk"].features)).fit(background)
    assert fe.transform(background).shape == (len(background), 5)


def test_feature_engineer_raises_on_impossible_column(raw_df):
    background = raw_df.drop(columns=["math_score", "reading_score", "writing_score"])
    with pytest.raises(KeyError):
        FeatureEngineer(keep=["average_score"]).fit(background).transform(background)


def test_preprocessor_one_hot_and_ordinal(raw_df):
    feats = TASKS["math_score"].features
    pre = build_preprocessor(feats).fit(raw_df[feats])
    Z = pre.transform(raw_df[feats])
    assert isinstance(Z, pd.DataFrame)
    assert "parental_level_of_education" in Z.columns  # ordinal keeps one column
    assert any(c.startswith("race_ethnicity_") for c in Z.columns)  # one-hot expands
    assert Z["parental_level_of_education"].between(0, 5).all()
    assert Z.isna().sum().sum() == 0


def test_preprocessor_handles_unknown_category(raw_df):
    feats = TASKS["at_risk"].features
    pre = build_preprocessor(feats).fit(raw_df[feats])
    weird = raw_df[feats].head(1).copy()
    weird["gender"] = "other"
    Z = pre.transform(weird)
    assert Z.filter(like="gender_").sum(axis=1).item() == 0


@pytest.mark.parametrize("name", list(TASKS))
def test_tasks_build_xy(engineered_df, name):
    task = get_task(name)
    X, y = task.build_xy(engineered_df)
    assert list(X.columns) == task.features
    assert len(X) == len(y)
    if task.kind == "multiclass":
        assert set(y.unique()) <= {0, 1, 2}
        assert task.decode([0, 2]) == ["low", "high"]
    if task.kind == "binary":
        assert set(y.unique()) <= {0, 1}


def test_task_lookup_accepts_dashes():
    assert get_task("at-risk").name == "at_risk"
    with pytest.raises(KeyError):
        get_task("nope")


def test_at_risk_task_uses_no_score_columns():
    assert not any("score" in f for f in TASKS["at_risk"].features)
