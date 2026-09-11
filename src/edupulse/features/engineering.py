"""Feature engineering.

Two layers:

1. :func:`engineer_features` / :class:`FeatureEngineer` - *domain* features
   derived from raw columns (totals, averages, pass flags, tiers).  These are
   pure pandas transformations and are also used to build training targets.
2. :func:`build_preprocessor` - a scikit-learn ``ColumnTransformer`` that
   turns a feature frame into a numeric matrix (ordinal encoding for the
   naturally-ordered parental education, one-hot for nominal columns, scaling
   for numeric columns).  It is embedded in every persisted model pipeline so
   that inference receives raw, human-readable inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from edupulse.data.schema import PARENTAL_EDUCATION, SCORE_COLUMNS

PERFORMANCE_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class DomainRules:
    pass_mark: int = 40
    at_risk_threshold: int = 60
    medium_cutoff: int = 60
    high_cutoff: int = 80


def _parental_rank(series: pd.Series) -> pd.Series:
    rank = {lvl: i for i, lvl in enumerate(PARENTAL_EDUCATION)}
    return series.map(rank).astype("Int64")


def engineer_features(df: pd.DataFrame, rules: DomainRules | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with derived columns appended.

    Added columns
    -------------
    total_score, average_score, score_std, min_score, max_score, score_range,
    pass_math, pass_reading, pass_writing, n_passed, all_passed,
    performance_level (low/medium/high), at_risk (average < threshold),
    parental_education_rank (ordinal 0-5)
    """
    rules = rules or DomainRules()
    out = df.copy()
    scores = out[SCORE_COLUMNS]

    out["total_score"] = scores.sum(axis=1)
    out["average_score"] = scores.mean(axis=1).round(2)
    out["score_std"] = scores.std(axis=1).round(2)
    out["min_score"] = scores.min(axis=1)
    out["max_score"] = scores.max(axis=1)
    out["score_range"] = out["max_score"] - out["min_score"]

    for subject in ("math", "reading", "writing"):
        out[f"pass_{subject}"] = (out[f"{subject}_score"] >= rules.pass_mark).astype(int)
    out["n_passed"] = out[["pass_math", "pass_reading", "pass_writing"]].sum(axis=1)
    out["all_passed"] = (out["n_passed"] == 3).astype(int)

    out["performance_level"] = pd.cut(
        out["average_score"],
        bins=[-np.inf, rules.medium_cutoff, rules.high_cutoff, np.inf],
        right=False,
        labels=PERFORMANCE_LEVELS,
    ).astype(str)
    out["at_risk"] = (out["average_score"] < rules.at_risk_threshold).astype(int)

    if "parental_level_of_education" in out.columns:
        out["parental_education_rank"] = _parental_rank(out["parental_level_of_education"])
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """scikit-learn wrapper around :func:`engineer_features`.

    Only the columns listed in ``keep`` are returned so the transformer can be
    dropped into a pipeline that expects a fixed feature set.  Score-derived
    features are only computed when the raw score columns are present, which
    lets the same transformer serve background-only tasks at inference time.
    """

    def __init__(self, keep: list[str] | None = None, rules: DomainRules | None = None):
        self.keep = keep
        self.rules = rules

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - sklearn API
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()
        if all(c in df.columns for c in SCORE_COLUMNS):
            df = engineer_features(df, self.rules)
        elif "parental_level_of_education" in df.columns:
            df["parental_education_rank"] = _parental_rank(df["parental_level_of_education"])
        if self.keep:
            missing = [c for c in self.keep if c not in df.columns]
            if missing:
                raise KeyError(f"FeatureEngineer cannot produce columns {missing}")
            df = df[self.keep]
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep or self.feature_names_in_, dtype=object)


NOMINAL = ["gender", "race_ethnicity", "lunch", "test_preparation_course"]
ORDINAL = {"parental_level_of_education": list(PARENTAL_EDUCATION)}


def build_preprocessor(feature_columns: list[str], *, scale_numeric: bool = True) -> ColumnTransformer:
    """Create a ``ColumnTransformer`` for the given feature columns.

    - nominal categoricals -> one-hot (unknown categories ignored at inference)
    - parental education   -> ordinal (respects its natural order)
    - numeric columns      -> optional standard scaling
    """
    nominal = [c for c in feature_columns if c in NOMINAL]
    ordinal = [c for c in feature_columns if c in ORDINAL]
    numeric = [c for c in feature_columns if c not in nominal and c not in ordinal]

    transformers = []
    if nominal:
        transformers.append(("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal))
    if ordinal:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ORDINAL[c] for c in ordinal],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal,
            )
        )
    if numeric:
        num_pipe: Pipeline | str = Pipeline([("scale", StandardScaler())]) if scale_numeric else "passthrough"
        transformers.append(("numeric", num_pipe, numeric))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    return pre
