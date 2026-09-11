"""Model zoo: candidate estimators + Optuna search spaces per task kind.

Optional gradient-boosting libraries (XGBoost, LightGBM) are used when
installed and silently skipped otherwise, so the project runs on a minimal
scikit-learn install.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

try:  # optional
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:  # optional
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False


@dataclass(frozen=True)
class Candidate:
    """An estimator factory plus its Optuna search space."""

    name: str
    family: str
    make: Callable[..., BaseEstimator]
    space: Callable[[Any], dict[str, Any]] | None = None
    supports_shap_tree: bool = False

    def build(self, **overrides) -> BaseEstimator:
        return self.make(**overrides)


# --------------------------------------------------------------------------- #
# Search spaces (Optuna trial -> params)
# --------------------------------------------------------------------------- #
def _rf_space(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": t.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _gb_space(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 500, step=25),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 30),
    }


def _hgb_space(t):
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": t.suggest_int("max_iter", 50, 500, step=25),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 4, 64),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 60),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    }


def _xgb_space(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": t.suggest_int("min_child_weight", 1, 20),
    }


def _lgbm_space(t):
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": t.suggest_int("num_leaves", 4, 64),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 60),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _logreg_space(t):
    return {"C": t.suggest_float("C", 1e-3, 100.0, log=True)}


def _ridge_space(t):
    return {"alpha": t.suggest_float("alpha", 1e-3, 100.0, log=True)}


def _svm_space(t):
    return {
        "C": t.suggest_float("C", 1e-2, 100.0, log=True),
        "gamma": t.suggest_float("gamma", 1e-4, 1.0, log=True),
    }


def _knn_space(t):
    return {
        "n_neighbors": t.suggest_int("n_neighbors", 3, 60),
        "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
    }


# --------------------------------------------------------------------------- #
# Candidate lists
# --------------------------------------------------------------------------- #
def _mk(cls, **defaults):
    """Factory whose call-time kwargs (e.g. tuned params) override the defaults."""

    def make(**overrides):
        return cls(**{**defaults, **overrides})

    return make


def _classification(random_state: int, n_jobs: int, balanced: bool) -> list[Candidate]:
    cw = "balanced" if balanced else None
    cands = [
        Candidate("baseline_majority", "dummy", _mk(DummyClassifier, strategy="prior")),
        Candidate(
            "logistic_regression", "linear", _mk(LogisticRegression, max_iter=2000, class_weight=cw), _logreg_space
        ),
        Candidate(
            "random_forest",
            "tree",
            _mk(RandomForestClassifier, n_estimators=300, class_weight=cw, random_state=random_state, n_jobs=n_jobs),
            _rf_space,
            supports_shap_tree=True,
        ),
        Candidate(
            "extra_trees",
            "tree",
            _mk(ExtraTreesClassifier, n_estimators=300, class_weight=cw, random_state=random_state, n_jobs=n_jobs),
            _rf_space,
            supports_shap_tree=True,
        ),
        Candidate(
            "gradient_boosting",
            "tree",
            _mk(GradientBoostingClassifier, random_state=random_state),
            _gb_space,
            supports_shap_tree=True,
        ),
        Candidate(
            "hist_gradient_boosting",
            "tree",
            _mk(HistGradientBoostingClassifier, class_weight=cw, random_state=random_state),
            _hgb_space,
            supports_shap_tree=True,
        ),
        Candidate(
            "svm_rbf", "kernel", _mk(SVC, probability=True, class_weight=cw, random_state=random_state), _svm_space
        ),
        Candidate("knn", "neighbors", _mk(KNeighborsClassifier), _knn_space),
    ]
    if HAS_XGB:
        cands.append(
            Candidate(
                "xgboost",
                "tree",
                _mk(
                    XGBClassifier,
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    eval_metric="logloss",
                    verbosity=0,
                ),
                _xgb_space,
                supports_shap_tree=True,
            )
        )
    if HAS_LGBM:
        cands.append(
            Candidate(
                "lightgbm",
                "tree",
                _mk(
                    LGBMClassifier,
                    n_estimators=300,
                    learning_rate=0.05,
                    class_weight=cw,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbose=-1,
                ),
                _lgbm_space,
                supports_shap_tree=True,
            )
        )
    return cands


def _regression(random_state: int, n_jobs: int) -> list[Candidate]:
    cands = [
        Candidate("baseline_mean", "dummy", _mk(DummyRegressor, strategy="mean")),
        Candidate("ridge", "linear", _mk(Ridge), _ridge_space),
        Candidate(
            "random_forest",
            "tree",
            _mk(RandomForestRegressor, n_estimators=300, random_state=random_state, n_jobs=n_jobs),
            _rf_space,
            supports_shap_tree=True,
        ),
        Candidate(
            "extra_trees",
            "tree",
            _mk(ExtraTreesRegressor, n_estimators=300, random_state=random_state, n_jobs=n_jobs),
            _rf_space,
            supports_shap_tree=True,
        ),
        Candidate(
            "gradient_boosting",
            "tree",
            _mk(GradientBoostingRegressor, random_state=random_state),
            _gb_space,
            supports_shap_tree=True,
        ),
        Candidate(
            "hist_gradient_boosting",
            "tree",
            _mk(HistGradientBoostingRegressor, random_state=random_state),
            _hgb_space,
            supports_shap_tree=True,
        ),
        Candidate("svm_rbf", "kernel", _mk(SVR), _svm_space),
        Candidate("knn", "neighbors", _mk(KNeighborsRegressor), _knn_space),
    ]
    if HAS_XGB:
        cands.append(
            Candidate(
                "xgboost",
                "tree",
                _mk(
                    XGBRegressor,
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=0,
                ),
                _xgb_space,
                supports_shap_tree=True,
            )
        )
    if HAS_LGBM:
        cands.append(
            Candidate(
                "lightgbm",
                "tree",
                _mk(
                    LGBMRegressor,
                    n_estimators=300,
                    learning_rate=0.05,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbose=-1,
                ),
                _lgbm_space,
                supports_shap_tree=True,
            )
        )
    return cands


def get_candidates(kind: str, *, random_state: int = 42, n_jobs: int = -1, balanced: bool = True) -> list[Candidate]:
    """Return the candidate models for a task kind (binary/multiclass/regression)."""
    if kind == "regression":
        return _regression(random_state, n_jobs)
    if kind in ("binary", "multiclass"):
        return _classification(random_state, n_jobs, balanced)
    raise ValueError(f"Unknown task kind: {kind}")


def get_candidate(kind: str, name: str, **kwargs) -> Candidate:
    for c in get_candidates(kind, **kwargs):
        if c.name == name:
            return c
    raise KeyError(f"No candidate '{name}' for kind '{kind}'")
