"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edupulse.config import Settings
from edupulse.data.schema import ETHNICITIES, GENDERS, LUNCH, PARENTAL_EDUCATION, TEST_PREP
from edupulse.features.engineering import engineer_features
from edupulse.models.registry import ModelRegistry
from edupulse.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "StudentsPerformance.csv"


def synthetic_students(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the real schema (keeps tests independent of the CSV)."""
    rng = np.random.default_rng(seed)
    prep = rng.choice(TEST_PREP, n)
    lunch = rng.choice(LUNCH, n)
    base = rng.normal(66, 14, n) + np.where(prep == "completed", 6, 0) + np.where(lunch == "standard", 5, -3)
    df = pd.DataFrame(
        {
            "gender": rng.choice(GENDERS, n),
            "race_ethnicity": rng.choice(ETHNICITIES, n),
            "parental_level_of_education": rng.choice(PARENTAL_EDUCATION, n),
            "lunch": lunch,
            "test_preparation_course": prep,
            "math_score": np.clip(base + rng.normal(0, 8, n), 0, 100).round().astype(int),
            "reading_score": np.clip(base + rng.normal(2, 7, n), 0, 100).round().astype(int),
            "writing_score": np.clip(base + rng.normal(1, 7, n), 0, 100).round().astype(int),
        }
    )
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return synthetic_students()


@pytest.fixture(scope="session")
def engineered_df(raw_df) -> pd.DataFrame:
    return engineer_features(raw_df)


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory, raw_df) -> Path:
    p = tmp_path_factory.mktemp("data") / "students.csv"
    raw_df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def fast_settings(tmp_path_factory, synthetic_csv) -> Settings:
    root = tmp_path_factory.mktemp("work")
    return Settings(
        raw_data_path=synthetic_csv,
        models_dir=root / "models",
        figures_dir=root / "figures",
        reports_dir=root / "reports",
        processed_dir=root / "processed",
        tracking_uri="sqlite:///" + (root / "mlruns" / "mlflow.db").as_posix(),
        tracking_artifacts=root / "mlruns" / "artifacts",
        tracking_experiment="edupulse-tests",
        cv_folds=3,
        cv_repeats=1,
        tune=False,
        n_jobs=1,
    )


@pytest.fixture(scope="session")
def trained_at_risk(fast_settings):
    """A fully trained + registered at_risk model (session-scoped: trained once)."""
    from edupulse.models import zoo

    cands = [
        c
        for c in zoo.get_candidates("binary", n_jobs=1)
        if c.name in ("baseline_majority", "logistic_regression", "random_forest")
    ]
    from edupulse.models import train as train_mod

    original = train_mod.get_candidates
    train_mod.get_candidates = lambda *a, **k: cands  # limit zoo for speed
    try:
        out = run_pipeline(
            "at_risk",
            settings=fast_settings,
            registry=ModelRegistry(fast_settings.models_dir),
            data_path=fast_settings.raw_data_path,
        )
    finally:
        train_mod.get_candidates = original
    return out


@pytest.fixture(scope="session")
def has_real_data() -> bool:
    return RAW.exists()
