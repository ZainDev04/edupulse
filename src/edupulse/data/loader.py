"""Data ingestion: read the raw CSV, normalise it and validate it."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from edupulse.config import get_settings
from edupulse.data.schema import validate_dataframe
from edupulse.logging_utils import get_logger

log = get_logger(__name__)

_COLUMN_ALIASES = {
    "race/ethnicity": "race_ethnicity",
    "parental level of education": "parental_level_of_education",
    "test preparation course": "test_preparation_course",
    "math score": "math_score",
    "reading score": "reading_score",
    "writing score": "writing_score",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw column names to ``snake_case`` identifiers."""
    renamed = {}
    for col in df.columns:
        key = col.strip().lower()
        new = _COLUMN_ALIASES.get(key) or re.sub(r"[^0-9a-z]+", "_", key).strip("_")
        renamed[col] = new
    return df.rename(columns=renamed)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw dataset exactly as shipped (only column names normalised)."""
    path = Path(path) if path else get_settings().raw_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'StudentsPerformance.csv' "
            "(Kaggle: spscientist/students-performance-in-exams) into data/raw/."
        )
    df = pd.read_csv(path)
    df = normalise_columns(df)
    log.info("Loaded raw dataset %s with shape %s", path.name, df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning: strip strings, drop exact duplicates, validate."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    # Category values in the schema use lower-case except "group A"... keep canonical casing
    if "race_ethnicity" in df.columns:
        df["race_ethnicity"] = df["race_ethnicity"].str.replace(
            r"^group ([a-e])$", lambda m: f"group {m.group(1).upper()}", regex=True
        )
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if before != len(df):
        log.info("Dropped %d duplicate rows", before - len(df))
    validate_dataframe(df)
    return df


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """Load + clean + validate in one call."""
    return clean(load_raw(path))
