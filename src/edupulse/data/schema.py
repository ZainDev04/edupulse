"""Declarative schema + validation for the Students Performance dataset.

A lightweight alternative to *pandera* with zero extra dependencies: every
column declares its dtype, allowed categories or numeric range, and the
validator raises a rich :class:`SchemaError` listing *all* violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

GENDERS = ("female", "male")
ETHNICITIES = ("group A", "group B", "group C", "group D", "group E")
PARENTAL_EDUCATION = (
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
)
LUNCH = ("free/reduced", "standard")
TEST_PREP = ("none", "completed")

CATEGORICAL_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
SCORE_COLUMNS = ["math_score", "reading_score", "writing_score"]


class SchemaError(ValueError):
    """Raised when a dataframe violates :class:`DataSchema`."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str  # "categorical" | "numeric"
    categories: tuple[str, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class DataSchema:
    columns: tuple[ColumnSpec, ...] = field(
        default_factory=lambda: (
            ColumnSpec("gender", "categorical", GENDERS),
            ColumnSpec("race_ethnicity", "categorical", ETHNICITIES),
            ColumnSpec("parental_level_of_education", "categorical", PARENTAL_EDUCATION),
            ColumnSpec("lunch", "categorical", LUNCH),
            ColumnSpec("test_preparation_course", "categorical", TEST_PREP),
            ColumnSpec("math_score", "numeric", min_value=0, max_value=100),
            ColumnSpec("reading_score", "numeric", min_value=0, max_value=100),
            ColumnSpec("writing_score", "numeric", min_value=0, max_value=100),
        )
    )

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]


def validate_dataframe(
    df: pd.DataFrame, schema: DataSchema | None = None, *, require_scores: bool = True
) -> pd.DataFrame:
    """Validate ``df`` against ``schema`` and return it unchanged.

    Parameters
    ----------
    df:
        Dataframe with *normalised* column names.
    schema:
        Schema to validate against (defaults to :class:`DataSchema`).
    require_scores:
        When ``False`` score columns are optional (useful for inference where
        only background features are available).
    """
    schema = schema or DataSchema()
    problems: list[str] = []

    for spec in schema.columns:
        if spec.name not in df.columns:
            if spec.kind == "numeric" and not require_scores:
                continue
            problems.append(f"missing column '{spec.name}'")
            continue
        col = df[spec.name]
        if col.isna().any():
            problems.append(f"column '{spec.name}' has {int(col.isna().sum())} missing values")
        if spec.kind == "categorical":
            bad = set(col.dropna().unique()) - set(spec.categories or ())
            if bad:
                problems.append(f"column '{spec.name}' has unknown categories {sorted(map(str, bad))}")
        else:
            if not pd.api.types.is_numeric_dtype(col):
                problems.append(f"column '{spec.name}' must be numeric, got {col.dtype}")
                continue
            lo, hi = spec.min_value, spec.max_value
            out = col.dropna()
            if lo is not None and (out < lo).any():
                problems.append(f"column '{spec.name}' has values < {lo}")
            if hi is not None and (out > hi).any():
                problems.append(f"column '{spec.name}' has values > {hi}")

    if problems:
        raise SchemaError("Dataframe failed schema validation:\n  - " + "\n  - ".join(problems))
    return df
