import pandas as pd
import pytest

from edupulse.data.loader import clean, load_clean, normalise_columns
from edupulse.data.schema import SchemaError, validate_dataframe
from tests.conftest import RAW


def test_normalise_columns_maps_raw_names():
    df = pd.DataFrame(
        columns=[
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
            "math score",
            "reading score",
            "writing score",
        ]
    )
    out = normalise_columns(df)
    assert list(out.columns) == [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
        "math_score",
        "reading_score",
        "writing_score",
    ]


def test_validate_accepts_clean_frame(raw_df):
    assert validate_dataframe(raw_df) is raw_df


def test_validate_reports_all_problems(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "gender"] = "unknown"
    bad.loc[1, "math_score"] = 140
    bad = bad.drop(columns=["lunch"])
    with pytest.raises(SchemaError) as exc:
        validate_dataframe(bad)
    msg = str(exc.value)
    assert "gender" in msg and "math_score" in msg and "lunch" in msg


def test_validate_allows_missing_scores_for_inference(raw_df):
    background = raw_df.drop(columns=["math_score", "reading_score", "writing_score"])
    validate_dataframe(background, require_scores=False)
    with pytest.raises(SchemaError):
        validate_dataframe(background)


def test_clean_strips_and_dedupes(raw_df):
    dirty = pd.concat([raw_df, raw_df.head(5)], ignore_index=True)
    dirty["gender"] = dirty["gender"].str.upper() + "  "
    out = clean(dirty)
    assert len(out) == len(raw_df)
    assert set(out["gender"]) <= {"female", "male"}


def test_load_clean_from_path(synthetic_csv):
    df = load_clean(synthetic_csv)
    assert df.shape[1] == 8 and len(df) > 0


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_clean(tmp_path / "nope.csv")


@pytest.mark.skipif(not RAW.exists(), reason="real dataset not present")
def test_real_dataset_validates():
    df = load_clean(RAW)
    assert len(df) == 1000
