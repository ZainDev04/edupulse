"""Conformal prediction intervals for the regression task."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge

from edupulse.models.conformal import ConformalInterval, calibrate, conformal_quantile, evaluate_coverage
from edupulse.serving import PredictionService

STUDENT = {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "high school",
    "lunch": "free/reduced",
    "test_preparation_course": "none",
    "reading_score": 70,
    "writing_score": 68,
}


def test_conformal_quantile_uses_finite_sample_correction():
    residuals = np.arange(1, 11, dtype=float)  # |r| = 1..10, n = 10
    # ceil(11 * 0.9) = 10 -> the 10th smallest = 10
    assert conformal_quantile(residuals, alpha=0.1) == 10.0
    # ceil(11 * 0.5) = 6 -> the 6th smallest = 6
    assert conformal_quantile(-residuals, alpha=0.5) == 6.0
    # alpha so small that k > n: fall back to the largest residual
    assert conformal_quantile(residuals, alpha=0.01) == 10.0
    with pytest.raises(ValueError):
        conformal_quantile(residuals, alpha=1.5)
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), alpha=0.1)


def test_interval_predict_clips_and_round_trips():
    iv = ConformalInterval(alpha=0.2, quantile=7.5, n_calibration=50, lower_bound=0.0, upper_bound=100.0)
    lower, upper = iv.predict(np.array([3.0, 50.0, 98.0]))
    assert lower.tolist() == [0.0, 42.5, 90.5]
    assert upper.tolist() == [10.5, 57.5, 100.0]
    assert iv.confidence == pytest.approx(0.8)
    restored = ConformalInterval.from_dict({**iv.to_dict(), "extra_key": "ignored"})
    assert restored == iv


def test_calibrate_reaches_nominal_coverage_on_synthetic_data():
    rng = np.random.default_rng(1)
    n = 1200
    X = rng.normal(size=(n, 3))
    y = X @ np.array([3.0, -2.0, 1.0]) + rng.normal(scale=2.0, size=n)
    import pandas as pd

    Xdf = pd.DataFrame(X, columns=["a", "b", "c"])
    train, test = slice(0, 800), slice(800, n)
    iv = calibrate(Ridge(), Xdf.iloc[train], pd.Series(y[train]), alpha=0.1, n_splits=5, bounds=None)
    model = Ridge().fit(Xdf.iloc[train], y[train])
    cov = evaluate_coverage(iv, y[test], model.predict(Xdf.iloc[test]))
    assert 0.86 <= cov["coverage"] <= 0.96
    assert cov["mean_width"] == pytest.approx(2 * iv.quantile)
    assert iv.n_calibration == 800


def test_pipeline_registers_interval_and_reports_coverage(trained_math_score):
    out, settings = trained_math_score
    iv = out.training.conformal
    assert iv is not None and iv.alpha == pytest.approx(settings.conformal_alpha)
    assert iv.quantile > 0 and iv.n_calibration == 320

    m = out.test_metrics
    assert {"interval_coverage", "interval_nominal", "interval_width"} <= m.keys()
    assert m["interval_nominal"] == pytest.approx(0.9)
    assert m["interval_coverage"] >= 0.8  # 80 test rows, so allow sampling noise around the 90% target
    assert "interval" in out.headline

    meta = json.loads((out.artefact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["conformal"]["quantile"] == pytest.approx(iv.quantile)
    card = (out.artefact_dir / "model_card.md").read_text(encoding="utf-8")
    assert "Prediction intervals (conformal)" in card
    assert "intervals" in meta["figures"]


def test_service_and_api_return_interval(trained_math_score):
    out, settings = trained_math_score
    svc = PredictionService(settings.models_dir)
    rec = svc.predict("math_score", [STUDENT])[0]
    assert rec["lower"] <= rec["prediction"] <= rec["upper"]
    assert rec["confidence"] == pytest.approx(0.9)
    assert rec["upper"] - rec["lower"] <= 2 * out.training.conformal.quantile + 1e-9
    assert svc.info("math_score")["conformal"]["n_calibration"] == 320

    import api.main as api_main

    original = api_main.get_service
    api_main.get_service = lambda: svc
    try:
        with TestClient(api_main.app) as client:
            r = client.post("/predict/math-score", json=STUDENT)
    finally:
        api_main.get_service = original
    assert r.status_code == 200
    body = r.json()
    assert body["lower"] <= body["prediction"] <= body["upper"] and body["confidence"] == pytest.approx(0.9)
