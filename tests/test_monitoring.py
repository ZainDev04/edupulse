"""Drift monitoring: PSI, KS, the prediction log and the API endpoints."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from edupulse.monitoring import MIN_ROWS, PredictionLog, drift_report, psi_categorical, psi_numeric, score_drift
from edupulse.serving import PredictionService

STUDENT = {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "high school",
    "lunch": "free/reduced",
    "test_preparation_course": "none",
}


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.choice(["a", "b", "c"], 1000, p=[0.5, 0.3, 0.2]))
    same, table = psi_categorical(ref, ref)
    assert same == pytest.approx(0.0, abs=1e-6)
    assert set(table["bin"]) == {"a", "b", "c"}
    shifted, _ = psi_categorical(ref, pd.Series(["c"] * 500))
    assert shifted > 0.25

    num = pd.Series(rng.normal(60, 15, 2000))
    low, _ = psi_numeric(num, num.sample(500, random_state=1))
    assert low < 0.1
    high, table = psi_numeric(num, num + 30)
    assert high > 0.25 and len(table) == 10


def test_score_drift_reports_ks():
    rng = np.random.default_rng(2)
    ref = rng.beta(2, 5, 1000)
    out = score_drift(ref, rng.beta(2, 5, 300))
    assert out["status"] == "ok" and out["ks_pvalue"] > 0.01
    out = score_drift(ref, rng.beta(5, 2, 300))
    assert out["status"] == "alert" and out["ks_pvalue"] < 1e-6


def test_drift_report_needs_enough_rows(raw_df):
    small = drift_report(raw_df, raw_df.head(MIN_ROWS - 1), ["gender", "lunch"])
    assert small["status"] == "insufficient" and small["features"] == []
    ok = drift_report(raw_df, raw_df.sample(100, random_state=0), ["gender", "lunch", "math_score"])
    assert ok["status"] == "ok"
    assert {f["feature"] for f in ok["features"]} == {"gender", "lunch", "math_score"}
    assert {f["kind"] for f in ok["features"]} == {"categorical", "numeric"}
    shifted = raw_df.sample(100, random_state=0).assign(lunch="free/reduced")
    bad = drift_report(raw_df, shifted, ["gender", "lunch"])
    assert bad["status"] == "alert"
    assert next(f for f in bad["features"] if f["feature"] == "lunch")["status"] == "alert"


def test_prediction_log_is_bounded_per_task():
    log = PredictionLog(maxlen=5)
    log.append("at_risk", [{"x": i} for i in range(8)], [i / 10 for i in range(8)])
    log.append("math_score", [{"x": 1}], [50.0])
    assert log.count("at_risk") == 5 and log.count("math_score") == 1
    frame = log.frame("at_risk")
    assert list(frame["x"]) == [3, 4, 5, 6, 7] and {"_score", "_ts"} <= set(frame.columns)
    log.clear("math_score")
    assert log.count("math_score") == 0 and log.frame("math_score").empty
    log.clear()
    assert log.count("at_risk") == 0


def test_service_logs_predictions_and_reports_drift(trained_at_risk, fast_settings, monkeypatch):
    from edupulse.data.loader import load_clean
    from edupulse.features.engineering import engineer_features

    reference = engineer_features(load_clean(fast_settings.raw_data_path))
    monkeypatch.setattr("edupulse.serving._full_frame", lambda: reference)
    svc = PredictionService(fast_settings.models_dir)

    assert svc.drift("at_risk")["status"] == "insufficient"
    rows = reference.sample(200, random_state=3)[list(svc.model("at_risk").task.features)].to_dict(orient="records")
    svc.predict("at_risk", rows)
    assert svc.log.count("at_risk") == 200
    report = svc.drift("at_risk")
    assert report["source"] == "prediction_log" and report["n_current"] == 200
    assert report["status"] in {"ok", "warn"}
    assert report["scores"] is not None and 0 <= report["scores"]["ks_pvalue"] <= 1

    skewed = pd.DataFrame([{**STUDENT, "lunch": "free/reduced", "test_preparation_course": "none"}] * 40)
    report = svc.drift("at_risk", skewed)
    assert report["source"] == "provided" and report["status"] == "alert"
    assert svc.log.count("at_risk") == 200  # comparing a supplied frame does not touch the log


def test_monitoring_endpoints(trained_at_risk, fast_settings, monkeypatch):
    import api.main as api_main

    from edupulse.data.loader import load_clean
    from edupulse.features.engineering import engineer_features

    reference = engineer_features(load_clean(fast_settings.raw_data_path))
    monkeypatch.setattr("edupulse.serving._full_frame", lambda: reference)
    svc = PredictionService(fast_settings.models_dir)
    monkeypatch.setattr(api_main, "get_service", lambda: svc)
    with TestClient(api_main.app) as client:
        assert client.get("/monitoring/drift/at-risk").json()["status"] == "insufficient"
        assert client.get("/monitoring/drift/nope").status_code == 404
        batch = {"students": [STUDENT] * 25}
        assert client.post("/predict/at_risk/batch", json=batch).status_code == 200
        r = client.get("/monitoring/drift/at-risk")
        assert r.status_code == 200 and r.json()["n_current"] == 25 and r.json()["status"] == "alert"
        r = client.post("/monitoring/drift/at-risk", json={"students": [STUDENT] * 30})
        assert r.status_code == 200 and r.json()["source"] == "provided"
        assert client.delete("/monitoring/log/at-risk").status_code == 204
        assert client.get("/monitoring/drift/at-risk").json()["n_current"] == 0
