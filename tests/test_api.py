"""API + serving tests (use the session-trained at_risk model in a temp registry)."""

import pytest
from fastapi.testclient import TestClient

from edupulse.serving import PredictionService, risk_band

STUDENT = {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "high school",
    "lunch": "free/reduced",
    "test_preparation_course": "none",
}


@pytest.fixture(scope="module")
def client(trained_at_risk, fast_settings, monkeypatch_module):
    import api.main as api_main

    svc = PredictionService(fast_settings.models_dir)
    monkeypatch_module.setattr(api_main, "get_service", lambda: svc)
    monkeypatch_module.setattr("edupulse.serving._background_frame", lambda: _bg(fast_settings))
    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _bg(settings):
    from edupulse.data.loader import load_clean
    from edupulse.features.engineering import engineer_features

    return engineer_features(load_clean(settings.raw_data_path)).head(100)


def test_risk_band_thresholds():
    assert (
        risk_band(0.1) == "low"
        and risk_band(0.4) == "moderate"
        and risk_band(0.7) == "high"
        and risk_band(0.95) == "critical"
    )


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models"]["at_risk"] is not None


def test_models_listing(client):
    r = client.get("/models")
    assert r.status_code == 200 and "at_risk" in r.json()
    r = client.get("/models/at-risk")
    assert r.json()["kind"] == "binary" and r.json()["threshold"] is not None
    assert client.get("/models/nope").status_code == 404


def test_predict_at_risk(client):
    r = client.post("/predict/at-risk", json=STUDENT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability"] <= 1 and isinstance(body["at_risk"], bool)
    assert body["risk_band"] in {"low", "moderate", "high", "critical"}
    assert body["mitigated"]["attribute"] == "lunch" and isinstance(body["mitigated"]["at_risk"], bool)
    assert "X-Process-Time-ms" in r.headers


def test_predict_at_risk_with_explanation(client):
    r = client.post("/predict/at-risk?explain=true", json=STUDENT)
    assert r.status_code == 200, r.text
    ex = r.json()["explanation"]
    assert ex and {"feature", "contribution"} <= ex[0].keys()


def test_predict_rejects_invalid_category(client):
    r = client.post("/predict/at-risk", json={**STUDENT, "gender": "robot"})
    assert r.status_code == 422


def test_predict_missing_model_gives_503(client):
    r = client.post("/predict/math-score", json={**STUDENT, "reading_score": 70, "writing_score": 68})
    assert r.status_code == 503


def test_batch_predict(client):
    r = client.post(
        "/predict/at-risk/batch", json={"students": [STUDENT, {**STUDENT, "test_preparation_course": "completed"}]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2 and len(body["predictions"]) == 2
    # completing test prep should not increase risk
    assert body["predictions"][1]["probability"] <= body["predictions"][0]["probability"] + 1e-9


def test_batch_limits(client):
    r = client.post("/predict/at-risk/batch", json={"students": []})
    assert r.status_code == 422


def test_service_ignores_null_optional_fields(trained_at_risk, fast_settings):
    svc = PredictionService(fast_settings.models_dir)
    out = svc.predict("at_risk", [{**STUDENT, "math_score": None, "reading_score": None, "writing_score": None}])
    assert 0 <= out[0]["probability"] <= 1


def test_meta_endpoints(client):
    r = client.get("/models/at-risk/leaderboard")
    assert r.status_code == 200 and r.json()["rows"][0]["rank"] == 1
    r = client.get("/models/at-risk/fairness")
    assert r.status_code == 200 and "summary" in r.json()
    r = client.get("/models/at-risk/importance")
    assert r.status_code == 200 and r.json()["shap"]
    r = client.get("/stats")
    assert r.status_code == 200 and 0 < r.json()["at_risk_rate"] < 1 and "lunch" in r.json()["by_attribute"]
    assert client.get("/models/nope/leaderboard").status_code == 404
