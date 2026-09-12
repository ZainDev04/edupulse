"""Fairness mitigation by per-group thresholds."""

import json

import numpy as np
import pandas as pd
import pytest

from edupulse.models.mitigation import GroupThresholds, fit_group_thresholds, overall_metrics, threshold_at_recall
from edupulse.models.train import optimise_threshold
from edupulse.serving import PredictionService

STUDENT = {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "high school",
    "lunch": "free/reduced",
    "test_preparation_course": "none",
}


def _scores(seed=0, n=600):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(y * 0.35 + rng.normal(0.3, 0.2, n), 0, 1)
    return y, p


def test_threshold_at_recall_matches_global_rule():
    y, p = _scores()
    t_global, _ = optimise_threshold(y, p, target_recall=0.8)
    assert threshold_at_recall(y, p, 0.8) == pytest.approx(t_global)


def test_fit_group_thresholds_equalises_recall_and_guards_small_groups():
    y, p = _scores(1)
    groups = pd.Series(np.where(np.arange(len(y)) % 3 == 0, "a", "b"))
    # shift group "a" probabilities down so one global threshold would under-serve it
    p = np.where(groups == "a", p * 0.6, p)
    gt = fit_group_thresholds(y, p, groups, attribute="g", target_recall=0.8, default=0.4)
    assert set(gt.thresholds) == {"a", "b"}
    assert gt.thresholds["a"] < gt.thresholds["b"]
    for g in ("a", "b"):
        assert gt.calibration[g]["recall"] >= 0.8
    pred = gt.predict(p, groups)
    for g in ("a", "b"):
        idx = (groups == g).to_numpy()
        assert overall_metrics(y[idx], pred[idx])["recall"] >= 0.8

    # a group with too few positives keeps the default threshold, unknown groups too
    tiny = pd.Series(["a"] * (len(y) - 5) + ["z"] * 5)
    gt2 = fit_group_thresholds(y, p, tiny, attribute="g", target_recall=0.8, default=0.42)
    assert gt2.thresholds["z"] == 0.42
    assert gt2.threshold_for("never-seen") == 0.42
    assert GroupThresholds.from_dict({**gt2.to_dict(), "junk": 1}) == gt2


def test_pipeline_records_group_thresholds_and_before_after_audit(trained_at_risk, fast_settings):
    gt = trained_at_risk.training.group_thresholds
    assert gt is not None and gt.attribute == fast_settings.fairness_attribute == "lunch"
    assert gt.default == pytest.approx(trained_at_risk.training.threshold)

    mit = trained_at_risk.fairness["mitigated"]
    assert mit["attribute"] == "lunch" and set(mit["thresholds"]) == set(gt.thresholds)
    assert {"overall_before", "overall_after", "groups", "summary"} <= mit.keys()
    assert "tpr_gap" in mit["summary"]["lunch"]
    assert mit["figure"] and mit["figure"].endswith("fairness_mitigation.png")

    meta = json.loads((trained_at_risk.artefact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["group_thresholds"]["attribute"] == "lunch"
    card = (trained_at_risk.artefact_dir / "model_card.md").read_text(encoding="utf-8")
    assert "Fairness mitigation: recall equalised across lunch" in card


def test_service_returns_both_flags(trained_at_risk, fast_settings):
    svc = PredictionService(fast_settings.models_dir)
    rec = svc.predict("at_risk", [STUDENT, {**STUDENT, "lunch": "standard"}])
    gt = trained_at_risk.training.group_thresholds
    for r, lunch in zip(rec, ("free/reduced", "standard"), strict=True):
        assert r["mitigated"]["attribute"] == "lunch"
        assert r["mitigated"]["threshold"] == pytest.approx(gt.thresholds[lunch])
        assert r["mitigated"]["at_risk"] == (r["probability"] >= gt.thresholds[lunch])
        assert r["at_risk"] == (r["probability"] >= r["threshold"])
    fair = svc.fairness("at_risk")
    assert fair["mitigated"]["attribute"] == "lunch"
    assert svc.info("at_risk")["group_thresholds"]["attribute"] == "lunch"
