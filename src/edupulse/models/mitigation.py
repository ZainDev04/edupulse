"""Fairness mitigation by group-specific decision thresholds (equal opportunity).

The at-risk audit shows that one global threshold catches at-risk students at
very different rates across lunch groups, because base rates differ and the
score distributions sit at different places. A cheap and transparent fix is to
pick the threshold per group so that every group reaches the same target
recall on out-of-fold predictions. The model itself is untouched; only the
cut-off moves, and the plain global threshold stays available for comparison.

This is post-processing in the sense of Hardt et al. (2016), restricted to
equalising true-positive rates. Precision and selection rates are allowed to
differ, and the model card reports what the change costs overall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from edupulse.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class GroupThresholds:
    """Per-group decision thresholds on ``attribute``; unknown groups use ``default``."""

    attribute: str
    thresholds: dict[str, float]
    default: float
    target_recall: float
    method: str = "equal opportunity: per-group threshold at target recall on out-of-fold probabilities"
    calibration: dict[str, dict[str, float]] = field(default_factory=dict)

    def threshold_for(self, group: Any) -> float:
        return self.thresholds.get(str(group), self.default)

    def thresholds_for(self, groups: pd.Series | np.ndarray) -> np.ndarray:
        return np.array([self.threshold_for(g) for g in np.asarray(groups)], dtype=float)

    def predict(self, proba: np.ndarray, groups: pd.Series | np.ndarray) -> np.ndarray:
        return (np.asarray(proba, dtype=float) >= self.thresholds_for(groups)).astype(int)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupThresholds:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def threshold_at_recall(y: np.ndarray, proba: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose recall is still at least ``target_recall``.

    Same rule as :func:`edupulse.models.train.optimise_threshold`: walk the observed
    probabilities in ascending order and stop at the first one that drops below target.
    """
    y = np.asarray(y)
    grid = np.unique(np.round(np.asarray(proba, dtype=float), 4))
    best = float(grid[0])
    for t in grid:
        if recall_score(y, (proba >= t).astype(int), zero_division=0) >= target_recall:
            best = float(t)
        else:
            break
    return best


def fit_group_thresholds(
    y: pd.Series | np.ndarray,
    proba: np.ndarray,
    groups: pd.Series,
    *,
    attribute: str,
    target_recall: float,
    default: float,
    min_positives: int = 10,
) -> GroupThresholds:
    """Choose one threshold per group of ``groups`` so each reaches ``target_recall``.

    Groups with fewer than ``min_positives`` positive cases keep the global ``default``
    threshold, since a recall estimate on a handful of students is noise.
    """
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    groups = pd.Series(np.asarray(groups)).astype(str)
    thresholds: dict[str, float] = {}
    calibration: dict[str, dict[str, float]] = {}
    for g in sorted(groups.unique()):
        idx = (groups == g).to_numpy()
        yg, pg = y[idx], proba[idx]
        n_pos = int(yg.sum())
        if n_pos < min_positives:
            t = default
            log.info("Group %s=%s has %d positives; keeping the global threshold", attribute, g, n_pos)
        else:
            t = threshold_at_recall(yg, pg, target_recall)
        pred = (pg >= t).astype(int)
        thresholds[g] = t
        calibration[g] = {
            "n": int(idx.sum()),
            "n_positive": n_pos,
            "threshold": t,
            "recall": float(recall_score(yg, pred, zero_division=0)),
            "precision": float(precision_score(yg, pred, zero_division=0)),
            "flagged_rate": float(pred.mean()),
        }
    log.info("Group thresholds on %s: %s", attribute, {k: round(v, 3) for k, v in thresholds.items()})
    return GroupThresholds(
        attribute=attribute,
        thresholds=thresholds,
        default=default,
        target_recall=target_recall,
        calibration=calibration,
    )


def overall_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y, pred = np.asarray(y), np.asarray(pred)
    return {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "flagged_rate": float(pred.mean()),
        "fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
    }
