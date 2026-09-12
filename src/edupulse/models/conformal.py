"""Conformal prediction intervals for the regression task.

A point estimate of a math score is of limited use without a sense of how far
off it might be. Split conformal prediction turns any regressor into one that
returns an interval with a guaranteed marginal coverage, under the assumption
that new students are exchangeable with the calibration students.

The calibration residuals come from out-of-fold predictions on the training
split (cross-conformal), so no extra hold-out is needed and the test split
stays untouched for reporting. The interval is symmetric and constant-width:

    [prediction - q, prediction + q]

where ``q`` is the finite-sample-corrected ``(1 - alpha)`` quantile of the
absolute out-of-fold residuals. Empirical coverage on the test split is
reported next to the nominal level so the guarantee can be checked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from edupulse.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ConformalInterval:
    """A calibrated half-width ``quantile`` for a nominal ``1 - alpha`` coverage."""

    alpha: float
    quantile: float
    n_calibration: int
    method: str = "cross-conformal, absolute out-of-fold residuals"
    lower_bound: float | None = None
    upper_bound: float | None = None

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    def predict(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` arrays around ``point``, clipped to the declared bounds."""
        point = np.asarray(point, dtype=float)
        lower, upper = point - self.quantile, point + self.quantile
        if self.lower_bound is not None:
            lower = np.maximum(lower, self.lower_bound)
        if self.upper_bound is not None:
            upper = np.minimum(upper, self.upper_bound)
        return lower, upper

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConformalInterval:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected quantile of the absolute residuals.

    Uses ``ceil((n + 1)(1 - alpha)) / n`` (Vovk et al.), which gives coverage of at
    least ``1 - alpha`` for exchangeable data rather than approximately that.
    """
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = len(r)
    if n == 0:
        raise ValueError("Cannot calibrate on zero residuals")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:  # too few residuals for this alpha: the interval has to cover everything seen
        return float(r[-1])
    return float(r[k - 1])


def calibrate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    alpha: float = 0.1,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int | None = None,
    bounds: tuple[float, float] | None = (0.0, 100.0),
) -> ConformalInterval:
    """Calibrate an interval for ``pipeline`` from out-of-fold residuals on ``(X, y)``."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = cross_val_predict(clone(pipeline), X, y, cv=cv, n_jobs=n_jobs)
    residuals = np.asarray(y, dtype=float) - np.asarray(oof, dtype=float)
    q = conformal_quantile(residuals, alpha)
    lo, hi = bounds if bounds else (None, None)
    interval = ConformalInterval(alpha=alpha, quantile=q, n_calibration=len(residuals), lower_bound=lo, upper_bound=hi)
    log.info("Conformal half-width %.2f for %.0f%% coverage (%d calibration residuals)", q, 100 * (1 - alpha), len(y))
    return interval


def evaluate_coverage(interval: ConformalInterval, y_true: np.ndarray, point: np.ndarray) -> dict[str, float]:
    """Empirical coverage and mean width of the intervals on a held-out set."""
    y_true = np.asarray(y_true, dtype=float)
    lower, upper = interval.predict(point)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(covered.mean()),
        "nominal_coverage": float(interval.confidence),
        "mean_width": float(np.mean(upper - lower)),
    }
