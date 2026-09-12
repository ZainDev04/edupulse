"""Drift monitoring on the inference stream.

Two questions matter once a model is serving: do the students being scored
still look like the students it was trained on, and do the scores it emits
still have the same distribution? This module answers both with plain,
explainable statistics:

- **Population stability index (PSI)** per input feature, computed on category
  frequencies for nominal columns and on reference-quantile bins for numeric
  ones. The usual reading applies: below 0.10 is stable, 0.10 to 0.25 is worth
  a look, above 0.25 is a material shift.
- **Kolmogorov-Smirnov** two-sample test on the model's output (probability of
  being at risk, or the predicted score), plus its PSI.

:class:`PredictionLog` is a bounded, thread-safe ring buffer that the
:class:`~edupulse.serving.PredictionService` appends to on every prediction, so
the API can report drift on the last few thousand requests without a database.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN = 0.10
PSI_ALERT = 0.25
MIN_ROWS = 20
EPS = 1e-6


def _status(psi_value: float) -> str:
    if psi_value >= PSI_ALERT:
        return "alert"
    if psi_value >= PSI_WARN:
        return "warn"
    return "ok"


def psi_categorical(reference: pd.Series, current: pd.Series) -> tuple[float, pd.DataFrame]:
    """PSI over category shares; also returns the per-category share table."""
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)
    cats = sorted(set(ref.index) | set(cur.index))
    r = np.array([ref.get(c, 0.0) for c in cats]) + EPS
    c = np.array([cur.get(c, 0.0) for c in cats]) + EPS
    contrib = (c - r) * np.log(c / r)
    table = pd.DataFrame({"bin": cats, "reference": r - EPS, "current": c - EPS, "contribution": contrib})
    return float(contrib.sum()), table


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """PSI over ``bins`` quantile bins of the reference distribution."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference column
        edges = np.array([ref.min() - 0.5, ref.max() + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / max(len(ref), 1) + EPS
    c = np.histogram(cur, bins=edges)[0] / max(len(cur), 1) + EPS
    contrib = (c - r) * np.log(c / r)
    labels = [f"{lo:.0f} to {hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    table = pd.DataFrame({"bin": labels, "reference": r - EPS, "current": c - EPS, "contribution": contrib})
    return float(contrib.sum()), table


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """PSI per feature with the two or three bins that moved most."""
    rows = []
    for f in features:
        if f not in reference.columns or f not in current.columns:
            continue
        numeric = pd.api.types.is_numeric_dtype(reference[f])
        value, table = (psi_numeric if numeric else psi_categorical)(reference[f], current[f])
        table["shift"] = table["current"] - table["reference"]
        movers = table.reindex(table["shift"].abs().sort_values(ascending=False).index).head(3)
        rows.append(
            {
                "feature": f,
                "kind": "numeric" if numeric else "categorical",
                "psi": value,
                "status": _status(value),
                "top_shifts": [
                    {"bin": str(m["bin"]), "reference": float(m["reference"]), "current": float(m["current"])}
                    for _, m in movers.iterrows()
                ],
            }
        )
    return rows


def score_drift(reference_scores: np.ndarray, current_scores: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)
    value, _ = psi_numeric(pd.Series(ref), pd.Series(cur))
    ks = ks_2samp(ref, cur)
    return {
        "psi": value,
        "status": _status(value),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "reference_mean": float(ref.mean()),
        "current_mean": float(cur.mean()),
    }


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    *,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full report: per-feature PSI, output drift and an overall status."""
    if len(current) < MIN_ROWS:
        return {
            "status": "insufficient",
            "n_reference": int(len(reference)),
            "n_current": int(len(current)),
            "min_rows": MIN_ROWS,
            "features": [],
            "scores": None,
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    feats = feature_drift(reference, current, features)
    scores = (
        score_drift(reference_scores, current_scores)
        if reference_scores is not None and current_scores is not None and len(current_scores) >= MIN_ROWS
        else None
    )
    statuses = [f["status"] for f in feats] + ([scores["status"]] if scores else [])
    overall = "alert" if "alert" in statuses else "warn" if "warn" in statuses else "ok"
    return {
        "status": overall,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "min_rows": MIN_ROWS,
        "features": feats,
        "scores": scores,
        "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@dataclass
class PredictionLog:
    """Bounded in-memory record of scored rows, per task."""

    maxlen: int = 5000
    _rows: dict[str, deque] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, task_name: str, rows: list[dict[str, Any]], scores: list[float]) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            buf = self._rows.setdefault(task_name, deque(maxlen=self.maxlen))
            for row, score in zip(rows, scores, strict=True):
                buf.append({**row, "_score": float(score), "_ts": now})

    def frame(self, task_name: str) -> pd.DataFrame:
        with self._lock:
            rows = list(self._rows.get(task_name, ()))
        return pd.DataFrame(rows)

    def count(self, task_name: str) -> int:
        with self._lock:
            return len(self._rows.get(task_name, ()))

    def clear(self, task_name: str | None = None) -> None:
        with self._lock:
            if task_name is None:
                self._rows.clear()
            else:
                self._rows.pop(task_name, None)
