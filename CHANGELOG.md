# Changelog

Notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.1.0] - 2026-09-12

### Added
- MLflow experiment tracking behind the model registry (`edupulse.models.tracking`). Every `run_pipeline` call records settings, tuned parameters, CV leaderboard scores, the Optuna trial curve, hold-out metrics, fairness gaps, figures, the model card and the fitted pipeline (cloudpickle) to a SQLite store under `mlruns/`. The run id is written into `metadata.json`, the model card and `/models/{task}`.
- `edupulse runs` lists recorded runs; `edupulse ui` opens the MLflow UI; `edupulse train --no-track` or `EDUPULSE_TRACKING=false` switches tracking off. Missing `mlflow` degrades to a no-op with a warning.
- `mlflow` service in docker-compose on port 5000, `tracking` extra in `pyproject.toml`, 6 new tests (48 total).

## [1.0.0] - 2026-09-12

### Added
- `edupulse` Python package: schema validation, feature engineering, task registry, model zoo (10 families including XGBoost and LightGBM), repeated-CV leaderboard, Optuna tuning, recall-targeted threshold selection.
- Three prediction tasks: `at_risk` (early warning from background only), `math_score` (regression), `performance_level` (the original coursework task, documented as a leakage case study).
- Explainability: SHAP (tree, linear and kernel explainers) with one-hot aggregation, permutation importance, beeswarm plots.
- Fairness audit: subgroup metrics, demographic-parity and equal-opportunity gaps, disparate-impact ratio.
- Versioned model registry with generated model cards.
- FastAPI inference service (single, batch and explained predictions) and a Streamlit dashboard.
- Typer CLI: `edupulse data validate | eda | train | evaluate | leaderboard | predict | serve | dashboard`.
- Notebooks generated from `scripts/build_notebooks.py` and executed.
- Docker multi-stage image, docker-compose, GitHub Actions CI (lint, test matrix, training smoke test, Docker build), pre-commit.
- 42 pytest tests.

### Changed
- Replaced the original single-notebook random forest project (kept under `legacy/`) with a modular codebase.
