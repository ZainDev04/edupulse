"""Centralised configuration for EduPulse.

All paths and hyper-parameters are resolved from environment variables
(prefixed with ``EDUPULSE_``) or a ``.env`` file, falling back to sane
defaults so the project runs out-of-the-box.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings (12-factor style)."""

    model_config = SettingsConfigDict(
        env_prefix="EDUPULSE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Paths -----------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "StudentsPerformance.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"

    # --- Experiment ------------------------------------------------------
    random_state: int = 42
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    cv_folds: int = Field(5, ge=2, le=20)
    cv_repeats: int = Field(2, ge=1, le=10)
    n_jobs: int = -1

    # --- Hyper-parameter tuning -----------------------------------------
    tune: bool = True
    n_trials: int = Field(40, ge=1)
    tuning_timeout_s: int | None = 600

    # --- Domain rules ----------------------------------------------------
    pass_mark: int = Field(40, ge=0, le=100)
    at_risk_threshold: int = Field(60, ge=0, le=100)
    target_recall: float = Field(0.80, ge=0.1, le=1.0)
    medium_cutoff: int = 60
    high_cutoff: int = 80

    # --- Experiment tracking (MLflow) -----------------------------------
    tracking: bool = True
    tracking_uri: str = "sqlite:///" + (PROJECT_ROOT / "mlruns" / "mlflow.db").as_posix()
    tracking_artifacts: Path = PROJECT_ROOT / "mlruns" / "artifacts"
    tracking_experiment: str = "edupulse"

    # --- Serving ---------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        for d in (self.processed_dir, self.models_dir, self.reports_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached global settings instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
