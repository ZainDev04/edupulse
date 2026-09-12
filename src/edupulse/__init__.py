"""EduPulse - Student Performance Intelligence Platform.

An end-to-end, production-grade machine learning system for predicting student
outcomes from demographic and academic background data.

Public API
----------
- :func:`edupulse.pipeline.run_pipeline` - train, tune, evaluate and register a model
- :class:`edupulse.models.registry.ModelRegistry` - load persisted models
- :mod:`edupulse.tasks` - the task registry (at_risk, math_score, performance_level)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("edupulse")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "1.2.0"

__all__ = ["__version__"]
