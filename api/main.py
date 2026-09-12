"""EduPulse REST API (FastAPI).

Run locally::

    edupulse serve            # or: uvicorn api.main:app --reload

Interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from edupulse import __version__
from edupulse.data.schema import ETHNICITIES, GENDERS, LUNCH, PARENTAL_EDUCATION, TEST_PREP
from edupulse.logging_utils import get_logger
from edupulse.serving import PredictionService, get_service
from edupulse.tasks import TASKS

log = get_logger("edupulse.api")

Gender = Literal[GENDERS]  # type: ignore[valid-type]
Ethnicity = Literal[ETHNICITIES]  # type: ignore[valid-type]
ParentalEducation = Literal[PARENTAL_EDUCATION]  # type: ignore[valid-type]
Lunch = Literal[LUNCH]  # type: ignore[valid-type]
TestPrep = Literal[TEST_PREP]  # type: ignore[valid-type]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class StudentBackground(BaseModel):
    """Background information available before any exam."""

    gender: Gender = Field(..., examples=["female"])
    race_ethnicity: Ethnicity = Field(..., examples=["group C"])
    parental_level_of_education: ParentalEducation = Field(..., examples=["some college"])
    lunch: Lunch = Field(..., examples=["standard"])
    test_preparation_course: TestPrep = Field(..., examples=["none"])

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


class StudentFull(StudentBackground):
    """Background + exam scores (0-100)."""

    math_score: float | None = Field(None, ge=0, le=100, examples=[68])
    reading_score: float | None = Field(None, ge=0, le=100, examples=[72])
    writing_score: float | None = Field(None, ge=0, le=100, examples=[70])


class BatchRequest(BaseModel):
    students: list[StudentFull] = Field(..., min_length=1, max_length=1000)
    explain: bool = False


class Contribution(BaseModel):
    feature: str
    value: Any
    contribution: float


class MitigatedFlag(BaseModel):
    attribute: str = Field(description="Sensitive attribute whose groups have equalised recall")
    threshold: float = Field(description="Decision threshold for this student's group")
    at_risk: bool


class AtRiskPrediction(BaseModel):
    probability: float
    at_risk: bool
    threshold: float
    risk_band: str
    mitigated: MitigatedFlag | None = Field(None, description="Flag under per-group thresholds with equalised recall")
    model_version: str
    explanation: list[Contribution] | None = None


class ScorePrediction(BaseModel):
    prediction: float
    lower: float | None = Field(None, description="Lower end of the conformal prediction interval")
    upper: float | None = Field(None, description="Upper end of the conformal prediction interval")
    confidence: float | None = Field(None, description="Nominal coverage of the interval, e.g. 0.9")
    model_version: str
    explanation: list[Contribution] | None = None


class LevelPrediction(BaseModel):
    label: str
    probabilities: dict[str, float]
    model_version: str
    explanation: list[Contribution] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict[str, str | None]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = get_service()
    for t in svc.available_tasks():
        try:
            svc.model(t)
        except Exception as exc:  # pragma: no cover
            log.warning("Could not pre-load %s: %s", t, exc)
    app.state.service = svc
    yield


app = FastAPI(
    title="EduPulse API",
    version=__version__,
    description=(
        "Student performance intelligence: early-warning risk scoring, cross-subject score "
        "prediction and performance tiering - with SHAP explanations on demand."
    ),
    lifespan=lifespan,
    contact={"name": "Shaikh Muhammad Zain"},
    license_info={"name": "MIT"},
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def not_found_handler(_: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _svc(request: Request) -> PredictionService:
    return getattr(request.app.state, "service", None) or get_service()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request):
    svc = _svc(request)
    return HealthResponse(status="ok", version=__version__, models={t: svc.registry.latest_version(t) for t in TASKS})


@app.get("/models", tags=["meta"])
def list_models(request: Request):
    svc = _svc(request)
    return {t: svc.info(t) for t in svc.available_tasks()}


@app.get("/models/{task}", tags=["meta"])
def model_info(task: str, request: Request):
    if task.replace("-", "_") not in TASKS:
        raise HTTPException(404, f"Unknown task '{task}'")
    return _svc(request).info(task)


@app.get("/models/{task}/leaderboard", tags=["meta"])
def model_leaderboard(task: str, request: Request):
    """Cross-validation leaderboard of every candidate model for the task."""
    if task.replace("-", "_") not in TASKS:
        raise HTTPException(404, f"Unknown task '{task}'")
    return {"task": task.replace("-", "_"), "rows": _svc(request).leaderboard(task)}


@app.get("/models/{task}/fairness", tags=["meta"])
def model_fairness(task: str, request: Request):
    """Subgroup metrics and parity gaps from the hold-out audit."""
    if task.replace("-", "_") not in TASKS:
        raise HTTPException(404, f"Unknown task '{task}'")
    return _svc(request).fairness(task)


@app.get("/models/{task}/importance", tags=["meta"])
def model_importance(task: str, request: Request):
    """Global SHAP and permutation importance."""
    if task.replace("-", "_") not in TASKS:
        raise HTTPException(404, f"Unknown task '{task}'")
    return _svc(request).importance(task)


@app.get("/stats", tags=["meta"])
def dataset_stats(request: Request):
    """Headline dataset statistics and at-risk rates by background attribute."""
    return _svc(request).dataset_stats()


@app.post("/predict/at-risk", response_model=AtRiskPrediction, tags=["predict"])
def predict_at_risk(student: StudentBackground, request: Request, explain: bool = Query(False)):
    """Probability that a student will average below 60 - from background only."""
    return _svc(request).predict("at_risk", [student.model_dump(exclude_none=True)], explain=explain)[0]


@app.post("/predict/math-score", response_model=ScorePrediction, tags=["predict"])
def predict_math_score(student: StudentFull, request: Request, explain: bool = Query(False)):
    """Expected math score given background + reading/writing scores."""
    if student.reading_score is None or student.writing_score is None:
        raise HTTPException(422, "reading_score and writing_score are required for math-score prediction")
    return _svc(request).predict("math_score", [student.model_dump(exclude_none=True)], explain=explain)[0]


@app.post("/predict/performance-level", response_model=LevelPrediction, tags=["predict"])
def predict_performance_level(student: StudentFull, request: Request, explain: bool = Query(False)):
    """Low / medium / high tier from background + all three scores."""
    if None in (student.math_score, student.reading_score, student.writing_score):
        raise HTTPException(422, "math_score, reading_score and writing_score are all required")
    return _svc(request).predict("performance_level", [student.model_dump(exclude_none=True)], explain=explain)[0]


@app.post("/predict/{task}/batch", tags=["predict"])
def predict_batch(task: str, body: BatchRequest, request: Request):
    """Batch scoring (up to 1000 students per call)."""
    if task.replace("-", "_") not in TASKS:
        raise HTTPException(404, f"Unknown task '{task}'")
    rows = [s.model_dump(exclude_none=True) for s in body.students]
    preds = _svc(request).predict(task, rows, explain=body.explain)
    return {"task": task.replace("-", "_"), "n": len(preds), "predictions": preds}


@app.get("/", include_in_schema=False)
def root():
    return {"name": "EduPulse API", "version": __version__, "docs": "/docs", "health": "/health"}
