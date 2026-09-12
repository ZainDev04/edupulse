"""EduPulse command-line interface.

Examples
--------
    edupulse data validate
    edupulse eda
    edupulse train --task at_risk --trials 60
    edupulse train --all
    edupulse evaluate --task at_risk
    edupulse predict --task at_risk --gender female --lunch standard ...
    edupulse serve --port 8000
    edupulse dashboard
    edupulse runs --task at_risk
    edupulse ui
    edupulse drift --task at_risk --path new_intake.csv
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from edupulse import __version__
from edupulse.config import get_settings
from edupulse.tasks import TASKS

app = typer.Typer(
    help="EduPulse - student performance intelligence platform.", no_args_is_help=True, rich_markup_mode="rich"
)
data_app = typer.Typer(help="Dataset utilities.")
app.add_typer(data_app, name="data")


def _version_callback(value: bool):
    if value:
        typer.echo(f"edupulse {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version.")
    ] = False,
):
    """EduPulse CLI."""


# --------------------------------------------------------------------------- #
@data_app.command("validate")
def data_validate(path: Annotated[Path | None, typer.Option(help="CSV path (defaults to data/raw).")] = None):
    """Validate the dataset against the declared schema."""
    from edupulse.data import load_clean

    df = load_clean(path)
    typer.secho(f"OK - {len(df)} rows x {df.shape[1]} columns passed schema validation.", fg="green")


@data_app.command("profile")
def data_profile():
    """Print a compact statistical profile."""
    from edupulse.eda import profile
    from edupulse.pipeline import load_engineered

    p = profile(load_engineered())
    p.pop("categorical", None)
    typer.echo(json.dumps(p, indent=2, default=str))


# --------------------------------------------------------------------------- #
@app.command()
def eda():
    """Generate the EDA figures and profile under reports/."""
    from edupulse.eda import run_eda
    from edupulse.pipeline import load_engineered

    s = get_settings()
    prof = run_eda(load_engineered(s), s.figures_dir, s.reports_dir)
    typer.secho(f"EDA complete - {len(prof['figures'])} figures in {s.figures_dir / 'eda'}", fg="green")


@app.command()
def train(
    task: Annotated[str | None, typer.Option(help=f"Task name: {', '.join(TASKS)}")] = None,
    all_tasks: Annotated[bool, typer.Option("--all", help="Train every registered task.")] = False,
    trials: Annotated[int | None, typer.Option(help="Optuna trials (overrides settings).")] = None,
    no_tune: Annotated[bool, typer.Option("--no-tune", help="Skip hyper-parameter tuning.")] = False,
    folds: Annotated[int | None, typer.Option(help="CV folds.")] = None,
    repeats: Annotated[int | None, typer.Option(help="CV repeats.")] = None,
    version: Annotated[str | None, typer.Option(help="Explicit artefact version tag.")] = None,
    no_track: Annotated[bool, typer.Option("--no-track", help="Do not record the run in MLflow.")] = False,
):
    """Train, tune, evaluate, explain, audit and register model(s)."""
    from edupulse.pipeline import run_all, run_pipeline

    overrides = {"n_trials": trials, "tune": not no_tune, "cv_folds": folds, "cv_repeats": repeats}
    if no_track:
        overrides["tracking"] = False
    s = get_settings().model_copy(update={k: v for k, v in overrides.items() if v is not None})
    if not all_tasks and task is None:
        raise typer.BadParameter("Provide --task NAME or --all")
    if all_tasks:
        outputs = run_all(s, version=version)
    else:
        outputs = {task: run_pipeline(task, settings=s, version=version)}
    typer.echo("")
    for name, o in outputs.items():
        typer.secho(f"[{name}] {o.training.best_candidate} -> {o.headline}", fg="green")
        typer.echo(f"        artefacts: {o.artefact_dir}")
        if o.run_id:
            typer.echo(f"        mlflow run: {o.run_id}")


@app.command()
def evaluate(
    task: Annotated[str, typer.Option(help="Task name.")], version: Annotated[str | None, typer.Option()] = None
):
    """Print hold-out metrics of a registered model."""
    from edupulse.models.registry import ModelRegistry

    m = ModelRegistry().load(task, version)
    typer.echo(f"{m.task.name} v{m.metadata.version} ({m.metadata.model_name})")
    typer.echo(
        json.dumps({k: v for k, v in m.metadata.test_metrics.items() if not isinstance(v, (list, dict))}, indent=2)
    )


@app.command()
def leaderboard(
    task: Annotated[str, typer.Option(help="Task name.")], version: Annotated[str | None, typer.Option()] = None
):
    """Show the cross-validated model leaderboard."""
    import pandas as pd

    from edupulse.models.registry import ModelRegistry

    m = ModelRegistry().load(task, version)
    board = pd.read_csv(m.path / "leaderboard.csv")
    cols = ["rank", "model", f"{m.task.primary_metric}_mean", f"{m.task.primary_metric}_std", "fit_seconds"]
    typer.echo(board[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


@app.command()
def runs(
    task: Annotated[str | None, typer.Option(help="Filter by task name.")] = None,
    limit: Annotated[int, typer.Option(help="Number of runs to show.")] = 20,
):
    """List MLflow runs (newest first)."""
    from edupulse.models.tracking import mlflow_available, search_runs

    if not mlflow_available():
        typer.secho("mlflow is not installed. Run: pip install mlflow", fg="yellow")
        raise typer.Exit(1)
    df = search_runs(task, max_results=limit)
    if df.empty:
        typer.echo("No runs recorded yet. Train a model first.")
        return
    cols = ["run_id", "start_time", "run_name", "model", "cv_metric", "cv_score", "metrics.train_seconds"]
    view = df[[c for c in cols if c in df.columns]].rename(columns={"metrics.train_seconds": "train_s"}).copy()
    view["run_id"] = view["run_id"].str[:8]
    view["start_time"] = view["start_time"].dt.strftime("%Y-%m-%d %H:%M")
    typer.echo(view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    typer.echo("")
    typer.echo(f"Tracking URI: {df['tracking_uri'].iloc[0]}  (open with: edupulse ui)")


@app.command()
def ui(port: int = 5000, host: str = "127.0.0.1"):
    """Open the MLflow tracking UI on the local run store."""
    from edupulse.models.tracking import ExperimentTracker, mlflow_available

    if not mlflow_available():
        typer.secho("mlflow is not installed. Run: pip install mlflow", fg="yellow")
        raise typer.Exit(1)
    uri = ExperimentTracker(get_settings(), enabled=False).tracking_uri
    typer.echo(f"MLflow UI on http://{host}:{port} (store: {uri})")
    subprocess.run(
        [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri, "--host", host, "--port", str(port)],
        check=False,
    )


@app.command()
def models():
    """List registered models."""
    from edupulse.models.registry import ModelRegistry

    reg = ModelRegistry()
    for name, ver in reg.available().items():
        typer.echo(f"{name:20s} latest={ver}  versions={reg.versions(name)}")


@app.command()
def predict(
    task: Annotated[str, typer.Option(help="Task name.")],
    gender: str = "female",
    race_ethnicity: str = "group C",
    parental_level_of_education: str = "some college",
    lunch: str = "standard",
    test_preparation_course: str = "none",
    math_score: float | None = None,
    reading_score: float | None = None,
    writing_score: float | None = None,
    explain: bool = typer.Option(False, help="Include SHAP contributions."),
):
    """Score a single student from the command line."""
    from edupulse.serving import PredictionService

    svc = PredictionService()
    payload = {
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course,
        "math_score": math_score,
        "reading_score": reading_score,
        "writing_score": writing_score,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    result = svc.predict(task, [payload], explain=explain)[0]
    typer.echo(json.dumps(result, indent=2))


@app.command()
def drift(
    task: Annotated[str, typer.Option(help="Task name.")],
    path: Annotated[Path, typer.Option(help="CSV of new student rows to compare with the training data.")],
):
    """Report feature and score drift of a CSV against the training population."""
    import pandas as pd

    from edupulse.data.loader import load_clean
    from edupulse.serving import PredictionService

    current = load_clean(path)
    report = PredictionService().drift(task, current)
    typer.echo(
        f"{report['task']} v{report['model_version']}: {report['n_current']} rows vs {report['n_reference']} training rows"
    )
    if report["status"] == "insufficient":
        typer.secho(f"Need at least {report['min_rows']} rows.", fg="yellow")
        raise typer.Exit(1)
    rows = [{"feature": f["feature"], "psi": f["psi"], "status": f["status"]} for f in report["features"]]
    if report["scores"]:
        s = report["scores"]
        rows.append({"feature": "(model output)", "psi": s["psi"], "status": s["status"]})
    typer.echo(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if report["scores"]:
        s = report["scores"]
        typer.echo(f"KS on output: statistic {s['ks_statistic']:.3f}, p={s['ks_pvalue']:.3g}")
    colour = {"ok": "green", "warn": "yellow", "alert": "red"}[report["status"]]
    typer.secho(f"Overall: {report['status'].upper()}", fg=colour)


@app.command()
def serve(host: str | None = None, port: int | None = None, reload: bool = False):
    """Run the FastAPI inference server."""
    import uvicorn

    s = get_settings()
    uvicorn.run("api.main:app", host=host or s.api_host, port=port or s.api_port, reload=reload)


@app.command()
def dashboard(port: int = 8501):
    """Launch the Streamlit dashboard."""
    app_path = get_settings().project_root / "app" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)], check=False)


if __name__ == "__main__":  # pragma: no cover
    app()
