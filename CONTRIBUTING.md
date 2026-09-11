# Contributing

The project is deliberately small. Please keep it that way.

## Setup

```bash
git clone https://github.com/ZainDev04/edupulse.git && cd edupulse
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
make install-dev                                 # dependencies, editable install, pre-commit hooks
edupulse data validate                           # sanity-check the dataset
make test
```

## Workflow

1. Create a branch: `git checkout -b feat/<short-name>`.
2. Add or update tests in `tests/`. Every module has a test file. The end-to-end pipeline test trains a fast model in a temporary directory.
3. Run `make lint test`. CI runs the same commands on Ubuntu and Windows.
4. Open a pull request that says why the change is needed.

## Adding a prediction task

Register a `Task` in `src/edupulse/tasks.py`. The CLI, API, dashboard, model registry and tests pick it up on their own. Add a route in `api/main.py` only if the request schema differs from the existing ones.

## Adding a model family

Add a `Candidate` (factory plus Optuna search space) in `src/edupulse/models/zoo.py`. Import optional dependencies inside `try/except` so the project still works without them.

## Style

- Python 3.10 or newer, type hints everywhere, docstrings on public functions.
- `ruff` decides formatting and linting (`make format`).
- Prefer pure functions. Keep I/O at the edges (`loader.py`, `registry.py`, `cli.py`).
