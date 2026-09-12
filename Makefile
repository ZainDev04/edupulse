.PHONY: install install-dev data validate eda train train-fast test lint format serve dashboard mlflow-ui runs docker clean notebooks

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . && pre-commit install

validate:
	edupulse data validate

eda:
	edupulse eda

train:
	edupulse train --all --trials 50

train-fast:
	edupulse train --all --no-tune --repeats 1

test:
	$(PY) -m pytest --cov=edupulse --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

serve:
	edupulse serve --reload

dashboard:
	edupulse dashboard

mlflow-ui:
	edupulse ui

runs:
	edupulse runs

notebooks:
	$(PY) scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage build dist *.egg-info
