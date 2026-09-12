# Serving image for the hosted API (Render, or any Docker host). Uses the committed pipelines;
# nothing is trained at build time, so the served numbers match the model cards.
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    EDUPULSE_TRACKING=false EDUPULSE_LOG_LEVEL=INFO
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* \
    && useradd -m user
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt
COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user data/raw ./data/raw
COPY --chown=user models ./models
RUN pip install --no-deps -e . && mkdir -p data/processed reports/figures && chown -R user /app
USER user
EXPOSE 8000
# Render injects PORT; default to 8000 elsewhere
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
