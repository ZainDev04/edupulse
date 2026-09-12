# Hugging Face Space image: serves the FastAPI app with the trained models shipped in the upload.
# No training at build time, so the numbers match the committed model cards exactly.
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    EDUPULSE_TRACKING=false EDUPULSE_LOG_LEVEL=INFO
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt
COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user data/raw ./data/raw
COPY --chown=user models ./models
RUN pip install --no-deps -e . && mkdir -p data/processed reports/figures && chown -R user /app
USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -fs http://localhost:7860/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
