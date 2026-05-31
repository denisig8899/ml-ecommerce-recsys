FROM python:3.12-slim

WORKDIR /app

COPY docker/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

ARG MODEL_PATH=models/als_model.pkl
ARG ARTIFACT_PATH=models/artifact.pkl

ENV MODEL_PATH=models/als_model.pkl
ENV ARTIFACT_PATH=models/artifact.pkl
ENV MODEL_VERSION=als_v1
ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
