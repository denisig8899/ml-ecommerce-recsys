"""FastAPI-сервис рекомендаций товаров."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from models import Recommender
from monitoring import MetricsCollector

from .schemas import HealthResponse, RecommendRequest, RecommendResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/als_model.pkl"))
ARTIFACT_PATH = Path(os.getenv("ARTIFACT_PATH", "models/artifact.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", MODEL_PATH.stem)

_recommender: Recommender | None = None
_metrics: MetricsCollector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _recommender, _metrics

    for path in (MODEL_PATH, ARTIFACT_PATH):
        if not path.exists():
            raise RuntimeError(f"Артефакт не найден: {path}")

    _recommender = Recommender(MODEL_PATH, ARTIFACT_PATH)
    _metrics = MetricsCollector(model_version=MODEL_VERSION)
    yield


app = FastAPI(
    title="E-commerce Recommender",
    description="Персонализированные рекомендации товаров на основе ALS.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    art = _recommender.artifact
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        model_type=art.model_type,
        n_users=art.n_users,
        n_items=art.n_items,
        top_k=art.top_k,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    art = _recommender.artifact
    is_cold = request.visitor_id not in art.user_map

    try:
        recs = _recommender.recommend(request.visitor_id, n=request.n)
    except Exception as exc:
        _metrics.record_request(latency_seconds=0.0, error=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = time.perf_counter() - t0
    _metrics.record_request(
        latency_seconds=latency,
        is_cold_start=is_cold,
        n_recommendations=len(recs),
    )
    return RecommendResponse(
        visitor_id=request.visitor_id,
        recommendations=recs,
        is_cold_start=is_cold,
        model_version=MODEL_VERSION,
        latency_ms=round(latency * 1000, 2),
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return _metrics.prometheus_text()


@app.get("/metrics/snapshot")
def metrics_snapshot() -> dict[str, Any]:
    return _metrics.snapshot()
