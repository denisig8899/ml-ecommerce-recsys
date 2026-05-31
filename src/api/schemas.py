"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    visitor_id: int = Field(..., description="ID пользователя (visitorid из events)")
    n: int = Field(10, ge=1, le=100, description="Количество рекомендаций")


class RecommendResponse(BaseModel):
    visitor_id: int
    recommendations: list[int] = Field(
        ..., description="Список itemid в порядке убывания релевантности"
    )
    is_cold_start: bool = Field(
        ..., description="True если пользователь не встречался при обучении"
    )
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_type: str
    n_users: int
    n_items: int
    top_k: int
