"""Тесты сборщика метрик."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from monitoring import MetricsCollector


def test_initial_state():
    m = MetricsCollector(model_version="test")
    snap = m.snapshot()
    assert snap["requests_total"] == 0
    assert snap["errors_total"] == 0
    assert snap["cold_start_total"] == 0


def test_record_request():
    m = MetricsCollector(model_version="test")
    m.record_request(latency_seconds=0.05, is_cold_start=False, n_recommendations=10)
    snap = m.snapshot()
    assert snap["requests_total"] == 1
    assert snap["errors_total"] == 0
    assert snap["latency_avg_ms"] == pytest.approx(50.0, abs=1.0)


def test_record_error():
    m = MetricsCollector(model_version="test")
    m.record_request(latency_seconds=0.0, error=True)
    snap = m.snapshot()
    assert snap["requests_total"] == 1
    assert snap["errors_total"] == 1


def test_cold_start_counter():
    m = MetricsCollector(model_version="test")
    m.record_request(latency_seconds=0.01, is_cold_start=True)
    m.record_request(latency_seconds=0.01, is_cold_start=True)
    m.record_request(latency_seconds=0.01, is_cold_start=False)
    assert m.snapshot()["cold_start_total"] == 2


def test_prometheus_text_format():
    m = MetricsCollector(model_version="v1")
    m.record_request(latency_seconds=0.02)
    text = m.prometheus_text()
    assert "recommend_requests_total" in text
    assert "recommend_latency_seconds" in text
    assert 'model="v1"' in text
