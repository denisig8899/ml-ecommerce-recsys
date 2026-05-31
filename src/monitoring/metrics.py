"""Сбор метрик сервиса рекомендаций с экспортом в формат Prometheus."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Histogram:
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf"))
    _counts: list[int] = field(default_factory=list)
    _sum: float = 0.0
    _total: int = 0

    def __post_init__(self) -> None:
        self._counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        self._sum += value
        self._total += 1
        for i, b in enumerate(self.buckets):
            if value <= b:
                self._counts[i] += 1

    def prometheus_lines(self, name: str, labels: str = "") -> list[str]:
        lines = []
        label_str = f"{{{labels}}}" if labels else ""
        for bucket, count in zip(self.buckets, self._counts):
            le = "+Inf" if bucket == float("inf") else str(bucket)
            prefix = f"{labels}," if labels else ""
            lines.append(f'{name}_bucket{{{prefix}le="{le}"}} {count}')
        lines.append(f"{name}_sum{label_str} {self._sum:.6f}")
        lines.append(f"{name}_count{label_str} {self._total}")
        return lines


class MetricsCollector:
    """Потокобезопасный сборщик метрик с экспортом в формат Prometheus.

    Отслеживает:
        recommend_requests_total   — счётчик запросов
        recommend_errors_total     — счётчик ошибок
        recommend_latency_seconds  — гистограмма задержки
        cold_start_total           — счётчик cold-start запросов
        service_uptime_seconds     — время работы сервиса
    """

    def __init__(self, model_version: str = "unknown") -> None:
        self._lock = threading.Lock()
        self._model_version = model_version
        self._start_time = time.time()
        self._requests_total: int = 0
        self._errors_total: int = 0
        self._cold_start_total: int = 0
        self._latency = _Histogram()

    def record_request(
        self,
        latency_seconds: float,
        error: bool = False,
        is_cold_start: bool = False,
        n_recommendations: int = 0,
    ) -> None:
        with self._lock:
            self._requests_total += 1
            if error:
                self._errors_total += 1
            else:
                self._latency.observe(latency_seconds)
                if is_cold_start:
                    self._cold_start_total += 1

    def prometheus_text(self) -> str:
        with self._lock:
            uptime = time.time() - self._start_time
            mv = self._model_version
            lines = [
                "# HELP recommend_requests_total Total recommendation requests",
                "# TYPE recommend_requests_total counter",
                f'recommend_requests_total{{model="{mv}"}} {self._requests_total}',
                "",
                "# HELP recommend_errors_total Total recommendation errors",
                "# TYPE recommend_errors_total counter",
                f'recommend_errors_total{{model="{mv}"}} {self._errors_total}',
                "",
                "# HELP recommend_cold_start_total Requests served with popularity fallback",
                "# TYPE recommend_cold_start_total counter",
                f'recommend_cold_start_total{{model="{mv}"}} {self._cold_start_total}',
                "",
                "# HELP recommend_latency_seconds Recommendation latency in seconds",
                "# TYPE recommend_latency_seconds histogram",
                *self._latency.prometheus_lines("recommend_latency_seconds", f'model="{mv}"'),
                "",
                "# HELP service_uptime_seconds Seconds since process start",
                "# TYPE service_uptime_seconds gauge",
                f'service_uptime_seconds{{model="{mv}"}} {uptime:.1f}',
                "",
            ]
            return "\n".join(lines)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "model_version": self._model_version,
                "requests_total": self._requests_total,
                "errors_total": self._errors_total,
                "cold_start_total": self._cold_start_total,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "latency_avg_ms": round(
                    self._latency._sum / max(self._latency._total, 1) * 1000, 2
                ),
            }
