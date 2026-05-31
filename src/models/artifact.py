"""Артефакт обученной рекомендательной модели."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ModelArtifact:
    """Все данные, необходимые для инференса без переобучения.

    Атрибуты:
        model_type: тип модели ('als' | 'popularity')
        top_k: количество рекомендаций по умолчанию
        user_map: visitorid → индекс в матрице
        item_map: itemid → индекс в матрице
        item_ids: массив itemid (индекс → itemid)
        popular_items: топ-N товаров по количеству addtocart (для cold-start)
        n_users: размер пространства пользователей
        n_items: размер пространства товаров
        metrics: метрики качества на валидации
    """

    model_type: str
    top_k: int
    user_map: dict[int, int]
    item_map: dict[int, int]
    item_ids: np.ndarray
    popular_items: list[int]
    n_users: int
    n_items: int
    metrics: dict[str, float] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ModelArtifact":
        import joblib
        return joblib.load(path)
