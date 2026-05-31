"""Загрузка модели и инференс рекомендаций."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .artifact import ModelArtifact


class Recommender:
    """Обёртка над ALS / popularity-моделью для инференса.

    Поддерживает cold-start: если пользователь не встречался при обучении,
    возвращает топ-K популярных товаров.
    """

    def __init__(self, model_path: str | Path, artifact_path: str | Path) -> None:
        import joblib
        self.artifact: ModelArtifact = ModelArtifact.load(artifact_path)
        self._model_path = Path(model_path)
        self._model = None
        if self.artifact.model_type == "als":
            self._model = joblib.load(model_path)

    def recommend(
        self,
        visitor_id: int,
        n: int | None = None,
        user_items: sp.csr_matrix | None = None,
    ) -> list[int]:
        """Вернуть топ-n itemid для пользователя.

        Аргументы:
            visitor_id: идентификатор пользователя
            n: количество рекомендаций (по умолчанию artifact.top_k)
            user_items: строка user-item матрицы для фильтрации уже просмотренных
        """
        top_k = n or self.artifact.top_k
        art = self.artifact

        if art.model_type == "popularity" or visitor_id not in art.user_map:
            return self._cold_start(top_k, visitor_id)

        user_idx = art.user_map[visitor_id]

        if user_items is None:
            user_items = sp.csr_matrix((1, art.n_items), dtype=np.float32)

        ids, _ = self._model.recommend(
            user_idx,
            user_items,
            N=top_k,
            filter_already_liked_items=True,
        )
        return [int(art.item_ids[i]) for i in ids]

    def _cold_start(self, top_k: int, visitor_id: int) -> list[int]:
        """Популярные товары, которые пользователь ещё не взаимодействовал."""
        art = self.artifact
        if visitor_id in art.user_map:
            user_idx = art.user_map[visitor_id]
        else:
            return art.popular_items[:top_k]

        # Исключить товары, с которыми уже было взаимодействие
        seen_set: set[int] = set()
        if hasattr(self, "_matrix") and self._matrix is not None:
            row = self._matrix.user_item[user_idx]
            seen_set = set(art.item_ids[row.indices].tolist())

        result = [iid for iid in art.popular_items if iid not in seen_set]
        return result[:top_k]
