"""Построение матрицы взаимодействий пользователь-товар из событийных логов."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 5.0,
    "transaction": 10.0,
}


class InteractionMatrix:
    """Разреженная матрица взаимодействий пользователь × товар.

    Строится из лога событий с весами по типу события:
    view=1, addtocart=5, transaction=10.

    Атрибуты:
        user_item: CSR-матрица (n_users × n_items)
        item_user: CSR-матрица (n_items × n_users) для implicit
        user_map: visitorid → строка матрицы
        item_map: itemid → столбец матрицы
        user_ids: массив id пользователей (индекс → id)
        item_ids: массив id товаров (индекс → id)
    """

    def __init__(self) -> None:
        self.user_item: sp.csr_matrix | None = None
        self.item_user: sp.csr_matrix | None = None
        self.user_map: dict[int, int] = {}
        self.item_map: dict[int, int] = {}
        self.user_ids: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None

    def fit(self, events: pd.DataFrame) -> "InteractionMatrix":
        """Построить матрицы из DataFrame событий.

        Аргументы:
            events: DataFrame с колонками [visitorid, itemid, event].
                    Веса суммируются по парам (user, item).
        """
        events = events.copy()
        events["weight"] = events["event"].map(EVENT_WEIGHTS).fillna(0.0)
        events = events[events["weight"] > 0]

        agg = (
            events.groupby(["visitorid", "itemid"])["weight"]
            .sum()
            .reset_index()
        )

        unique_users = np.sort(agg["visitorid"].unique())
        unique_items = np.sort(agg["itemid"].unique())

        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_map = {iid: i for i, iid in enumerate(unique_items)}
        self.user_ids = unique_users
        self.item_ids = unique_items

        rows = agg["visitorid"].map(self.user_map).values
        cols = agg["itemid"].map(self.item_map).values
        data = agg["weight"].values.astype(np.float32)

        n_users = len(unique_users)
        n_items = len(unique_items)

        self.user_item = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
        )
        self.item_user = self.user_item.T.tocsr()
        return self

    @property
    def n_users(self) -> int:
        return self.user_item.shape[0]

    @property
    def n_items(self) -> int:
        return self.user_item.shape[1]

    @property
    def density(self) -> float:
        return self.user_item.nnz / (self.n_users * self.n_items)

    def get_user_items(self, visitor_id: int) -> sp.csr_matrix:
        """Вернуть строку матрицы для пользователя (для implicit)."""
        idx = self.user_map.get(visitor_id)
        if idx is None:
            return sp.csr_matrix((1, self.n_items), dtype=np.float32)
        return self.user_item[idx]

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "InteractionMatrix":
        import joblib
        return joblib.load(path)
