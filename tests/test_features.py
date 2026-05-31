"""Тесты построения матрицы взаимодействий."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import InteractionMatrix


@pytest.fixture()
def sample_events() -> pd.DataFrame:
    return pd.DataFrame({
        "visitorid": [1, 1, 2, 2, 3],
        "itemid":    [10, 20, 10, 30, 20],
        "event":     ["view", "addtocart", "addtocart", "view", "transaction"],
    })


def test_fit_shape(sample_events):
    m = InteractionMatrix()
    m.fit(sample_events)
    assert m.n_users == 3
    assert m.n_items == 3


def test_fit_weights(sample_events):
    m = InteractionMatrix()
    m.fit(sample_events)
    # user 1, item 10: view=1.0
    u_idx = m.user_map[1]
    i_idx = m.item_map[10]
    assert m.user_item[u_idx, i_idx] == pytest.approx(1.0)
    # user 1, item 20: addtocart=5.0
    i_idx2 = m.item_map[20]
    assert m.user_item[u_idx, i_idx2] == pytest.approx(5.0)


def test_user_map_index_bounds(sample_events):
    m = InteractionMatrix()
    m.fit(sample_events)
    for uid, idx in m.user_map.items():
        assert 0 <= idx < m.n_users, f"user_map[{uid}]={idx} вне диапазона"


def test_item_map_index_bounds(sample_events):
    m = InteractionMatrix()
    m.fit(sample_events)
    for iid, idx in m.item_map.items():
        assert 0 <= idx < m.n_items, f"item_map[{iid}]={idx} вне диапазона"


def test_item_user_transpose(sample_events):
    m = InteractionMatrix()
    m.fit(sample_events)
    assert m.item_user.shape == (m.n_items, m.n_users)


def test_density_positive(sample_events):
    m = InteractionMatrix()
    m.fit(sample_events)
    assert m.density > 0


def test_save_load(sample_events, tmp_path):
    m = InteractionMatrix()
    m.fit(sample_events)
    path = tmp_path / "matrix.pkl"
    m.save(path)
    m2 = InteractionMatrix.load(path)
    assert m2.n_users == m.n_users
    assert m2.n_items == m.n_items
