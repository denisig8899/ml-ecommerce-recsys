"""Тесты сохранения и загрузки ModelArtifact."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import ModelArtifact


def make_artifact() -> ModelArtifact:
    return ModelArtifact(
        model_type="popularity",
        top_k=10,
        user_map={1: 0, 2: 1},
        item_map={10: 0, 20: 1, 30: 2},
        item_ids=np.array([10, 20, 30]),
        popular_items=[10, 20],
        n_users=2,
        n_items=3,
        metrics={"recall_at_10": 0.015},
    )


def test_save_load(tmp_path):
    art = make_artifact()
    path = tmp_path / "artifact.pkl"
    art.save(path)
    art2 = ModelArtifact.load(path)
    assert art2.model_type == "popularity"
    assert art2.n_users == 2
    assert art2.n_items == 3
    assert art2.metrics["recall_at_10"] == pytest.approx(0.015)


def test_user_map_preserved(tmp_path):
    art = make_artifact()
    path = tmp_path / "artifact.pkl"
    art.save(path)
    art2 = ModelArtifact.load(path)
    assert art2.user_map[1] == 0
    assert art2.user_map[2] == 1


def test_popular_items(tmp_path):
    art = make_artifact()
    path = tmp_path / "artifact.pkl"
    art.save(path)
    art2 = ModelArtifact.load(path)
    assert art2.popular_items == [10, 20]
