"""Загрузка артефакта и инференс рекомендательной модели."""

from .artifact import ModelArtifact
from .predictor import Recommender

__all__ = ["ModelArtifact", "Recommender"]
