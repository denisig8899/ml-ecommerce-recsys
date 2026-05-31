"""Залогировать оба эксперимента в MLflow из готовых артефактов.

Запуск:
    python scripts/log_mlflow_runs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mlflow

MLFLOW_URI = ROOT / "mlruns"
MODELS_DIR = ROOT / "models"


def log_popularity_baseline() -> None:
    mlflow.set_tracking_uri(str(MLFLOW_URI))
    mlflow.set_experiment("ecommerce-recsys")

    with mlflow.start_run(run_name="popularity_baseline"):
        mlflow.log_params({
            "model_type": "popularity",
            "top_k": 10,
            "target": "addtocart",
        })
        mlflow.log_metrics({
            "recall@10": 0.0150,
            "ndcg@10": 0.0098,
            "precision@10": 0.0024,
        })
        artifact = MODELS_DIR / "artifact_popularity.pkl"
        if artifact.exists():
            mlflow.log_artifact(str(artifact))


def log_als_model() -> None:
    mlflow.set_tracking_uri(str(MLFLOW_URI))
    mlflow.set_experiment("ecommerce-recsys")

    with mlflow.start_run(run_name="als_v1"):
        mlflow.log_params({
            "model_type": "als",
            "factors": 64,
            "iterations": 20,
            "regularization": 0.05,
            "random_state": 42,
            "top_k": 10,
            "target": "addtocart",
            "weights": "view=1, addtocart=5, transaction=10",
        })
        mlflow.log_metrics({
            "recall@10": 0.0167,
            "ndcg@10": 0.0104,
            "precision@10": 0.0028,
        })
        for name in ("als_model.pkl", "artifact.pkl"):
            path = MODELS_DIR / name
            if path.exists():
                mlflow.log_artifact(str(path))


if __name__ == "__main__":
    print("Логирование popularity baseline...")
    log_popularity_baseline()
    print("Логирование ALS v1...")
    log_als_model()
    print("Готово. Запустите MLflow UI: bash scripts/setup_mlflow.sh")
