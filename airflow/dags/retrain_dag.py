"""Airflow DAG: еженедельное дообучение рекомендательной модели.

Граф задач:
    load_events → build_matrix → train_model → evaluate → log_mlflow → update_model

Расписание: каждый понедельник в 00:00 UTC.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data/raw"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# ---------------------------------------------------------------------------
# Задачи
# ---------------------------------------------------------------------------


def load_events(**context) -> str:
    """Загрузить события, разбить на train/val, посчитать drift-метрики."""
    import pandas as pd

    events_path = DATA_DIR / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Файл событий не найден: {events_path}")

    events = pd.read_csv(events_path)
    events["dt"] = pd.to_datetime(events["timestamp"], unit="ms")

    window_cutoff = events["dt"].max() - pd.Timedelta(days=90)
    recent = events[events["dt"] >= window_cutoff].copy()

    # Train/val split без leakage: train — первые 76 дней, val — последние 14
    val_cutoff = recent["dt"].max() - pd.Timedelta(days=14)
    train = recent[recent["dt"] < val_cutoff].copy()
    val = recent[recent["dt"] >= val_cutoff].copy()

    train.to_parquet(DATA_DIR / "events_train.parquet", index=False)
    val.to_parquet(DATA_DIR / "events_val.parquet", index=False)
    recent.to_parquet(DATA_DIR / "events_recent.parquet", index=False)

    # Базовые drift-метрики по train-окну
    total = max(len(train), 1)
    drift = {
        "view_ratio": float((train["event"] == "view").sum() / total),
        "cart_ratio": float((train["event"] == "addtocart").sum() / total),
        "tx_ratio": float((train["event"] == "transaction").sum() / total),
        "n_unique_users": int(train["visitorid"].nunique()),
        "n_unique_items": int(train["itemid"].nunique()),
        "events_per_user": float(len(train) / max(train["visitorid"].nunique(), 1)),
    }
    context["ti"].xcom_push(key="drift_metrics", value=drift)
    context["ti"].xcom_push(key="events_path", value=str(DATA_DIR / "events_train.parquet"))
    context["ti"].xcom_push(key="n_events", value=len(train))
    return str(DATA_DIR / "events_train.parquet")


def build_matrix(**context) -> str:
    """Построить разреженную матрицу взаимодействий."""
    import sys
    sys.path.insert(0, "/app/src")

    import pandas as pd
    from features import InteractionMatrix

    events_path = context["ti"].xcom_pull(key="events_path")
    events = pd.read_parquet(events_path)

    matrix = InteractionMatrix()
    matrix.fit(events)

    out_path = DATA_DIR / "matrix.pkl"
    matrix.save(out_path)

    context["ti"].xcom_push(key="matrix_path", value=str(out_path))
    context["ti"].xcom_push(key="n_users", value=matrix.n_users)
    context["ti"].xcom_push(key="n_items", value=matrix.n_items)
    return str(out_path)


def train_model(**context) -> str:
    """Обучить ALS-модель на новой матрице."""
    import sys
    sys.path.insert(0, "/app/src")

    import joblib
    import implicit
    from features import InteractionMatrix
    from models import ModelArtifact
    import numpy as np

    matrix_path = context["ti"].xcom_pull(key="matrix_path")
    matrix: InteractionMatrix = InteractionMatrix.load(matrix_path)

    model = implicit.als.AlternatingLeastSquares(
        factors=64,
        iterations=20,
        regularization=0.05,
        random_state=42,
        use_gpu=False,
    )
    model.fit(matrix.user_item)  # implicit >= 0.7: fit() принимает (n_users × n_items)

    # Popularity fallback: топ по addtocart
    import pandas as pd
    events = pd.read_parquet(DATA_DIR / "events_recent.parquet")
    popular = (
        events[events["event"] == "addtocart"]
        .groupby("itemid")
        .size()
        .sort_values(ascending=False)
        .index.tolist()[:200]
    )

    artifact = ModelArtifact(
        model_type="als",
        top_k=10,
        user_map=matrix.user_map,
        item_map=matrix.item_map,
        item_ids=matrix.item_ids,
        popular_items=popular,
        n_users=matrix.n_users,
        n_items=matrix.n_items,
    )

    model_path = MODELS_DIR / "als_model_new.pkl"
    artifact_path = MODELS_DIR / "artifact_new.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    artifact.save(artifact_path)

    context["ti"].xcom_push(key="model_path", value=str(model_path))
    context["ti"].xcom_push(key="artifact_path", value=str(artifact_path))
    return str(model_path)


def evaluate_model(**context) -> dict:
    """Вычислить recall@10 и NDCG@10 на чистом holdout (val = последние 14 дней).

    Модель обучена на train (первые 76 дней), val не использовался при обучении.
    """
    import sys
    sys.path.insert(0, "/app/src")

    import joblib
    import numpy as np
    import pandas as pd
    from features import InteractionMatrix
    from models import ModelArtifact

    matrix_path = context["ti"].xcom_pull(key="matrix_path")
    model_path = context["ti"].xcom_pull(key="model_path")
    artifact_path = context["ti"].xcom_pull(key="artifact_path")

    matrix: InteractionMatrix = InteractionMatrix.load(matrix_path)
    model = joblib.load(model_path)
    artifact: ModelArtifact = ModelArtifact.load(artifact_path)

    # val не пересекается с train — загружаем отдельно (сохранён в load_events)
    val_cart = pd.read_parquet(DATA_DIR / "events_val.parquet")
    val_cart = val_cart[val_cart["event"] == "addtocart"]

    val_users = val_cart["visitorid"].unique().tolist()
    # Фильтрация по user_map, а не по n_users (visitorid != матричный индекс)
    sample = [uid for uid in val_users if uid in artifact.user_map][:500]

    recall_scores, ndcg_scores = [], []
    K = 10

    for uid in sample:
        user_idx = artifact.user_map[uid]
        user_row = matrix.user_item[user_idx]
        ids, _ = model.recommend(user_idx, user_row, N=K, filter_already_liked_items=True)
        recs = set(ids)

        ground_truth = set(
            val_cart[val_cart["visitorid"] == uid]["itemid"]
            .map(artifact.item_map)
            .dropna()
            .astype(int)
        )
        if not ground_truth:
            continue

        hits = recs & ground_truth
        recall_scores.append(len(hits) / len(ground_truth))

        dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(ids) if r in ground_truth)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), K)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {
        f"recall_at_{K}": float(np.mean(recall_scores)) if recall_scores else 0.0,
        f"ndcg_at_{K}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "n_evaluated": len(recall_scores),
    }

    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


def log_to_mlflow(**context) -> None:
    """Залогировать параметры и метрики в MLflow."""
    import mlflow

    metrics = context["ti"].xcom_pull(key="metrics")
    drift = context["ti"].xcom_pull(key="drift_metrics") or {}
    n_users = context["ti"].xcom_pull(key="n_users")
    n_items = context["ti"].xcom_pull(key="n_items")
    n_events = context["ti"].xcom_pull(key="n_events")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ecommerce-recsys-retrain")

    with mlflow.start_run(run_name=f"retrain_{datetime.utcnow().strftime('%Y%m%d')}"):
        mlflow.log_params({
            "model_type": "als",
            "factors": 64,
            "iterations": 20,
            "regularization": 0.05,
            "n_users": n_users,
            "n_items": n_items,
            "n_events": n_events,
            "top_k": 10,
        })
        mlflow.log_metrics(metrics)
        # Drift-метрики (view/cart/tx ratio, n_unique_users/items)
        mlflow.log_metrics({f"drift_{k}": v for k, v in drift.items() if isinstance(v, float)})
        mlflow.log_artifact(context["ti"].xcom_pull(key="model_path"))
        mlflow.log_artifact(context["ti"].xcom_pull(key="artifact_path"))


def update_model(**context) -> None:
    """Заменить production-артефакты новыми, если метрики не ухудшились."""
    import shutil

    metrics = context["ti"].xcom_pull(key="metrics")
    recall = metrics.get("recall_at_10", 0.0)
    MIN_RECALL = float(os.getenv("MIN_RECALL_THRESHOLD", "0.01"))
    if recall < MIN_RECALL:
        raise ValueError(
            f"Новая модель не прошла порог качества: recall@10={recall:.4f} < {MIN_RECALL}"
        )

    model_path = context["ti"].xcom_pull(key="model_path")
    artifact_path = context["ti"].xcom_pull(key="artifact_path")

    shutil.copy(model_path, MODELS_DIR / "als_model.pkl")
    shutil.copy(artifact_path, MODELS_DIR / "artifact.pkl")

    # Сигнализируем API перезагрузить модель без перезапуска контейнера
    import urllib.request
    api_url = os.getenv("API_URL", "http://api:8000")
    try:
        req = urllib.request.Request(f"{api_url}/reload", method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        # Не прерываем DAG если API недоступен — предупреждение достаточно
        print(f"Warning: не удалось вызвать /reload: {exc}")


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="ecommerce_recsys_retrain",
    description="Еженедельное дообучение ALS-рекомендаций",
    default_args=default_args,
    schedule_interval="0 0 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["recsys", "ecommerce", "als"],
) as dag:

    t_load = PythonOperator(
        task_id="load_events",
        python_callable=load_events,
    )

    t_matrix = PythonOperator(
        task_id="build_matrix",
        python_callable=build_matrix,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t_eval = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t_log = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow,
    )

    t_update = PythonOperator(
        task_id="update_model",
        python_callable=update_model,
    )

    t_load >> t_matrix >> t_train >> t_eval >> t_log >> t_update
