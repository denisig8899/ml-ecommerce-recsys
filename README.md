# ML E-commerce Recommender

Рекомендательная система для маркетплейса на основе событийных логов.
Предсказывает товары, которые пользователь с высокой вероятностью добавит в корзину.

**Данные**: Retailrocket — 2.76M событий, 1.4M пользователей, 235K товаров.
**Стратегия оптимизации**: максимизация `addtocart`-конверсии.

---

## Структура репозитория

```
notebooks/          — Jupyter-ноутбуки по этапам работы
scripts/            — вспомогательные скрипты
src/
  api/              — FastAPI-сервис рекомендаций
  features/         — построение матрицы взаимодействий
  models/           — загрузка артефакта и инференс
  monitoring/       — сбор и экспорт метрик
airflow/
  dags/             — DAG дообучения (еженедельно)
docker/             — Dockerfile и docker-compose
docs/               — документация по мониторингу
data/               — данные (не в git)
models/             — артефакты моделей (не в git)
mlflow/             — метаданные MLflow
```

---

## Данные

Файлы датасета Retailrocket кладутся в `data/raw/`:

```
data/raw/
  events.csv                  — лог событий (view / addtocart / transaction)
  item_properties_part1.csv   — свойства товаров
  item_properties_part2.csv
  category_tree.csv           — иерархия категорий
```

| Статистика | Значение |
|---|---|
| Всего событий | 2 756 101 |
| view | 2 664 312 (96.7%) |
| addtocart | 69 332 (2.5%) |
| transaction | 22 457 (0.8%) |
| Уникальных пользователей | 1 407 580 |
| Уникальных товаров | 235 061 |

---

## Этапы работы

### 1. EDA событийных данных (`01_eda.ipynb`)

- Конверсионная воронка: view → addtocart → transaction
- Поведение пользователей: распределение числа событий на пользователя (тяжёлый хвост)
- Популярность товаров: топ-100 товаров покрывают ~30% всех addtocart
- Временная динамика: события за период с мая по сентябрь 2015
- Свойства товаров: 417K записей, ключевое свойство — `categoryid`

### 2. Выбор стратегии и метрик

**Цель**: максимизация добавлений в корзину (`addtocart`).

Обоснование:
- `transaction` редки (0.8%) и имеют задержку относительно intent
- `view` — слабый сигнал, не отражает явного интереса к покупке
- `addtocart` — явный сигнал коммерческого интереса с достаточным объёмом (+3× больше транзакций)

**Метрика оценки**: recall@10, NDCG@10.

- **recall@10**: доля реально добавленных в корзину товаров, которые попали в топ-10 рекомендаций.
  В e-commerce пользователь видит limited widget; попасть в эти 10 — уже победа.
- **NDCG@10**: учитывает позицию рекомендации; более высокая позиция = выше вероятность клика.

### 3. Моделирование (`02_models.ipynb`)

#### Постановка задачи

Implicit feedback: нет явных оценок (ratings), только факты взаимодействия.
Задача — ранжирование: для каждого пользователя вернуть топ-K товаров.

**Веса событий**: view=1, addtocart=5, transaction=10.

#### Матрица взаимодействий

Sparse CSR-матрица (1 407 580 × 235 061), density ≈ 0.0001%.
После фильтрации «активных» пользователей (≥ 2 события) — ~210K пользователей.

#### Базовая модель — Popularity Baseline

Рекомендовать топ-K самых популярных по `addtocart` товаров, исключая уже просмотренные.

| Метрика | Значение |
|---|---|
| recall@10 | 0.034 |
| NDCG@10 | 0.021 |
| precision@10 | 0.009 |

#### ALS — Alternating Least Squares (`implicit`)

Матричная факторизация (implicit feedback). Пространство факторов: 64.

```
AlternatingLeastSquares(factors=64, iterations=20, regularization=0.05)
```

| Метрика | Значение | vs Baseline |
|---|---|---|
| recall@10 | **0.087** | +156% |
| NDCG@10 | **0.056** | +167% |
| precision@10 | **0.024** | +167% |

**Cold-start**: пользователи без истории получают popularity-рекомендации.

### 4. MLflow (`03_mlflow_tracking.ipynb`, `scripts/log_mlflow_runs.py`)

Оба эксперимента залогированы в `mlruns/`.
Для повторного логирования:

```bash
python scripts/log_mlflow_runs.py
```

Запуск UI:

```bash
bash scripts/setup_mlflow.sh
# или
python -m mlflow ui --backend-store-uri ./mlruns
```

---

## Установка и запуск

### Локально

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### API-сервис

```bash
export MODEL_PATH=models/als_model.pkl
export ARTIFACT_PATH=models/artifact.pkl
export PYTHONPATH=src
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Эндпоинты:

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | Статус сервиса, тип модели, размер пространства |
| POST | `/recommend` | Топ-K рекомендаций для пользователя |
| GET | `/metrics` | Метрики в формате Prometheus |
| GET | `/metrics/snapshot` | Метрики в JSON |

Пример запроса:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"visitor_id": 257597, "n": 10}'
```

### Docker

```bash
# API + MLflow + Airflow
docker compose -f docker/docker-compose.yml up -d

# Только API
docker build -f docker/Dockerfile -t ml-ecommerce-recsys .
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro ml-ecommerce-recsys
```

Airflow UI доступен на `http://localhost:8080` (admin / admin).

---

## Airflow DAG — дообучение

DAG `ecommerce_recsys_retrain` запускается каждый понедельник в 00:00 UTC.

```
load_events → build_matrix → train_model → evaluate → log_mlflow → update_model
```

| Задача | Описание |
|---|---|
| load_events | Загрузка событий за последние 90 дней |
| build_matrix | Построение разреженной матрицы взаимодействий |
| train_model | Обучение ALS (factors=64, iter=20) |
| evaluate | recall@10, NDCG@10 на holdout (последние 14 дней) |
| log_mlflow | Логирование параметров, метрик и артефактов |
| update_model | Замена production-артефактов (при recall@10 ≥ 0.010) |

---

## Воспроизведение экспериментов

Ноутбуки запускаются последовательно:

```
01_eda.ipynb
02_models.ipynb
03_mlflow_tracking.ipynb
```

Для запуска без переобучения скопировать артефакты:

```
models/als_model.pkl
models/artifact.pkl
```

---

## Мониторинг

Описание метрик, порогов и сигналов дрейфа — в [`docs/monitoring.md`](docs/monitoring.md).

---

## Зависимости

Python 3.12. Основные библиотеки: pandas 2.2.3, implicit 0.7.2, fastapi 0.115.5,
mlflow 2.18.0, apache-airflow 2.10.3.
