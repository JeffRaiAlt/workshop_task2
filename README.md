# Cистема матчинга и дедупликации профилей клиентов

Проект для entity resolution / identity resolution: объединяет профили одного пользователя на основе графовых признаков, эмбеддингов GraphSAGE и pairwise-классификатора CatBoost, а также предоставляет простой Streamlit-интерфейс для пакетного скоринга входных данных.

## Что делает проект

Проект строит пайплайн сопоставления профилей, в котором данные сначала нормализуются и извлекаются из вложенных полей, затем агрегируются на уровне `profile_id`, после чего формируются кандидаты на совпадение и рассчитывается score вероятности того, что два профиля принадлежат одному человеку.

**Важно!** В проекте реализованы два pipeline:
 - Модель `GraphSAGE + CatBoost`
 - [Модель Graph_Table: витрина + blocking + XGBoost + граф](#graph-table).

## Модель GraphSAGE + CatBoost

В обучающем ноутбуке используется гетерогенный граф по связям профиля с атрибутами: `domain`, `phone`, `device`, `city` и `region`. Затем обучается GraphSAGE, создаются эмбеддинги профилей, генерируются кандидатные пары и обучается CatBoost на pairwise-признаках.

В production-коде инференс работает через набор артефактов в директории `artifacts/`: 
- `catboost_pair_model.cbm`
- `feature_cols.json`
- `graph_schema.pkl`
- `graphsage_state_dict.pt`
- `summary.json`

## Состав репозитория

| Файл | Назначение |
|------|------------|
| `app.py` | Streamlit-приложение для загрузки данных, запуска препроцессинга, скоринга и просмотра результатов |
| `pipeline.py` | Основная inference-логика: агрегация профилей, генерация кандидатных пар, расчет признаков, CatBoost scoring и fallback-режим |
| `preprocess.py` | Подготовка входного датафрейма: парсинг `non_processing_features`, `realtime_features`, `fs_features`, выделение служебных колонок и split |
| `services.py` | Вспомогательные сервисные функции для загрузки/сохранения данных и оркестрации приложения |
| `report.py` | Генерация краткого отчета по результатам скоринга |
| `identity_graph_pipeline.ipynb` | Полный исследовательский и обучающий пайплайн с GraphSAGE и CatBoost |
| `EDA.ipynb` | Разведочный анализ данных |
| `model.ipynb` | Отдельные эксперименты с моделированием |
| `Dockerfile` | Docker-образ для запуска Streamlit-сервиса на Python 3.11 |
| `docker-compose.yml` | Docker Compose-конфигурация для локального запуска сервиса |
| `requirements.txt` | Python-зависимости проекта |

## Архитектура

### 1. Подготовка данных

Скрипт препроцессинга извлекает признаки из полей `non_processing_features`, `realtime_features` и `fs_features`, а также добавляет такие колонки, как `device`, `osfamily`, `browser`, `region_code`, `city_name`, `local_hour`, `visit_count`, `city_population`, `email_domain` и `split`.

Если колонка `split` отсутствует, она создается автоматически через разбиение по `entity_id`, чтобы не смешивать сущности между train и test.

### 2. Построение профилей

Во время инференса записи агрегируются до уровня `profile_id`: для контактных и категориальных атрибутов собираются множества значений, а для числовых признаков считаются агрегаты по среднему.

На этом этапе формируются базовые признаки: `n_email`, `n_phone`, `n_domain`, `n_device`, `n_city`, `n_region`, `n_browser`, `n_os`, а также агрегаты поведенческих колонок.

### 3. Генерация кандидатных пар

Кандидаты генерируются двумя способами: 
- через общие атрибуты (`domain`, `phone`, `device`, `city`, `region`)
- через nearest neighbors по числовому пространству признаков / эмбеддингов

Такой подход уменьшает количество пар для сравнения и позволяет не считать скор для полного декартова произведения профилей.

### 4. Pairwise scoring

Для каждой пары считаются признаки пересечения и похожести: общие телефоны, email, домены, устройства, города, регионы, а также Jaccard/overlap и числовые разницы по агрегированным характеристикам профилей.

В production-пайплайне основной скорер — CatBoost; если модель или `feature_cols.json` недоступны, код переключается на fallback-логику.

## Метрики из обучающего пайплайна

В ноутбуке `identity_graph_pipeline.ipynb` для полного пайплайна показаны высокие offline-метрики:
- Train ROC-AUC: 0.9958, PR-AUC: 0.9862
- Validation ROC-AUC: 0.9954, PR-AUC: 0.9405
- Test ROC-AUC: 0.9934, PR-AUC: 0.9147

Эти значения относятся к конкретной конфигурации с GraphSAGE + candidate generation + CatBoost и должны восприниматься как ориентир для приложенного набора данных и текущих гиперпараметров.

## Установка и запуск

### Локально

1. Создайте виртуальное окружение
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Убедитесь, что рядом с кодом есть директория `artifacts/` с обученными файлами модели
4. Запустите приложение:
```bash
streamlit run app.py
```

### Запуск в Docker

Dockerfile собирается на базе `python:3.11-slim`, устанавливает системные пакеты (`gcc`, `g++`, `libgomp1`, `wget`), копирует проект и запускает Streamlit на порту 8501.

Пример запуска через Docker Compose:
```bash
docker-compose up --build
```

После старта приложение будет доступно на порту `8501`, который пробрасывается в compose-конфигурации.

## Формат входных данных

Для корректной работы нужны как минимум идентификаторы профиля, а для качественного матчинга желательно передавать атрибуты: 
- `email`, `phone`, `device`, `city_name`, `region_code`, `browser`, `osfamily`, `sex`, `created_at`
- а также вложенные словари/JSON в `non_processing_features`, `realtime_features` и `fs_features`

Если часть колонок отсутствует, пайплайн старается деградировать мягко: использует доступные признаки и подставляет пустые множества или нули там, где это предусмотрено в коде.

## Артефакты модели

Проект ожидает следующие файлы в `artifacts/`:
- `catboost_pair_model.cbm` — обученный pairwise CatBoost-классификатор
- `feature_cols.json` — список признаков в точном порядке для инференса
- `graph_schema.pkl` — схема графа и статистики признаков из train
- `graphsage_state_dict.pt` — веса GraphSAGE-энкодера
- `summary.json` — сводка по запуску обучения и конфигурации

## Пример сценария использования

1. Подготовить CSV или parquet с сырыми профилями
2. Загрузить файл в Streamlit-приложение
3. Настроить параметры анализа (режим обработки, порог score, количество строк)
4. Запустить обработку
5. Получить таблицу пар `profile_id_1`, `profile_id_2`, `score` и сопутствующий Markdown-отчет

<a id="graph-table"></a>
## Модель Graph_Table: витрина, blocking, XGBoost и граф

### Файлы

| Путь | Назначение |
|---|---|
| `notebooks/03_build_er_profile_mart_multivalue.ipynb` | Витрина и blocking-правила |
| `src/build_graph_table_artifacts.py` | Обучение модели и сборка артефактов |
| `src/run_graph_table_inference.py` | Инференс и HTML-отчёт |
| `src/run_graph_table_threshold_sweep.py` | Проверка порогов |
| `src/build_graph_table_shap_report.py` | SHAP-отчёт |
| `src/streamlit_graph_table_inference.py` | Интерфейс инференса |

### Архитектура

1. Ноутбук `03_build_er_profile_mart_multivalue.ipynb` собирает значения
   признаков по `profile_id` и создаёт `blocking_index`.
2. `recommended_blocking_rules.csv` задаёт правила, по которым профили
   попадают в пары-кандидаты.
3. `build_graph_table_artifacts.py` формирует обучающие пары: положительные
   пары имеют один `entity_id`, отрицательные пары имеют разные `entity_id`.
4. Для пары рассчитываются совпадения значений и признаки сработавших
   blocking-правил.
5. XGBoost обучается оценивать вероятность совпадения двух профилей.
6. При инференсе входные профили сопоставляются с историческим
   `blocking_index`, и модель оценивает только найденные пары-кандидаты.
7. Пары выше порога проходят правило `graph_top_k` и становятся рёбрами
   графа; связанные профили образуют группы для объединения.

(Все указанные артефакты снабжены подробными комментариями)

### Запуск

Исходные данные:

```text
data/split_label_train_V3.snappy.parquet
```

Собрать витрину:

```powershell
cd notebooks
python -m jupyter nbconvert --to notebook --execute --inplace 03_build_er_profile_mart_multivalue.ipynb
cd ..
```

Обучить модель и собрать артефакты:

```powershell
python src/build_graph_table_artifacts.py --max-negative-pairs 1000000
```

Создать тестовый пакет:

```powershell
python src/make_graph_table_test_packet.py --n-profiles 1000 --entity-split test --existing-share 0.25
```

Запустить инференс:

```powershell
python src/run_graph_table_inference.py --input src/graph_table_test_data/raw_packet_test_1000_profiles.parquet
```

HTML-отчёт сохраняется в:

```text
reports/graph_table_inference/
```

Запустить интерфейс:

```powershell
python -m streamlit run src/streamlit_graph_table_inference.py
```

### Артефакты

После выполнения ноутбука:

| Файл | Назначение |
|---|---|
| `data/processed/er_profile_mart_multivalue/profile_core.parquet` | Профили и разметка |
| `data/processed/er_profile_mart_multivalue/profile_value_summary_long.parquet` | Значения признаков |
| `data/processed/er_profile_mart_multivalue/blocking_index.parquet` | Blocking index |
| `data/processed/er_profile_mart_multivalue/recommended_blocking_rules.csv` | Используемые правила |

После обучения:

| Файл | Назначение |
|---|---|
| `src/graph_table_artifacts/graph_edge_model.joblib` | Модель XGBoost |
| `src/graph_table_artifacts/feature_cols.json` | Признаки модели |
| `src/graph_table_artifacts/policy_config.json` | Runtime-настройки |
| `src/graph_table_artifacts/recommended_rule_names.json` | Blocking-правила |
| `src/graph_table_artifacts/artifact_manifest.json` | Параметры модели |
| `src/graph_table_artifacts/historical_profile_core.parquet` | Исторические профили |
| `src/graph_table_artifacts/historical_profile_values.parquet` | Значения исторических профилей |
| `src/graph_table_artifacts/historical_blocking_index.parquet` | Индекс истории |

`*.parquet` не коммитятся и создаются локально.

### Дополнительные проверки

```powershell
python src/run_graph_table_threshold_sweep.py --input src/graph_table_test_data/raw_packet_test_1000_profiles.parquet --graph-top-k 1,2,3 --threshold-start 0.0 --threshold-stop 1.0 --threshold-step 0.01
```

```powershell
python src/build_graph_table_shap_report.py --pair-scores reports/model_eval/pair_scores_<timestamp>.parquet --packet src/graph_table_test_data/raw_packet_test_1000_profiles.parquet --score-threshold 0.95
```
