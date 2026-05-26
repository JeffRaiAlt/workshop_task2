from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
import zlib

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

import graph_table_pipeline as graph_pipeline
from graph_table_definitions import MODEL_FEATURES, PAIR_FEATURES
from graph_table_pipeline import (
    add_block_stats,
    aggregate_pair_events,
    make_value_maps,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MART_DIR = ROOT_DIR / "data" / "processed" / "er_profile_mart_multivalue"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "graph_table_artifacts"
DEFAULT_WORK_DIR = ROOT_DIR / "reports" / "model_eval" / "graph_table_build"

MODEL_FILE = "graph_edge_model.joblib"
MODEL_TYPE = "xgboost"
# Метка используется в именах отчётов; параметры выбранной модели хранятся отдельно.
POLICY_NAME = "graph_table_xgboost"
ARTIFACT_VERSION = "graph-table-runtime-v1"
TRAIN_SCOPE = "train"
VALID_SCOPE = "valid"
TRAIN_RANDOM_STATE = 42
VALID_RANDOM_STATE = 84
DEFAULT_MAX_TRAIN_NEGATIVE_PAIRS = 700_000
DEFAULT_MAX_VALID_NEGATIVE_PAIRS = 200_000
DEFAULT_SCORE_THRESHOLD = 0.95
DEFAULT_GRAPH_TOP_K = 1
DEFAULT_MAX_HISTORY_CANDIDATES_PER_PROFILE = 300
DEFAULT_MAX_CANDIDATE_PAIRS_PER_REQUEST = 1_500_000


XGBOOST_BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 43,
}
XGBOOST_TREE_CANDIDATES = [
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.90, "colsample_bytree": 0.90, "min_child_weight": 1.0},
    {"n_estimators": 450, "max_depth": 3, "learning_rate": 0.025, "subsample": 0.95, "colsample_bytree": 0.95, "min_child_weight": 1.0},
    {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1.0},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.90, "colsample_bytree": 0.85, "min_child_weight": 2.0},
    {"n_estimators": 650, "max_depth": 4, "learning_rate": 0.015, "subsample": 0.90, "colsample_bytree": 0.90, "min_child_weight": 2.0},
    {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.75, "min_child_weight": 3.0},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.80, "min_child_weight": 3.0},
    {"n_estimators": 450, "max_depth": 6, "learning_rate": 0.02, "subsample": 0.80, "colsample_bytree": 0.75, "min_child_weight": 5.0},
]
XGBOOST_CLASS_WEIGHT_MULTIPLIERS = [0.50, 0.75, 1.00, 1.25]


def stable_entity_split(entity_id: str) -> str:
    """Стабильно разделяет клиентов, чтобы один клиент не попал в разные split."""
    bucket = int(hashlib.md5(str(entity_id).encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def setup_logger(work_dir: Path) -> logging.Logger:
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"build_graph_table_artifacts_{timestamp}.log"

    logger = logging.getLogger("build_graph_table_artifacts")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("log_path=%s", log_path)
    return logger


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR).as_posix())
    except ValueError:
        return str(path)


def scoped_training_frames(
    historical_index: pd.DataFrame,
    historical_values: pd.DataFrame,
    profile_core: pd.DataFrame,
    data_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Берёт данные одного split для обучения или выбора параметров.

    `test` здесь запрещён: он должен оставаться независимым до финальной
    проверки инференса, иначе итоговые метрики будут завышены.
    """
    if data_scope not in {TRAIN_SCOPE, VALID_SCOPE}:
        raise ValueError(f"Unsupported training data scope: {data_scope}")
    scoped_profile_ids = set(
        profile_core.loc[
            profile_core["entity_id"].map(stable_entity_split).eq(data_scope),
            "profile_id",
        ].astype(str)
    )
    scoped_index = historical_index[historical_index["profile_id"].isin(scoped_profile_ids)].copy()
    scoped_values = historical_values[historical_values["profile_id"].isin(scoped_profile_ids)].copy()
    scoped_core = profile_core[profile_core["profile_id"].isin(scoped_profile_ids)].copy()
    return scoped_index, scoped_values, scoped_core


def profile_entity_map(profile_core: pd.DataFrame) -> dict[str, str]:
    """Строит lookup разметки: по нему пара получает целевой класс 0/1."""
    entity_frame = profile_core[["profile_id", "entity_id"]].copy()
    entity_frame["profile_id"] = entity_frame["profile_id"].astype(str)
    entity_frame["entity_id"] = entity_frame["entity_id"].astype(str)
    entity_frame = entity_frame[~entity_frame["entity_id"].str.lower().isin({"", "nan", "none", "null"})]
    return dict(zip(entity_frame["profile_id"], entity_frame["entity_id"]))


def add_historical_derived_values(profile_values: pd.DataFrame, profile_core: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные значения, которые рассчитываются и при inference.

    Витрина хранит исходные значения, а при поиске дополнительно строится
    временной ключ регистрации. Для истории строим такой же ключ из
    `profile_core.first_event_at`.
    """
    if "first_event_at" not in profile_core.columns:
        return profile_values

    rows = []
    times = profile_core[["profile_id", "first_event_at"]].dropna().copy()
    for profile_id, first_event_at in times.itertuples(index=False, name=None):
        rows.extend(graph_pipeline.registration_60m_bucket_values(profile_id, first_event_at))
    if not rows:
        return profile_values

    derived = pd.DataFrame(rows)
    return pd.concat(
        [profile_values[["profile_id", "source", "feature", "value_norm"]], derived],
        ignore_index=True,
    ).drop_duplicates()


def negative_cache_tag(max_negative_pairs: int) -> str:
    """Кодирует лимит отрицательных пар в имени кеша, чтобы не смешать запуски."""
    return "all_negatives" if max_negative_pairs <= 0 else f"neg_{max_negative_pairs}"


def pair_hash_score(left: str, right: str, seed: int) -> int:
    """Возвращает стабильный псевдослучайный порядок отбора отрицательных пар."""
    return zlib.crc32(f"{seed}|{left}|{right}".encode("utf-8")) & 0xFFFFFFFF


def heap_worst_score(heap: list[tuple[int, tuple[str, str]]], selected_scores: dict[tuple[str, str], int]) -> int:
    while heap:
        neg_score, key = heap[0]
        score = -neg_score
        if selected_scores.get(key) == score:
            return score
        heapq.heappop(heap)
    return -1


def maybe_keep_negative_pair(
    key: tuple[str, str],
    selected_scores: dict[tuple[str, str], int],
    heap: list[tuple[int, tuple[str, str]]],
    max_negative_pairs: int,
    random_state: int,
) -> None:
    if max_negative_pairs <= 0:
        selected_scores.setdefault(key, pair_hash_score(key[0], key[1], random_state))
        return
    if key in selected_scores:
        return

    score = pair_hash_score(key[0], key[1], random_state)
    if len(selected_scores) < max_negative_pairs:
        selected_scores[key] = score
        heapq.heappush(heap, (-score, key))
        return

    worst_score = heap_worst_score(heap, selected_scores)
    if worst_score >= 0 and score < worst_score:
        _, worst_key = heapq.heappop(heap)
        selected_scores.pop(worst_key, None)
        selected_scores[key] = score
        heapq.heappush(heap, (-score, key))


def select_training_pairs_from_blocks(
    scoped_index: pd.DataFrame,
    profile_to_entity: dict[str, str],
    max_negative_pairs: int,
    random_state: int,
    logger: logging.Logger,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, int]]:
    """Отбирает обучающие пары до дорогого расчёта их признаков.

    Все положительные пары сохраняются. Отрицательные пары отбираются при
    обходе блоков, чтобы не создавать полный огромный набор заведомо неверных пар.
    """
    group_cols = ["block_family", "block_rule", "block_value"]
    grouped = scoped_index.groupby(group_cols, sort=False)
    total_blocks = grouped.ngroups
    positive_pairs: set[tuple[str, str]] = set()
    negative_scores: dict[tuple[str, str], int] = {}
    negative_heap: list[tuple[int, tuple[str, str]]] = []
    raw_pair_events = 0
    started_at = time.monotonic()

    logger.info(
        "select training pairs from blocks total_blocks=%s max_negative_pairs=%s",
        f"{total_blocks:,}",
        "all" if max_negative_pairs <= 0 else f"{max_negative_pairs:,}",
    )
    for block_no, (_, grp) in enumerate(grouped, start=1):
        profiles = sorted(map(str, grp["profile_id"].unique()))
        if len(profiles) < 2:
            continue
        for left, right in combinations(profiles, 2):
            raw_pair_events += 1
            key = (left, right)
            left_entity = profile_to_entity.get(left)
            right_entity = profile_to_entity.get(right)
            if left_entity is not None and right_entity is not None and left_entity == right_entity:
                positive_pairs.add(key)
            else:
                maybe_keep_negative_pair(key, negative_scores, negative_heap, max_negative_pairs, random_state)

        if block_no == 1 or block_no % 100 == 0 or block_no == total_blocks:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            blocks_per_sec = block_no / elapsed
            eta_sec = (total_blocks - block_no) / blocks_per_sec if blocks_per_sec > 0 else 0
            logger.info(
                "select training pairs progress blocks=%s/%s raw_pair_events=%s positive_pairs=%s selected_negative_pairs=%s elapsed=%.1fs eta=%.1fs",
                f"{block_no:,}",
                f"{total_blocks:,}",
                f"{raw_pair_events:,}",
                f"{len(positive_pairs):,}",
                f"{len(negative_scores):,}",
                elapsed,
                eta_sec,
            )

    selected_negative_pairs = set(negative_scores)
    logger.info(
        "select training pairs done raw_pair_events=%s positive_pairs=%s selected_negative_pairs=%s",
        f"{raw_pair_events:,}",
        f"{len(positive_pairs):,}",
        f"{len(selected_negative_pairs):,}",
    )
    return positive_pairs, selected_negative_pairs, {"raw_pair_events": raw_pair_events, "total_blocks": total_blocks}


def collect_selected_pair_events(
    scoped_index: pd.DataFrame,
    selected_pairs: set[tuple[str, str]],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Возвращает все срабатывания правил только для уже отобранных пар.

    В первом проходе выбираются пары, чтобы ограничить объём отрицательного
    класса. Второй проход сохраняет все правила выбранной пары: без них
    невозможно посчитать evidence-признаки модели.
    """
    group_cols = ["block_family", "block_rule", "block_value"]
    grouped = scoped_index.groupby(group_cols, sort=False)
    total_blocks = grouped.ngroups
    rows = []
    emitted_events = 0
    started_at = time.monotonic()

    logger.info("collect selected pair events total_blocks=%s selected_pairs=%s", f"{total_blocks:,}", f"{len(selected_pairs):,}")
    for block_no, ((family, rule, value), grp) in enumerate(grouped, start=1):
        profiles = sorted(map(str, grp["profile_id"].unique()))
        if len(profiles) < 2:
            continue
        block_size = int(grp["block_size"].iloc[0])
        block_weight = float(grp["block_weight"].iloc[0])
        for left, right in combinations(profiles, 2):
            key = (left, right)
            if key not in selected_pairs:
                continue
            rows.append((left, right, family, rule, value, block_size, block_weight, "training"))
            emitted_events += 1

        if block_no == 1 or block_no % 100 == 0 or block_no == total_blocks:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            blocks_per_sec = block_no / elapsed
            eta_sec = (total_blocks - block_no) / blocks_per_sec if blocks_per_sec > 0 else 0
            logger.info(
                "collect selected pair events progress blocks=%s/%s emitted_events=%s elapsed=%.1fs eta=%.1fs",
                f"{block_no:,}",
                f"{total_blocks:,}",
                f"{emitted_events:,}",
                elapsed,
                eta_sec,
            )

    logger.info("collect selected pair events done emitted_events=%s", f"{emitted_events:,}")
    return pd.DataFrame(
        rows,
        columns=["profile_id_l", "profile_id_r", "block_family", "block_rule", "block_value", "block_size", "block_weight", "match_scope"],
    )


def sample_training_pairs(
    pair_evidence: pd.DataFrame,
    max_negative_pairs: int,
    random_state: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Формирует конечную таблицу для fit, сохраняя все положительные пары.

    Ограничиваются только отрицательные примеры: их очень много, а полная
    выборка существенно увеличивает время обучения без гарантии роста recall.
    Дополнительная отсечка нужна и при чтении ранее сохранённого кеша.
    """
    use_all_negative_pairs = max_negative_pairs <= 0
    positives = pair_evidence[pair_evidence["label"].eq(1)]
    negatives = pair_evidence[pair_evidence["label"].eq(0)]
    logger.info(
        "training pair evidence rows=%s positives=%s negatives=%s max_negative_pairs=%s",
        f"{len(pair_evidence):,}",
        f"{len(positives):,}",
        f"{len(negatives):,}",
        "all" if max_negative_pairs <= 0 else f"{max_negative_pairs:,}",
    )
    if not use_all_negative_pairs and len(negatives) > max_negative_pairs:
        logger.info("downsample negatives from=%s to=%s", f"{len(negatives):,}", f"{max_negative_pairs:,}")
        negatives = negatives.sample(n=max_negative_pairs, random_state=random_state)
    out = pd.concat([positives, negatives], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info(
        "training dataframe ready rows=%s positives=%s negatives=%s",
        f"{len(out):,}",
        f"{len(positives):,}",
        f"{len(out) - len(positives):,}",
    )
    return out


def xgboost_param_grid(scale_pos_weight: float) -> list[dict[str, object]]:
    """Возвращает варианты параметров XGBoost для проверки на validation-выборке.

    В парах намного больше несовпадений, чем совпадений. Поэтому проверяем
    не только параметры деревьев, но и вес положительного класса.
    Базовый вес равен `отрицательные / положительные`; множители позволяют
    validation-выборке выбрать более мягкую или более строгую поправку.
    """
    candidates = []
    for params in XGBOOST_TREE_CANDIDATES:
        for multiplier in XGBOOST_CLASS_WEIGHT_MULTIPLIERS:
            candidates.append(
                {
                    **XGBOOST_BASE_PARAMS,
                    **params,
                    "scale_pos_weight": scale_pos_weight * multiplier,
                    "scale_pos_weight_multiplier": multiplier,
                }
            )
    return candidates


def score_binary_metric(y_true: pd.Series, y_score: np.ndarray, metric: str) -> float:
    """Считает метрику выбора конфигурации XGBoost на validation-парах."""
    if y_true.nunique() < 2:
        return float("nan")
    if metric == "average_precision":
        return float(average_precision_score(y_true, y_score))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    raise ValueError(f"Unknown metric: {metric}")


def select_xgboost_model_on_valid(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float,
    work_dir: Path,
    logger: logging.Logger,
) -> tuple[object, dict[str, object], dict[str, object]]:
    """Обучает варианты XGBoost на train и выбирает лучший по validation AP.

    AP, или average precision, оценивает ранжирование пар на validation.
    Финальное качество объединений затем отдельно проверяется через inference.
    """
    best_model = None
    best_report: dict[str, object] | None = None
    rows = []
    for candidate_no, params in enumerate(xgboost_param_grid(scale_pos_weight), start=1):
        model_params = {key: value for key, value in params.items() if key != "scale_pos_weight_multiplier"}
        logger.info("fit XGBClassifier candidate=%s params=%s", candidate_no, json.dumps(params, sort_keys=True))
        model = XGBClassifier(**model_params)
        model.fit(x_train, y_train)
        valid_score = model.predict_proba(x_valid)[:, 1]
        ap = score_binary_metric(y_valid, valid_score, "average_precision")
        roc_auc = score_binary_metric(y_valid, valid_score, "roc_auc")
        report = {
            "candidate_no": candidate_no,
            "valid_average_precision": ap,
            "valid_roc_auc": roc_auc,
            "scale_pos_weight_multiplier": params["scale_pos_weight_multiplier"],
            **model_params,
        }
        rows.append(report)
        logger.info(
            "valid candidate=%s average_precision=%.6f roc_auc=%.6f",
            candidate_no,
            ap,
            roc_auc,
        )
        if best_report is None or (not np.isnan(ap) and ap > float(best_report["valid_average_precision"])):
            best_model = model
            best_report = report

    if best_model is None or best_report is None:
        raise RuntimeError("XGBoost validation tuning did not produce a model")

    report_path = work_dir / "xgboost_validation_tuning_report.csv"
    pd.DataFrame(rows).sort_values("valid_average_precision", ascending=False).to_csv(report_path, index=False)
    logger.info(
        "selected XGBoost candidate=%s valid_average_precision=%.6f valid_roc_auc=%.6f report_path=%s",
        best_report["candidate_no"],
        float(best_report["valid_average_precision"]),
        float(best_report["valid_roc_auc"]),
        report_path,
    )
    return best_model, dict(best_report), {"validation_tuning_report": relative_to_project(report_path)}


def build_training_pair_evidence(
    historical_index: pd.DataFrame,
    historical_values: pd.DataFrame,
    profile_core: pd.DataFrame,
    work_dir: Path,
    use_training_cache: bool,
    data_scope: str,
    max_negative_pairs: int,
    random_state: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    cache_path = work_dir / f"training_pair_evidence_{data_scope}_{negative_cache_tag(max_negative_pairs)}.parquet"
    if use_training_cache and cache_path.exists():
        logger.info("load central training pair evidence cache path=%s", cache_path)
        cached = pd.read_parquet(cache_path)
        missing_features = [col for col in MODEL_FEATURES if col not in cached.columns]
        if not missing_features:
            return cached
        logger.warning(
            "training pair evidence cache is missing model features=%s; rebuild cache from mart",
            missing_features,
        )

    scoped_index, scoped_values, scoped_core = scoped_training_frames(
        historical_index=historical_index,
        historical_values=historical_values,
        profile_core=profile_core,
        data_scope=data_scope,
    )

    logger.info("rebuild central training pair evidence from historical blocking_index")
    logger.info(
        "data_scope=%s scoped_profiles=%s scoped_index_rows=%s",
        data_scope,
        f"{scoped_core['profile_id'].nunique():,}",
        f"{len(scoped_index):,}",
    )

    profile_to_entity = profile_entity_map(scoped_core)
    positive_pairs, negative_pairs, selection_stats = select_training_pairs_from_blocks(
        scoped_index=scoped_index,
        profile_to_entity=profile_to_entity,
        max_negative_pairs=max_negative_pairs,
        random_state=random_state,
        logger=logger,
    )
    selected_pairs = positive_pairs | negative_pairs
    logger.info(
        "selected training pairs positive=%s negative=%s total=%s raw_pair_events_seen=%s",
        f"{len(positive_pairs):,}",
        f"{len(negative_pairs):,}",
        f"{len(selected_pairs):,}",
        f"{selection_stats['raw_pair_events']:,}",
    )
    pair_events = collect_selected_pair_events(scoped_index, selected_pairs, logger)
    logger.info("selected pair events rows=%s", f"{len(pair_events):,}")

    logger.info("build value maps for training evidence")
    value_maps = make_value_maps(scoped_values)
    logger.info("aggregate selected pair evidence and similarity from scratch")
    pair_evidence = aggregate_pair_events(pair_events, value_maps)

    left_entity = pair_evidence["profile_id_l"].map(profile_to_entity)
    right_entity = pair_evidence["profile_id_r"].map(profile_to_entity)
    pair_evidence["label"] = (left_entity.notna() & right_entity.notna() & left_entity.eq(right_entity)).astype("int8")

    keep_cols = ["profile_id_l", "profile_id_r", "pair_key", "rules_key", "match_scopes", "label", *MODEL_FEATURES]
    pair_evidence = pair_evidence[keep_cols].copy()
    logger.info(
        "central pair evidence ready rows=%s positives=%s negatives=%s",
        f"{len(pair_evidence):,}",
        f"{int(pair_evidence['label'].sum()):,}",
        f"{int(len(pair_evidence) - pair_evidence['label'].sum()):,}",
    )
    logger.info("save central training pair evidence cache path=%s", cache_path)
    pair_evidence.to_parquet(cache_path, index=False)
    return pair_evidence


def feature_matrix(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Готовит матрицу признаков в фиксированном порядке обученной модели."""
    return pair_df[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)


def train_xgboost_on_train_select_on_valid(
    historical_index: pd.DataFrame,
    historical_values: pd.DataFrame,
    profile_core: pd.DataFrame,
    work_dir: Path,
    use_training_cache: bool,
    max_negative_pairs: int,
    validation_negative_pairs: int,
    logger: logging.Logger,
) -> tuple[XGBClassifier, dict[str, object]]:
    """Обучает XGBoost на `train` и выбирает параметры только по `valid`.

    `entity_id` используется здесь как целевая разметка пары. Клиенты
    предварительно разделены целиком по split, поэтому одна сущность не
    появляется одновременно в обучении и проверке параметров.

    `test` в этой функции не используется. Его роль - проверить уже готовый
    pipeline: blocking, score, порог и финальные группы профилей.
    """
    logger.info("prepare train pairs: fit XGBoost only on train entities")
    train_pair_evidence = build_training_pair_evidence(
        historical_index=historical_index,
        historical_values=historical_values,
        profile_core=profile_core,
        work_dir=work_dir,
        use_training_cache=use_training_cache,
        data_scope=TRAIN_SCOPE,
        max_negative_pairs=max_negative_pairs,
        random_state=TRAIN_RANDOM_STATE,
        logger=logger,
    )
    train_df = sample_training_pairs(
        train_pair_evidence,
        max_negative_pairs,
        random_state=TRAIN_RANDOM_STATE,
        logger=logger,
    )
    x_train = feature_matrix(train_df)
    y_train = train_df["label"].astype(int)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    # В кандидатах отрицательных пар значительно больше положительных.
    # Базовый вес компенсирует дисбаланс, а его итоговый множитель выбирает valid.
    scale_pos_weight = negatives / max(positives, 1)

    logger.info("prepare valid pairs: select XGBoost configuration without fitting on valid entities")
    valid_pair_evidence = build_training_pair_evidence(
        historical_index=historical_index,
        historical_values=historical_values,
        profile_core=profile_core,
        work_dir=work_dir,
        use_training_cache=use_training_cache,
        data_scope=VALID_SCOPE,
        max_negative_pairs=validation_negative_pairs,
        random_state=VALID_RANDOM_STATE,
        logger=logger,
    )
    valid_df = sample_training_pairs(
        valid_pair_evidence,
        validation_negative_pairs,
        random_state=VALID_RANDOM_STATE,
        logger=logger,
    )
    x_valid = feature_matrix(valid_df)
    y_valid = valid_df["label"].astype(int)
    logger.info(
        "validation rows=%s positives=%s negatives=%s",
        f"{len(valid_df):,}",
        f"{int(y_valid.sum()):,}",
        f"{int(len(y_valid) - y_valid.sum()):,}",
    )

    # Average Precision выбирает конфигурацию на несбалансированных парах:
    # модель должна поднимать реальные совпадения выше множества ложных пар.
    model, selected_report, validation_report = select_xgboost_model_on_valid(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        scale_pos_weight=scale_pos_weight,
        work_dir=work_dir,
        logger=logger,
    )
    model_params = {
        key: value
        for key, value in selected_report.items()
        if key not in {"candidate_no", "valid_average_precision", "valid_roc_auc", "scale_pos_weight_multiplier"}
    }
    validation_report.update(
        {
            "hyperparameter_selection": "valid_average_precision",
            "validation_scope": VALID_SCOPE,
            "validation_rows": int(len(valid_df)),
            "validation_positive_rows": int(y_valid.sum()),
            "validation_negative_rows": int(len(y_valid) - y_valid.sum()),
            "selected_candidate_no": int(selected_report["candidate_no"]),
            "selected_valid_average_precision": float(selected_report["valid_average_precision"]),
            "selected_valid_roc_auc": float(selected_report["valid_roc_auc"]),
            "selected_scale_pos_weight_multiplier": float(selected_report["scale_pos_weight_multiplier"]),
        }
    )
    training_report = {
        "model_params": model_params,
        "training_rows": int(len(train_df)),
        "positive_rows": positives,
        "negative_rows": negatives,
        **validation_report,
    }
    return model, training_report


def build_artifacts(
    mart_dir: Path,
    out_dir: Path,
    work_dir: Path,
    max_negative_pairs: int,
    use_training_cache: bool,
    validation_negative_pairs: int,
    logger: logging.Logger,
) -> None:
    """Собирает runtime-историю и обучает единственную production-модель XGBoost.

    Последовательность:
    1. Прочитать готовую витрину и список разрешённых blocking-правил.
    2. Собрать индекс и значения истории, нужные inference.
    3. На парах `train` обучить XGBoost, на парах `valid` выбрать его параметры.
    4. Сохранить модель, runtime-настройки и паспорт обучения.

    `test` намеренно отсутствует в builder: он используется только после
    сборки артефактов в `run_graph_table_inference.py` и threshold sweep.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("start build graph-table artifacts")
    logger.info("mart_dir=%s", mart_dir)
    logger.info("out_dir=%s", out_dir)
    logger.info("work_dir=%s", work_dir)
    logger.info(
        "max_negative_pairs=%s validation_negative_pairs=%s",
        "all" if max_negative_pairs <= 0 else f"{max_negative_pairs:,}",
        "all" if validation_negative_pairs <= 0 else f"{validation_negative_pairs:,}",
    )
    logger.info("use_training_cache=%s", use_training_cache)
    logger.info("fit_scope=%s parameter_selection_scope=%s test_scope=offline_inference_only", TRAIN_SCOPE, VALID_SCOPE)

    logger.info("load mart inputs")
    profile_core = pd.read_parquet(mart_dir / "profile_core.parquet")
    profile_values = pd.read_parquet(mart_dir / "profile_value_summary_long.parquet")
    recommended_rules = pd.read_csv(mart_dir / "recommended_blocking_rules.csv")
    logger.info(
        "loaded profiles=%s values=%s rules=%s",
        f"{len(profile_core):,}",
        f"{len(profile_values):,}",
        f"{len(recommended_rules):,}",
    )

    for df in [profile_core, profile_values]:
        for col in ["profile_id", "entity_id", "block_family", "block_rule", "block_value", "source", "feature", "value_norm"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
    profile_values = add_historical_derived_values(profile_values, profile_core)
    logger.info("profile values with historical derived rows=%s", f"{len(profile_values):,}")

    recommended_rule_names = set(
        recommended_rules.loc[
            recommended_rules["recommended_for_next_step"].astype(str).str.lower().eq("true"),
            "block_rule",
        ].astype(str)
    )
    logger.info("recommended rules selected=%s", f"{len(recommended_rule_names):,}")

    # Индекс пересобирается из значений и утверждённых правил, потому что
    # runtime должен использовать тот же генератор ключей, что и новый пакет.
    logger.info("build historical blocking index with runtime rule generator")
    historical_index = graph_pipeline.build_blocking_index(
        profile_values,
        recommended_rule_names=recommended_rule_names,
        limit_blocks=True,
    )
    missing_runtime_rules = sorted(recommended_rule_names - set(historical_index["block_rule"].astype(str).unique()))
    if missing_runtime_rules:
        logger.warning("recommended rules produced no historical rows: %s", missing_runtime_rules)
    historical_index = add_block_stats(historical_index[["profile_id", "block_family", "block_rule", "block_value"]])
    logger.info("historical blocking rows=%s profiles=%s", f"{len(historical_index):,}", f"{historical_index['profile_id'].nunique():,}")

    historical_values = profile_values[
        # В артефакт значений кладём только поля, для которых считаем прямой
        # Jaccard/overlap. Значения, нужные лишь для blocking (например
        # postman), уже зафиксированы в historical_blocking_index.
        profile_values[["source", "feature"]].apply(tuple, axis=1).isin(set(PAIR_FEATURES))
    ][["profile_id", "source", "feature", "value_norm"]].drop_duplicates()
    logger.info("historical values rows=%s", f"{len(historical_values):,}")

    logger.info("write historical artifacts")
    profile_core_cols = [col for col in ["profile_id", "entity_id", "event_count", "entity_size", "entity_kind"] if col in profile_core.columns]
    profile_core[profile_core_cols].to_parquet(out_dir / "historical_profile_core.parquet", index=False)
    historical_values.to_parquet(out_dir / "historical_profile_values.parquet", index=False)
    historical_index.to_parquet(out_dir / "historical_blocking_index.parquet", index=False)
    write_json(out_dir / "recommended_rule_names.json", sorted(recommended_rule_names))
    write_json(out_dir / "feature_cols.json", MODEL_FEATURES)

    model, training_report = train_xgboost_on_train_select_on_valid(
        historical_index=historical_index,
        historical_values=historical_values,
        profile_core=profile_core,
        work_dir=work_dir,
        use_training_cache=use_training_cache,
        max_negative_pairs=max_negative_pairs,
        validation_negative_pairs=validation_negative_pairs,
        logger=logger,
    )
    logger.info("save XGBoost model path=%s", out_dir / MODEL_FILE)
    joblib.dump(model, out_dir / MODEL_FILE)

    created_at = datetime.now().isoformat(timespec="seconds")
    config = {
        "model_file": MODEL_FILE,
        "policy_name": POLICY_NAME,
        # Начальный рабочий порог. Для нового эксперимента его можно
        # переопределить в inference или проверить threshold sweep.
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        # После оценки моделью каждый профиль может выбрать не более K лучших соседей.
        # Связь попадёт в граф только при взаимном выборе двух профилей.
        # При K=1 граф осторожнее объединяет клиентов; рост K повышает полноту,
        # но также увеличивает риск ошибочных объединений через цепочки связей.
        "graph_top_k": DEFAULT_GRAPH_TOP_K,
        # При поиске в истории одно шумное совпадение правила не должно породить
        # слишком много сравнений для одного входного profile_id. Ограничение
        # применяется до расчёта признаков пары и запуска XGBoost.
        "max_history_candidates_per_profile": DEFAULT_MAX_HISTORY_CANDIDATES_PER_PROFILE,
        # Общий предохранитель на один входной пакет после персонального лимита:
        # параметр управляет временем и памятью поиска; слишком низкое значение
        # может отрезать правильного исторического кандидата.
        "max_candidate_pairs_per_request": DEFAULT_MAX_CANDIDATE_PAIRS_PER_REQUEST,
    }
    model_manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": created_at,
        "model_file": config["model_file"],
        "policy_name": config["policy_name"],
        "model_type": MODEL_TYPE,
        "model_params": training_report.get("model_params"),
        "model_trained": True,
        "training_scope": TRAIN_SCOPE,
        "training_rows": training_report.get("training_rows"),
        "positive_rows": training_report.get("positive_rows"),
        "negative_rows": training_report.get("negative_rows"),
        "validation_rows": training_report.get("validation_rows"),
        "validation_positive_rows": training_report.get("validation_positive_rows"),
        "validation_negative_rows": training_report.get("validation_negative_rows"),
        "selected_valid_average_precision": training_report.get("selected_valid_average_precision"),
        "selected_valid_roc_auc": training_report.get("selected_valid_roc_auc"),
    }
    model_manifest = {key: value for key, value in model_manifest.items() if value is not None}
    write_json(out_dir / "policy_config.json", config)
    write_json(out_dir / "artifact_manifest.json", model_manifest)
    logger.info("saved policy_config path=%s", out_dir / "policy_config.json")
    logger.info("saved artifact_manifest path=%s", out_dir / "artifact_manifest.json")
    if "validation_tuning_report" in training_report:
        logger.info("saved validation_tuning_report path=%s", training_report["validation_tuning_report"])
    logger.info("done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать модель и артефакты graph-table для inference.")
    parser.add_argument("--mart-dir", type=Path, default=DEFAULT_MART_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Каталог кэша обучения, отчёта подбора параметров и логов; inference его не использует.",
    )
    parser.add_argument(
        "--max-negative-pairs",
        type=int,
        default=DEFAULT_MAX_TRAIN_NEGATIVE_PAIRS,
        help="Максимум отрицательных пар в train; положительные пары сохраняются полностью.",
    )
    parser.add_argument(
        "--use-training-cache",
        action="store_true",
        help="Использовать готовые признаки обучающих пар из --work-dir, если файл уже существует. По умолчанию пересчитать.",
    )
    parser.add_argument(
        "--validation-negative-pairs",
        type=int,
        default=DEFAULT_MAX_VALID_NEGATIVE_PAIRS,
        help="Максимум отрицательных пар validation-выборки при выборе параметров XGBoost.",
    )
    args = parser.parse_args()
    logger = setup_logger(args.work_dir)
    graph_pipeline.LOGGER = logger
    build_artifacts(
        args.mart_dir,
        args.out_dir,
        args.work_dir,
        args.max_negative_pairs,
        args.use_training_cache,
        args.validation_negative_pairs,
        logger,
    )
    print(f"saved graph-table artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
