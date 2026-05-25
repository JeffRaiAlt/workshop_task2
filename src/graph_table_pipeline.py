from __future__ import annotations

import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from graph_table_definitions import (
    BEHAVIOR_EVIDENCE_FAMILIES,
    BLOCKING_ATOMIC_RULE_GROUPS,
    BLOCKING_ATOMIC_RULES,
    BLOCKING_COMPOSITE_RULE_GROUPS,
    BLOCKING_COMPOSITE_RULES,
    BlockFamily,
    EvidenceColumn,
    GEO_KEYS,
    MatchScope,
    MODEL_FEATURE_DESCRIPTIONS,
    MODEL_FEATURES,
    PACKET_IDENTITY_COLUMNS,
    PAIR_FEATURES,
    REGISTRATION_TIME_FAMILY,
    STRONG_FAMILIES,
    TIME_AWARE_FAMILIES,
    TransformName,
    WEAK_FAMILIES,
    describe_blocking_rule,
    rule_family_from_name,
)
from graph_table_utils import (
    daypart_bucket,
    load_json,
    normalize_value,
    prefix6,
    registration_60m_bucket_values,
    stable_hash_bucket,
    weekend_bucket,
)


SRC_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SRC_DIR / "graph_table_artifacts"

BLOCK_MIN_SIZE = 2
BLOCK_MAX_SIZE = 1000


LOGGER = logging.getLogger("run_graph_table_inference")


# Добавляет одно плоское поле профиля в runtime long-слой.
def _add_flat_feature(rows: list[dict[str, str]], df: pd.DataFrame, source: str, feature: str, col: str) -> None:
    # Единая точка нормализации значения перед переводом в long-слой.
    if col not in df.columns:
        return
    for profile_id, value in df[["profile_id", col]].itertuples(index=False, name=None):
        norm = normalize_value(value, feature=feature)
        if norm is not None:
            rows.append({"profile_id": str(profile_id), "source": source, "feature": feature, "value_norm": norm})


# Разворачивает все колонки с префиксом np_/rt_/fs_ в признаки одного source.
def _add_prefixed_features(rows: list[dict[str, str]], df: pd.DataFrame, source: str) -> None:
    # Входные np_/rt_/fs_ колонки уже несут source в префиксе.
    prefix = f"{source}_"
    for col in df.columns:
        if not col.startswith(prefix) or col == f"{source}_features":
            continue
        feature = col[len(prefix):]
        _add_flat_feature(rows, df, source, feature, col)


# Собирает минимальную таблицу профилей для входного пакета.
def build_profile_core_from_flat(df: pd.DataFrame) -> pd.DataFrame:
    # Базовая таблица профилей: один profile_id и минимум служебной информации.
    if "profile_id" not in df.columns:
        raise ValueError("Input dataframe must contain profile_id")
    core = df.groupby("profile_id", as_index=False).size().rename(columns={"size": "event_count"})
    if "entity_id" in df.columns:
        # entity_id нужен только для обучения и проверки качества, не как признак.
        entity = (
            df[["profile_id", "entity_id"]]
            .dropna()
            .astype({"profile_id": "str", "entity_id": "str"})
            .drop_duplicates("profile_id")
        )
        core = core.merge(entity, on="profile_id", how="left")
    return core.astype({"profile_id": "str"})


# Собирает runtime long-слой значений для входного inference-пакета.
def build_packet_profile_values_from_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Собрать минимальный long-слой значений для входного inference-пакета."""
    # Переводим входной пакет в формат profile_id / source / feature / value_norm.
    # Такой формат одинаково удобен для blocking, similarity и аудита значений.
    # Это не mart-витрина из notebook 03, а легкая runtime-сборка для новых данных.
    if "profile_id" not in df.columns:
        raise ValueError("Input dataframe must contain profile_id")
    work = df.copy()
    work["profile_id"] = work["profile_id"].astype(str)
    rows: list[dict[str, str]] = []

    # Identity-поля остаются прямыми колонками профиля.
    for col in PACKET_IDENTITY_COLUMNS:
        _add_flat_feature(rows, work, "identity", col, col)

    # np_/rt_/fs_ не переименовываем: feature = часть имени после префикса.
    _add_prefixed_features(rows, work, "np")
    _add_prefixed_features(rows, work, "rt")
    _add_prefixed_features(rows, work, "fs")

    if "created_at" in work.columns:
        # Временное окно регистрации добавляем как derived-признак.
        created_at = work[["profile_id", "created_at"]].dropna().drop_duplicates()
        for profile_id, value in created_at.itertuples(index=False, name=None):
            rows.extend(registration_60m_bucket_values(profile_id, value))

    if not rows:
        return pd.DataFrame(columns=["profile_id", "source", "feature", "value_norm"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


# Возвращает готовый derived-признак или строит его из email.
def derived_feature_values(profile_values: pd.DataFrame, feature: str, colname: str) -> pd.DataFrame:
    # derived-признаки могут быть уже посчитаны раньше.
    stored = profile_values[
        (profile_values["source"].eq("derived")) & (profile_values["feature"].eq(feature))
    ][["profile_id", "value_norm"]].copy()
    if not stored.empty:
        stored[colname] = stored["value_norm"].astype(str)
        return stored[["profile_id", colname]].drop_duplicates()

    # Если derived-признака нет, восстанавливаем простые email-ключи на лету.
    emails = profile_values[
        (profile_values["source"].eq("identity")) & (profile_values["feature"].eq("email"))
    ][["profile_id", "value_norm"]].copy()
    if emails.empty:
        return pd.DataFrame(columns=["profile_id", colname])
    emails["email_norm"] = emails["value_norm"].astype(str).str.lower()
    emails["email_domain"] = emails["email_norm"].str.extract(r"@([^@]+)$", expand=False).fillna("missing_domain")
    emails["email_local"] = emails["email_norm"].str.extract(r"^([^@]+)", expand=False).fillna("missing_local")
    emails["email_initial2"] = emails["email_local"].str[:2].replace("", "missing")
    emails["email_hash_bucket_1024"] = emails["email_local"].map(lambda value: stable_hash_bucket(value, 1024))
    if feature not in emails.columns:
        return pd.DataFrame(columns=["profile_id", colname])
    return emails[["profile_id", feature]].rename(columns={feature: colname}).drop_duplicates()


# Достаёт значения одного признака из long-слоя.
def feature_values(profile_values: pd.DataFrame, source: str, feature: str, colname: str) -> pd.DataFrame:
    # Достаём значения одного признака в виде profile_id -> value.
    if source == "derived":
        return derived_feature_values(profile_values, feature, colname)
    sub = profile_values[
        (profile_values["source"].eq(source)) & (profile_values["feature"].eq(feature))
    ][["profile_id", "value_norm"]].copy()
    sub[colname] = sub["value_norm"].astype(str)
    return sub[["profile_id", colname]].drop_duplicates()


# Отсекает block_value, которые дают слишком маленькие или слишком большие блоки.
def apply_block_size_limits(
    sub: pd.DataFrame,
    block_value_col: str = "block_value",
    min_block_size: int = BLOCK_MIN_SIZE,
    max_block_size: int = BLOCK_MAX_SIZE,
) -> pd.DataFrame:
    # Размер блока = сколько разных профилей имеют один и тот же block_value.
    # Слишком маленькие и слишком большие блоки не используем как активные кандидаты.
    if sub.empty:
        return sub
    sizes = sub.groupby(block_value_col, sort=False)["profile_id"].nunique().rename("block_size").reset_index()
    keep_values = sizes[sizes["block_size"].between(min_block_size, max_block_size)][[block_value_col]]
    return sub.merge(keep_values, on=block_value_col, how="inner")


# Применяет ограничение размера блока только если оно включено.
def _maybe_limit_blocks(
    sub: pd.DataFrame,
    block_value_col: str,
    limit_blocks: bool,
    min_block_size: int,
    max_block_size: int,
) -> pd.DataFrame:
    if not limit_blocks:
        return sub
    return apply_block_size_limits(sub, block_value_col, min_block_size, max_block_size)


# Строит blocking-правило по одному признаку.
def atomic_block(
    profile_values: pd.DataFrame,
    rule: str,
    source: str,
    feature: str,
    family: str,
    transform=None,
    min_len: int = 1,
    min_block_size: int = BLOCK_MIN_SIZE,
    max_block_size: int = BLOCK_MAX_SIZE,
    limit_blocks: bool = True,
) -> pd.DataFrame:
    # Атомарное правило строится по одному признаку.
    sub = feature_values(profile_values, source, feature, "block_value")
    if sub.empty:
        return pd.DataFrame(columns=["profile_id", "block_family", "block_rule", "block_value"])
    if transform is not None:
        sub["block_value"] = sub["block_value"].map(transform)
    sub = sub[sub["block_value"].notna()]
    sub = sub[sub["block_value"].astype(str).str.len().ge(min_len)]
    sub = _maybe_limit_blocks(sub, "block_value", limit_blocks, min_block_size, max_block_size)
    sub["block_family"] = family
    sub["block_rule"] = rule
    return sub[["profile_id", "block_family", "block_rule", "block_value"]].drop_duplicates()


# Строит blocking-правило по комбинации нескольких признаков.
def composite_block(
    profile_values: pd.DataFrame,
    rule: str,
    specs: list[tuple[str, str, str, Any, int]],
    family: str,
    sep: str = "|",
    min_block_size: int = BLOCK_MIN_SIZE,
    max_block_size: int = BLOCK_MAX_SIZE,
    limit_blocks: bool = True,
) -> pd.DataFrame:
    # Композитное правило требует, чтобы у профиля были все компоненты ключа.
    out = None
    cols = []
    for source, feature, colname, transform, min_len in specs:
        # Каждый компонент отдельно нормализуется и фильтруется по минимальной длине.
        vals = feature_values(profile_values, source, feature, colname)
        if transform is not None:
            vals[colname] = vals[colname].map(transform)
        vals = vals[vals[colname].notna()]
        vals = vals[vals[colname].astype(str).str.len().ge(min_len)]
        out = vals if out is None else out.merge(vals, on="profile_id", how="inner")
        cols.append(colname)
    if out is None or out.empty:
        return pd.DataFrame(columns=["profile_id", "block_family", "block_rule", "block_value"])
    # Итоговый ключ блока — склейка компонент в стабильном порядке.
    out["block_value"] = out[cols].astype(str).agg(sep.join, axis=1)
    out = _maybe_limit_blocks(
        out[["profile_id", "block_value"]], "block_value", limit_blocks, min_block_size, max_block_size
    )
    out["block_family"] = family
    out["block_rule"] = rule
    return out[["profile_id", "block_family", "block_rule", "block_value"]].drop_duplicates()


TRANSFORM_BY_NAME = {
    None: None,
    TransformName.PREFIX6: prefix6,
    TransformName.DAYPART_BUCKET: daypart_bucket,
    TransformName.WEEKEND_BUCKET: weekend_bucket,
}


# Находит callable-функцию transform по имени из конфигурации rules.
def _resolve_transform(transform_name: str | None):
    if transform_name not in TRANSFORM_BY_NAME:
        raise ValueError(f"Unknown blocking transform: {transform_name}")
    return TRANSFORM_BY_NAME[transform_name]


# Подставляет feature в шаблон specs и заменяет имена transform на функции.
def _resolve_specs(specs: list[tuple[str, str, str, str | None, int]], feature: str | None = None):
    out = []
    for source, feature_name, colname, transform_name, min_len in specs:
        resolved_feature = feature if feature_name == "{feature}" else feature_name
        out.append((source, resolved_feature, colname, _resolve_transform(transform_name), min_len))
    return out


# Строит один атомарный блок из конфигурации.
def _build_atomic_rule(profile_values: pd.DataFrame, rule_config: dict[str, Any], limit_blocks: bool) -> pd.DataFrame:
    return atomic_block(
        profile_values,
        rule_config["rule"],
        rule_config["source"],
        rule_config["feature"],
        rule_config["family"],
        transform=_resolve_transform(rule_config.get("transform")),
        min_len=rule_config.get("min_len", 1),
        limit_blocks=limit_blocks,
    )


# Строит один композитный блок из конфигурации.
def _build_composite_rule(profile_values: pd.DataFrame, rule_config: dict[str, Any], limit_blocks: bool) -> pd.DataFrame:
    return composite_block(
        profile_values,
        rule_config["rule"],
        _resolve_specs(rule_config["specs"]),
        rule_config["family"],
        limit_blocks=limit_blocks,
    )


# Разворачивает группу однотипных атомарных правил по списку features.
def _build_atomic_rule_group(profile_values: pd.DataFrame, group_config: dict[str, Any], limit_blocks: bool) -> list[pd.DataFrame]:
    return [
        atomic_block(
            profile_values,
            group_config["rule_template"].format(feature=feature),
            group_config["source"],
            feature,
            group_config["family"],
            transform=_resolve_transform(group_config.get("transform")),
            min_len=group_config.get("min_len", 1),
            limit_blocks=limit_blocks,
        )
        for feature in group_config["features"]
    ]


# Разворачивает группу однотипных композитных правил по списку features.
def _build_composite_rule_group(profile_values: pd.DataFrame, group_config: dict[str, Any], limit_blocks: bool) -> list[pd.DataFrame]:
    return [
        composite_block(
            profile_values,
            group_config["rule_template"].format(feature=feature),
            _resolve_specs(group_config["spec_template"], feature=feature),
            group_config["family"],
            limit_blocks=limit_blocks,
        )
        for feature in group_config["features"]
    ]


# Собирает полный blocking index из конфигурации правил.
def build_blocking_index(
    profile_values: pd.DataFrame,
    recommended_rule_names: set[str] | None = None,
    limit_blocks: bool = True,
) -> pd.DataFrame:
    # Здесь собирается первый слой правил: какие профили вообще можно сравнивать.
    # На выходе many-to-many таблица profile_id -> правило -> значение блока.
    # Сами правила лежат в graph_table_definitions.py.
    # Здесь только применяем конфигурацию к profile_values.
    blocks = [_build_atomic_rule(profile_values, rule_config, limit_blocks) for rule_config in BLOCKING_ATOMIC_RULES]
    for group_config in BLOCKING_ATOMIC_RULE_GROUPS:
        blocks.extend(_build_atomic_rule_group(profile_values, group_config, limit_blocks))
    blocks.extend(
        _build_composite_rule(profile_values, rule_config, limit_blocks)
        for rule_config in BLOCKING_COMPOSITE_RULES
    )
    for group_config in BLOCKING_COMPOSITE_RULE_GROUPS:
        blocks.extend(_build_composite_rule_group(profile_values, group_config, limit_blocks))
    non_empty = [block for block in blocks if not block.empty]
    if not non_empty:
        return pd.DataFrame(columns=["profile_id", "block_family", "block_rule", "block_value"])
    # Склеиваем все правила в единый blocking index.
    out = pd.concat(non_empty, ignore_index=True).drop_duplicates()
    if recommended_rule_names is not None:
        # В production используем только зафиксированный список recommended-правил.
        out = out[out["block_rule"].isin(recommended_rule_names)].copy()
    return out.reset_index(drop=True)


# Добавляет к blocking index размер блока и вес блока.
def add_block_stats(index_df: pd.DataFrame) -> pd.DataFrame:
    # Добавляем размер блока и простой вес: маленький блок убедительнее большого.
    if index_df.empty:
        out = index_df.copy()
        out["block_size"] = pd.Series(dtype="int64")
        out["block_weight"] = pd.Series(dtype="float64")
        return out
    group_cols = ["block_family", "block_rule", "block_value"]
    sizes = index_df.groupby(group_cols, sort=False)["profile_id"].nunique().rename("block_size").reset_index()
    out = index_df.merge(sizes, on=group_cols, how="left")
    out["block_weight"] = 1.0 / np.log1p(out["block_size"].clip(lower=1))
    return out


# Готовит словари profile_id -> set(values) для similarity-признаков.
def make_value_maps(profile_values: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    # Для similarity нужны быстрые словари: profile_id -> множество значений.
    value_maps = {}
    for source, feature in PAIR_FEATURES:
        key = f"{source}__{feature}"
        sub = profile_values[
            (profile_values["source"].eq(source)) & (profile_values["feature"].eq(feature))
        ][["profile_id", "value_norm"]].drop_duplicates()
        value_maps[key] = sub.groupby("profile_id", observed=True)["value_norm"].agg(lambda values: set(map(str, values))).to_dict()
    return value_maps


# Разворачивает блоки внутри пакета в пары-кандидаты.
def pair_events_from_internal_index(index_df: pd.DataFrame) -> pd.DataFrame:
    # Пары внутри нового пакета: все профили, попавшие в один блок, становятся кандидатами.
    rows = []
    group_cols = ["block_family", "block_rule", "block_value"]
    grouped = index_df.groupby(group_cols, sort=False)
    total_blocks = grouped.ngroups
    started_at = time.monotonic()
    emitted_pairs = 0
    LOGGER.info("expand internal blocks total_blocks=%s index_rows=%s", f"{total_blocks:,}", f"{len(index_df):,}")
    for block_no, ((family, rule, value), grp) in enumerate(grouped, start=1):
        profiles = sorted(map(str, grp["profile_id"].unique()))
        if len(profiles) < 2:
            continue
        # Один блок размера N даёт N * (N - 1) / 2 пар-кандидатов.
        block_size = int(grp["block_size"].iloc[0])
        block_weight = float(grp["block_weight"].iloc[0])
        n_pairs = block_size * (block_size - 1) // 2
        emitted_pairs += n_pairs
        rows.extend((left, right, family, rule, value, block_size, block_weight, MatchScope.PACKET) for left, right in combinations(profiles, 2))
        if block_no == 1 or block_no % 100 == 0 or block_no == total_blocks:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            blocks_per_sec = block_no / elapsed
            remaining_blocks = total_blocks - block_no
            eta_sec = remaining_blocks / blocks_per_sec if blocks_per_sec > 0 else 0
            LOGGER.info(
                "expand internal blocks progress blocks=%s/%s emitted_pair_events=%s elapsed=%.1fs eta=%.1fs",
                f"{block_no:,}",
                f"{total_blocks:,}",
                f"{emitted_pairs:,}",
                elapsed,
                eta_sec,
            )
    LOGGER.info("expand internal blocks done emitted_pair_events=%s", f"{emitted_pairs:,}")
    return pd.DataFrame(
        rows,
        columns=["profile_id_l", "profile_id_r", "block_family", "block_rule", "block_value", "block_size", "block_weight", "match_scope"],
    )


# Находит пары новый профиль -> исторический профиль через совпавшие blocking-ключи.
def pair_events_from_history_lookup(
    packet_index: pd.DataFrame,
    historical_index: pd.DataFrame,
    exclude_profile_ids: set[str],
    max_candidates_per_profile: int | None = 300,
    max_total_candidate_pairs: int | None = 300_000,
) -> pd.DataFrame:
    # Пары с историей: новый профиль ищет исторические profile_id по тем же ключам blocking.
    if packet_index.empty or historical_index.empty:
        return pd.DataFrame(
            columns=["profile_id_l", "profile_id_r", "block_family", "block_rule", "block_value", "block_size", "block_weight", "match_scope"]
        )
    lookup_cols = ["block_family", "block_rule", "block_value"]
    hist = historical_index[lookup_cols + ["profile_id", "block_size", "block_weight"]].rename(
        columns={"profile_id": "profile_id_r"}
    )
    packet = packet_index[lookup_cols + ["profile_id"]].rename(columns={"profile_id": "profile_id_l"})
    # Join по block_family/block_rule/block_value создаёт candidate events.
    out = packet.merge(hist, on=lookup_cols, how="inner")
    LOGGER.info("history lookup raw events=%s", f"{len(out):,}")
    out["profile_id_l"] = out["profile_id_l"].astype(str)
    out["profile_id_r"] = out["profile_id_r"].astype(str)
    if exclude_profile_ids:
        out = out[~out["profile_id_r"].isin(exclude_profile_ids)]
    out = out[out["profile_id_l"] != out["profile_id_r"]].copy()
    out["match_scope"] = MatchScope.HISTORY
    out = limit_history_candidate_events(out, max_candidates_per_profile, max_total_candidate_pairs)
    LOGGER.info("history lookup limited events=%s", f"{len(out):,}")
    return out[
        ["profile_id_l", "profile_id_r", "block_family", "block_rule", "block_value", "block_size", "block_weight", "match_scope"]
    ].drop_duplicates()


# Ограничивает число исторических кандидатов после широкого lookup.
def limit_history_candidate_events(
    events: pd.DataFrame,
    max_candidates_per_profile: int | None,
    max_total_candidate_pairs: int | None,
) -> pd.DataFrame:
    """Ограничить поиск по истории, не меняя готовый исторический индекс."""
    # Широкие блоки могут дать слишком много исторических кандидатов.
    # Сначала ранжируем уникальные пары, потом возвращаем все события по оставшимся парам.
    if events.empty:
        return events
    if max_candidates_per_profile is None and max_total_candidate_pairs is None:
        return events

    work = events.copy()
    # pair_key нужен только как технический ключ для отбора.
    work["pair_key"] = work["profile_id_l"].astype(str) + "|" + work["profile_id_r"].astype(str)
    pair_rank = (
        work.groupby(["profile_id_l", "profile_id_r", "pair_key"], sort=False)
        .agg(
            min_block_size=("block_size", "min"),
            sum_block_weight=("block_weight", "sum"),
            n_block_rules=("block_rule", "nunique"),
            has_strong_family=("block_family", lambda s: int(bool(set(s) & STRONG_FAMILIES))),
        )
        .reset_index()
    )
    pair_rank = pair_rank.sort_values(
        ["profile_id_l", "has_strong_family", "min_block_size", "n_block_rules", "sum_block_weight", "profile_id_r"],
        ascending=[True, False, True, False, False, True],
    )
    # Лимит на один входной профиль защищает inference от одного очень шумного profile_id.
    if max_candidates_per_profile is not None:
        pair_rank = pair_rank.groupby("profile_id_l", sort=False).head(max_candidates_per_profile)
    # Общий лимит защищает весь запрос от слишком большого числа пар.
    if max_total_candidate_pairs is not None and len(pair_rank) > max_total_candidate_pairs:
        pair_rank = pair_rank.head(max_total_candidate_pairs)
    keep_pair_keys = set(pair_rank["pair_key"])
    return work[work["pair_key"].isin(keep_pair_keys)].drop(columns=["pair_key"])


# Считает пересечение и Jaccard similarity двух множеств значений.
def overlap(left_values: set[str], right_values: set[str]) -> tuple[int, float, int]:
    # Возвращаем пересечение и Jaccard: |A ∩ B| / |A ∪ B|.
    if not left_values or not right_values:
        return 0, 0.0, 0
    intersection = len(left_values & right_values)
    union = len(left_values | right_values)
    return intersection, intersection / union if union else 0.0, 1


# Добавляет к парам признаки сходства по fs/geo/identity значениям.
def add_similarity_features(pair_evidence: pd.DataFrame, value_maps: dict[str, dict[str, set[str]]]) -> pd.DataFrame:
    # Similarity-признаки считаются уже для готовых пар-кандидатов.
    C = EvidenceColumn
    for col, default_value in {
        C.FS_TOTAL_JACCARD: 0.0,
        C.GEO_TOTAL_JACCARD: 0.0,
        C.FS_SHARED_COUNT: 0,
        C.IDENTITY_EMAIL_MATCH: 0,
        C.IDENTITY_PHONE_MATCH: 0,
        C.IDENTITY_STRONG_MATCH: 0,
    }.items():
        pair_evidence[col] = default_value
    total = len(pair_evidence)
    started_at = time.monotonic()
    for row_no, (idx, row) in enumerate(pair_evidence.iterrows(), start=1):
        # Суммируем пересечения отдельно для поведения, географии и identity.
        fs_intersection = fs_union = geo_intersection = geo_union = 0
        identity_email_match = identity_phone_match = 0
        for key, profile_values in value_maps.items():
            left = profile_values.get(row[C.PROFILE_ID_L], set())
            right = profile_values.get(row[C.PROFILE_ID_R], set())
            intersection, _, _ = overlap(left, right)
            if key.startswith("fs__"):
                fs_intersection += intersection
                fs_union += len(left | right)
            if key in GEO_KEYS:
                geo_intersection += intersection
                geo_union += len(left | right)
            if key == "identity__email":
                identity_email_match = int(intersection > 0)
            if key == "identity__phone":
                identity_phone_match = int(intersection > 0)
        pair_evidence.at[idx, C.FS_TOTAL_JACCARD] = fs_intersection / fs_union if fs_union else 0.0
        pair_evidence.at[idx, C.GEO_TOTAL_JACCARD] = geo_intersection / geo_union if geo_union else 0.0
        pair_evidence.at[idx, C.FS_SHARED_COUNT] = fs_intersection
        pair_evidence.at[idx, C.IDENTITY_EMAIL_MATCH] = identity_email_match
        pair_evidence.at[idx, C.IDENTITY_PHONE_MATCH] = identity_phone_match
        pair_evidence.at[idx, C.IDENTITY_STRONG_MATCH] = int(identity_email_match or identity_phone_match)
        if row_no == 1 or row_no % 50_000 == 0 or row_no == total:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            rows_per_sec = row_no / elapsed
            eta_sec = (total - row_no) / rows_per_sec if rows_per_sec > 0 else 0
            LOGGER.info(
                "similarity progress pairs=%s/%s elapsed=%.1fs eta=%.1fs",
                f"{row_no:,}",
                f"{total:,}",
                elapsed,
                eta_sec,
            )
    return pair_evidence


# Схлопывает rule-events до одной строки evidence на пару профилей.
def aggregate_pair_events(pair_events: pd.DataFrame, value_maps: dict[str, dict[str, set[str]]]) -> pd.DataFrame:
    # Несколько правил могут породить одну и ту же пару.
    # Здесь схлопываем rule-events до одной строки на пару.
    C = EvidenceColumn
    if pair_events.empty:
        cols = [C.PROFILE_ID_L, C.PROFILE_ID_R, C.PAIR_KEY, C.RULES_KEY, *MODEL_FEATURES]
        return pd.DataFrame(columns=cols)
    pair_events = pair_events.copy()
    LOGGER.info("aggregate pair events rows=%s", f"{len(pair_events):,}")
    started_at = time.monotonic()
    # Канонический ключ пары уже направленный: left = профиль пакета, right = история или сосед.
    pair_events[C.PAIR_KEY] = pair_events[C.PROFILE_ID_L].astype(str) + "|" + pair_events[C.PROFILE_ID_R].astype(str)
    LOGGER.info("pair_key built elapsed=%.1fs", time.monotonic() - started_at)
    pair_evidence = aggregate_pair_events_chunked(pair_events, started_at=started_at, chunk_size=100_000)
    LOGGER.info(
        "chunked pair evidence done unique_candidate_pairs=%s elapsed=%.1fs",
        f"{len(pair_evidence):,}",
        time.monotonic() - started_at,
    )
    # Производные флаги описывают, какой тип evidence сработал у пары.
    pair_evidence[C.IS_FALLBACK_ONLY] = (
        pair_evidence[C.HIT_COVERAGE_FALLBACK].eq(1) & pair_evidence[C.N_BLOCK_FAMILIES].eq(1)
    ).astype("int8")
    pair_evidence[C.HAS_NON_FALLBACK_SIGNAL] = (
        pair_evidence[C.N_BLOCK_FAMILIES].gt(pair_evidence[C.HIT_COVERAGE_FALLBACK])
    ).astype("int8")
    pair_evidence[C.HAS_SMALL_BLOCK_LE2] = pair_evidence[C.N_SMALL_BLOCKS_LE2].gt(0).astype("int8")
    pair_evidence[C.HAS_SMALL_BLOCK_LE5] = pair_evidence[C.N_SMALL_BLOCKS_LE5].gt(0).astype("int8")
    pair_evidence[C.HAS_SMALL_BLOCK_LE10] = pair_evidence[C.N_SMALL_BLOCKS_LE10].gt(0).astype("int8")
    pair_evidence[C.SMALL_BLOCK_SHARE_LE5] = pair_evidence[C.N_SMALL_BLOCKS_LE5] / pair_evidence[C.N_BLOCK_HITS].clip(lower=1)
    pair_evidence[C.HAS_STRONG_FAMILY] = pair_evidence[C.FAMILIES].map(lambda x: int(bool(set(x) & STRONG_FAMILIES)))
    pair_evidence[C.STRONG_FAMILY_HIT_SHARE] = (
        pair_evidence[C.N_STRONG_FAMILY_HITS] / pair_evidence[C.N_BLOCK_HITS].clip(lower=1)
    )
    pair_evidence[C.HAS_BEHAVIOR] = pair_evidence[C.FAMILIES].map(lambda x: int(BlockFamily.BEHAVIOR in x))
    pair_evidence[C.HAS_BEHAVIOR_CONTEXT] = pair_evidence[C.FAMILIES].map(
        lambda x: int(
            bool(
                set(x)
                & {
                    BlockFamily.BEHAVIOR_CONTEXT,
                    BlockFamily.BEHAVIOR_CONTEXT_DEVICE,
                    BlockFamily.POSTMAN_CONTEXT,
                    *TIME_AWARE_FAMILIES,
                }
            )
        )
    )
    pair_evidence[C.HAS_POSTMAN_CONTEXT] = pair_evidence[C.FAMILIES].map(lambda x: int(BlockFamily.POSTMAN_CONTEXT in x))
    pair_evidence[C.POSTMAN_CONTEXT_HIT_SHARE] = (
        pair_evidence[C.N_POSTMAN_CONTEXT_HITS] / pair_evidence[C.N_BLOCK_HITS].clip(lower=1)
    )
    pair_evidence[C.HAS_TIME_AWARE_SIGNAL] = pair_evidence[C.FAMILIES].map(lambda x: int(bool(set(x) & TIME_AWARE_FAMILIES)))
    pair_evidence[C.HAS_TIME_AWARE_DEVICE_SIGNAL] = pair_evidence[C.FAMILIES].map(
        lambda x: int(BlockFamily.BEHAVIOR_DAYPART_DEVICE in x)
    )
    pair_evidence[C.TIME_AWARE_HIT_SHARE] = pair_evidence[C.N_TIME_AWARE_HITS] / pair_evidence[C.N_BLOCK_HITS].clip(lower=1)
    pair_evidence[C.HAS_REGISTRATION_TIME_WINDOW] = pair_evidence[C.FAMILIES].map(
        lambda x: int(REGISTRATION_TIME_FAMILY in x)
    )
    pair_evidence[C.REGISTRATION_TIME_WINDOW_HIT_SHARE] = (
        pair_evidence[C.N_REGISTRATION_TIME_WINDOW_HITS] / pair_evidence[C.N_BLOCK_HITS].clip(lower=1)
    )
    pair_evidence[C.REGISTRATION_TIME_WINDOW_ONLY] = pair_evidence[C.FAMILIES].map(
        lambda x: int(set(x) == {REGISTRATION_TIME_FAMILY})
    )
    pair_evidence[C.REGISTRATION_TIME_WINDOW_WITH_BEHAVIOR] = pair_evidence[C.FAMILIES].map(
        lambda x: int((REGISTRATION_TIME_FAMILY in x) and bool(set(x) & BEHAVIOR_EVIDENCE_FAMILIES))
    )
    pair_evidence[C.WEAK_GEO_TIME_ONLY] = pair_evidence[C.FAMILIES].map(
        lambda x: int(
            (REGISTRATION_TIME_FAMILY in x)
            and set(x).issubset({BlockFamily.CONTEXT, BlockFamily.COVERAGE_COMPOUND, REGISTRATION_TIME_FAMILY})
        )
    )
    pair_evidence[C.HAS_CONTEXT] = pair_evidence[C.FAMILIES].map(lambda x: int(BlockFamily.CONTEXT in x))
    pair_evidence[C.HAS_COVERAGE_COMPOUND] = pair_evidence[C.FAMILIES].map(lambda x: int(BlockFamily.COVERAGE_COMPOUND in x))
    pair_evidence[C.ONLY_WEAK_FAMILIES] = pair_evidence[C.FAMILIES].map(lambda x: int(set(x).issubset(WEAK_FAMILIES)))
    pair_evidence[C.RULES_KEY] = pair_evidence[C.RULES].map(lambda x: "|".join(sorted(x)))
    # Добавляем пересечения значений профилей поверх blocking evidence.
    pair_evidence = add_similarity_features(pair_evidence, value_maps)
    pair_evidence[C.REGISTRATION_TIME_WINDOW_FS_GAP] = (
        pair_evidence[C.HAS_REGISTRATION_TIME_WINDOW] * (1.0 - pair_evidence[C.FS_TOTAL_JACCARD].clip(lower=0.0, upper=1.0))
    )
    pair_evidence[C.SMALL_BLOCK_WEAK_FAMILY_ONLY] = (
        pair_evidence[C.HAS_SMALL_BLOCK_LE10].eq(1) & pair_evidence[C.ONLY_WEAK_FAMILIES].eq(1)
    ).astype("int8")
    return pair_evidence[[C.PROFILE_ID_L, C.PROFILE_ID_R, C.PAIR_KEY, C.RULES_KEY, C.MATCH_SCOPES, *MODEL_FEATURES]].copy()


# Считает, какие правила создали кандидатов и какие правила остались у принятых рёбер.
def summarize_rule_usage(pair_evidence: pd.DataFrame, accepted_edges: pd.DataFrame, top_n: int = 30) -> dict[str, Any]:
    """Показать, какие blocking-правила создали кандидатов и принятые рёбра."""
    # Отдельно показываем правила, которые создали кандидатов, и правила у принятых связей.
    if pair_evidence.empty or "rules_key" not in pair_evidence.columns:
        return {
            "candidate_rules": [],
            "accepted_rules": [],
            "candidate_families": [],
            "accepted_families": [],
        }

    def explode_rules(df: pd.DataFrame) -> pd.DataFrame:
        # rules_key хранит несколько правил через "|"; для отчёта разворачиваем их в строки.
        if df.empty:
            return pd.DataFrame(columns=["block_rule", "block_family"])
        rows = []
        for rules_key in df["rules_key"].fillna("").astype(str):
            for rule in rules_key.split("|"):
                if rule:
                    rows.append({"block_rule": rule, "block_family": rule_family_from_name(rule)})
        return pd.DataFrame(rows)

    def rule_counts(df: pd.DataFrame) -> list[dict[str, Any]]:
        exploded = explode_rules(df)
        if exploded.empty:
            return []
        counts = (
            exploded.groupby(["block_rule", "block_family"], sort=False)
            .size()
            .rename("pair_count")
            .reset_index()
            .sort_values("pair_count", ascending=False)
            .head(top_n)
        )
        return counts.to_dict(orient="records")

    def family_counts(df: pd.DataFrame) -> list[dict[str, Any]]:
        exploded = explode_rules(df)
        if exploded.empty:
            return []
        counts = (
            exploded.groupby("block_family", sort=False)
            .size()
            .rename("pair_count")
            .reset_index()
            .sort_values("pair_count", ascending=False)
            .head(top_n)
        )
        return counts.to_dict(orient="records")

    return {
        "candidate_rules": rule_counts(pair_evidence),
        "accepted_rules": rule_counts(accepted_edges),
        "candidate_families": family_counts(pair_evidence),
        "accepted_families": family_counts(accepted_edges),
    }


# Агрегирует события пар кусками, чтобы видеть прогресс на больших данных.
def aggregate_pair_events_chunked(
    pair_events: pd.DataFrame,
    started_at: float,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Агрегировать события пар кусками с видимым прогрессом."""
    C = EvidenceColumn
    # На десятках миллионов строк один groupby толи работает толи уже висит.
    # Поэтому считаем кусками и логируем прогресс.
    #
    # Почему так решил:
    # 1. Каждая строка pair_events - одно срабатывание правила для пары.
    # 2. Все признаки ниже считаются через операции, которые можно объединять по частям:
    #    sum, min, max, count и union множеств.
    # 3. Поэтому результат "groupby по всему датафрейму" равен схеме:
    #    groupby внутри кусков -> concat частичных итогов -> повторный groupby.
    # 4. Средние и доли считаем позже из уже объединённых сумм, поэтому они не искажаются.
    if C.PAIR_KEY not in pair_events.columns:
        pair_events = pair_events.copy()
        pair_events[C.PAIR_KEY] = pair_events[C.PROFILE_ID_L].astype(str) + "|" + pair_events[C.PROFILE_ID_R].astype(str)
    pair_events = pair_events.copy()
    # Служебные индикаторы нужны, чтобы быстро посчитать hit-count по семействам правил.
    pair_events[C.IS_TIME_AWARE_FAMILY] = pair_events[C.BLOCK_FAMILY].isin(TIME_AWARE_FAMILIES).astype("int8")
    pair_events[C.IS_REGISTRATION_TIME_WINDOW] = pair_events[C.BLOCK_FAMILY].eq(REGISTRATION_TIME_FAMILY).astype("int8")
    pair_events[C.IS_BEHAVIOR_DAYPART_DEVICE] = pair_events[C.BLOCK_FAMILY].eq(BlockFamily.BEHAVIOR_DAYPART_DEVICE).astype("int8")
    pair_events[C.IS_POSTMAN_CONTEXT] = pair_events[C.BLOCK_FAMILY].eq(BlockFamily.POSTMAN_CONTEXT).astype("int8")
    pair_events[C.IS_STRONG_FAMILY] = pair_events[C.BLOCK_FAMILY].isin(STRONG_FAMILIES).astype("int8")
    pair_events[C.IS_WEAK_FAMILY] = pair_events[C.BLOCK_FAMILY].isin(WEAK_FAMILIES).astype("int8")
    pair_events[C.TIME_AWARE_BLOCK_WEIGHT] = pair_events[C.BLOCK_WEIGHT] * pair_events[C.IS_TIME_AWARE_FAMILY]
    pair_events[C.REGISTRATION_TIME_WINDOW_BLOCK_WEIGHT] = (
        pair_events[C.BLOCK_WEIGHT] * pair_events[C.IS_REGISTRATION_TIME_WINDOW]
    )
    partials = []
    total_rows = len(pair_events)
    LOGGER.info("chunked pair aggregation start chunk_size=%s", f"{chunk_size:,}")
    for start in range(0, total_rows, chunk_size):
        stop = min(start + chunk_size, total_rows)
        chunk = pair_events.iloc[start:stop]
        # Частичная агрегация: одна строка на пару внутри текущего куска.
        partial = (
            chunk.groupby([C.PROFILE_ID_L, C.PROFILE_ID_R, C.PAIR_KEY], sort=False)
            .agg(
                **{
                    C.N_BLOCK_HITS: (C.BLOCK_RULE, "size"),
                    C.MIN_BLOCK_SIZE: (C.BLOCK_SIZE, "min"),
                    C.SUM_BLOCK_WEIGHT: (C.BLOCK_WEIGHT, "sum"),
                    C.N_SMALL_BLOCKS_LE2: (C.BLOCK_SIZE, lambda s: int((s <= 2).sum())),
                    C.N_SMALL_BLOCKS_LE5: (C.BLOCK_SIZE, lambda s: int((s <= 5).sum())),
                    C.N_SMALL_BLOCKS_LE10: (C.BLOCK_SIZE, lambda s: int((s <= 10).sum())),
                    C.HIT_COVERAGE_FALLBACK: (C.BLOCK_FAMILY, lambda s: int((s == BlockFamily.COVERAGE_FALLBACK).any())),
                    C.N_STRONG_FAMILY_HITS: (C.IS_STRONG_FAMILY, "sum"),
                    C.N_WEAK_FAMILY_HITS: (C.IS_WEAK_FAMILY, "sum"),
                    C.N_TIME_AWARE_HITS: (C.IS_TIME_AWARE_FAMILY, "sum"),
                    C.N_REGISTRATION_TIME_WINDOW_HITS: (C.IS_REGISTRATION_TIME_WINDOW, "sum"),
                    C.N_POSTMAN_CONTEXT_HITS: (C.IS_POSTMAN_CONTEXT, "sum"),
                    C.HIT_BEHAVIOR_DAYPART_DEVICE: (C.IS_BEHAVIOR_DAYPART_DEVICE, "max"),
                    C.SUM_TIME_AWARE_BLOCK_WEIGHT: (C.TIME_AWARE_BLOCK_WEIGHT, "sum"),
                    C.SUM_REGISTRATION_TIME_WINDOW_BLOCK_WEIGHT: (C.REGISTRATION_TIME_WINDOW_BLOCK_WEIGHT, "sum"),
                    C.RULES: (C.BLOCK_RULE, lambda s: frozenset(s)),
                    C.FAMILIES: (C.BLOCK_FAMILY, lambda s: frozenset(s)),
                    C.MATCH_SCOPE_SET: (C.MATCH_SCOPE, lambda s: frozenset(map(str, s))),
                }
            )
            .reset_index()
        )
        partials.append(partial)
        processed = stop
        elapsed = max(time.monotonic() - started_at, 1e-9)
        rows_per_sec = processed / elapsed
        eta_sec = (total_rows - processed) / rows_per_sec if rows_per_sec > 0 else 0
        LOGGER.info(
            "chunked pair aggregation progress rows=%s/%s partial_unique_pairs=%s chunks=%s elapsed=%.1fs eta=%.1fs",
            f"{processed:,}",
            f"{total_rows:,}",
            f"{len(partial):,}",
            f"{len(partials):,}",
            elapsed,
            eta_sec,
        )

    LOGGER.info("concat partial pair aggregates chunks=%s", f"{len(partials):,}")
    combined = pd.concat(partials, ignore_index=True)
    LOGGER.info("merge partial pair aggregates rows=%s", f"{len(combined):,}")

    def union_frozensets(values: pd.Series) -> frozenset:
        # Правила и семейства нужно объединять без дублей между кусками.
        out = set()
        for value_set in values:
            out.update(value_set)
        return frozenset(out)

    # Финальная агрегация объединяет частичные результаты в одну строку на пару.
    pair_evidence = (
        combined.groupby([C.PROFILE_ID_L, C.PROFILE_ID_R, C.PAIR_KEY], sort=False)
        .agg(
            **{
                C.N_BLOCK_HITS: (C.N_BLOCK_HITS, "sum"),
                C.MIN_BLOCK_SIZE: (C.MIN_BLOCK_SIZE, "min"),
                C.SUM_BLOCK_WEIGHT: (C.SUM_BLOCK_WEIGHT, "sum"),
                C.N_SMALL_BLOCKS_LE2: (C.N_SMALL_BLOCKS_LE2, "sum"),
                C.N_SMALL_BLOCKS_LE5: (C.N_SMALL_BLOCKS_LE5, "sum"),
                C.N_SMALL_BLOCKS_LE10: (C.N_SMALL_BLOCKS_LE10, "sum"),
                C.HIT_COVERAGE_FALLBACK: (C.HIT_COVERAGE_FALLBACK, "max"),
                C.N_STRONG_FAMILY_HITS: (C.N_STRONG_FAMILY_HITS, "sum"),
                C.N_WEAK_FAMILY_HITS: (C.N_WEAK_FAMILY_HITS, "sum"),
                C.N_TIME_AWARE_HITS: (C.N_TIME_AWARE_HITS, "sum"),
                C.N_REGISTRATION_TIME_WINDOW_HITS: (C.N_REGISTRATION_TIME_WINDOW_HITS, "sum"),
                C.N_POSTMAN_CONTEXT_HITS: (C.N_POSTMAN_CONTEXT_HITS, "sum"),
                C.HIT_BEHAVIOR_DAYPART_DEVICE: (C.HIT_BEHAVIOR_DAYPART_DEVICE, "max"),
                C.SUM_TIME_AWARE_BLOCK_WEIGHT: (C.SUM_TIME_AWARE_BLOCK_WEIGHT, "sum"),
                C.SUM_REGISTRATION_TIME_WINDOW_BLOCK_WEIGHT: (C.SUM_REGISTRATION_TIME_WINDOW_BLOCK_WEIGHT, "sum"),
                C.RULES: (C.RULES, union_frozensets),
                C.FAMILIES: (C.FAMILIES, union_frozensets),
                C.MATCH_SCOPE_SET: (C.MATCH_SCOPE_SET, union_frozensets),
            }
        )
        .reset_index()
    )
    pair_evidence[C.N_BLOCK_RULES] = pair_evidence[C.RULES].map(len)
    pair_evidence[C.N_BLOCK_FAMILIES] = pair_evidence[C.FAMILIES].map(len)
    pair_evidence[C.MATCH_SCOPES] = pair_evidence[C.MATCH_SCOPE_SET].map(lambda s: "|".join(sorted(s)))
    return pair_evidence.drop(columns=[C.MATCH_SCOPE_SET])


# Превращает score модели в рёбра графа через порог и mutual top-K.
def apply_mutual_top_k(pair_scores: pd.DataFrame, score_col: str, threshold: float, top_k: int | None) -> pd.DataFrame:
    # Сначала оставляем пары выше порога модели.
    edge_df = pair_scores[pair_scores[score_col].ge(threshold)].copy()
    if top_k is None or edge_df.empty:
        return edge_df
    # Для mutual top-K смотрим на пару с обеих сторон: A -> B и B -> A.
    left_view = edge_df[["profile_id_l", "profile_id_r", score_col]].rename(
        columns={"profile_id_l": "profile_id", "profile_id_r": "neighbor_id", score_col: "edge_score"}
    )
    right_view = edge_df[["profile_id_r", "profile_id_l", score_col]].rename(
        columns={"profile_id_r": "profile_id", "profile_id_l": "neighbor_id", score_col: "edge_score"}
    )
    directed = pd.concat([left_view, right_view], ignore_index=True)
    # Для каждого профиля оставляем top-K самых сильных соседей.
    directed = (
        directed.sort_values(["profile_id", "edge_score", "neighbor_id"], ascending=[True, False, True])
        .groupby("profile_id", sort=False)
        .head(top_k)
    )
    directed_pairs = set(zip(directed["profile_id"], directed["neighbor_id"]))
    # Ребро принимаем только если оба профиля выбрали друг друга.
    keep = [
        ((left, right) in directed_pairs) and ((right, left) in directed_pairs)
        for left, right in edge_df[["profile_id_l", "profile_id_r"]].itertuples(index=False, name=None)
    ]
    return edge_df.loc[keep].copy()


# Загружает production-артефакты: модель, историю, индекс, config и список правил.
def load_artifacts(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    # Все production-артефакты лежат в одном каталоге: модель, признаки, история и config.
    artifact_dir = Path(artifact_dir)
    LOGGER.info("load artifacts from=%s", artifact_dir)
    config = load_json(artifact_dir / "policy_config.json")
    artifacts = {
        "artifact_dir": artifact_dir,
        "config": config,
        "model": joblib.load(artifact_dir / config["model_file"]),
        "feature_cols": load_json(artifact_dir / "feature_cols.json"),
        "historical_core": pd.read_parquet(artifact_dir / "historical_profile_core.parquet"),
        "historical_values": pd.read_parquet(artifact_dir / "historical_profile_values.parquet"),
        "historical_index": pd.read_parquet(artifact_dir / "historical_blocking_index.parquet"),
        "recommended_rule_names": set(load_json(artifact_dir / "recommended_rule_names.json")),
    }
    LOGGER.info(
        "artifacts loaded historical_profiles=%s historical_values=%s historical_index_rows=%s model=%s",
        f"{len(artifacts['historical_core']):,}",
        f"{len(artifacts['historical_values']):,}",
        f"{len(artifacts['historical_index']):,}",
        config.get("policy_name"),
    )
    return artifacts


# Запускает полный inference для одного входного пакета профилей.
def score_packet(
    packet_core: pd.DataFrame,
    packet_values: pd.DataFrame,
    artifacts: dict[str, Any],
    score_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # Главная функция inference: входной пакет -> пары -> score -> граф -> assignment.
    packet_core = packet_core.copy()
    packet_values = packet_values.copy()
    packet_core["profile_id"] = packet_core["profile_id"].astype(str)
    packet_values["profile_id"] = packet_values["profile_id"].astype(str)
    packet_profile_ids = set(packet_core["profile_id"])

    config = artifacts["config"]
    threshold = float(score_threshold if score_threshold is not None else config.get("score_threshold", 0.95))
    # После score каждый профиль оставляет до K лучших соседей; ребро графа
    # принимается только при взаимном выборе. Меньший K осторожнее объединяет.
    top_k = config.get("graph_top_k", 1)

    LOGGER.info("build packet blocking keys profiles=%s values=%s", f"{len(packet_core):,}", f"{len(packet_values):,}")
    # lookup_index без отсечения нужен, чтобы не потерять совпадения с уже готовой историей.
    packet_lookup_index = build_blocking_index(packet_values, artifacts["recommended_rule_names"], limit_blocks=False)
    # internal_index с отсечением нужен для пар внутри самого пакета.
    packet_internal_index = add_block_stats(
        build_blocking_index(packet_values, artifacts["recommended_rule_names"], limit_blocks=True)
    )
    LOGGER.info(
        "packet blocking rows lookup=%s internal_limited=%s",
        f"{len(packet_lookup_index):,}",
        f"{len(packet_internal_index):,}",
    )

    LOGGER.info("lookup historical candidates")
    # Ищем кандидатов среди исторических профилей по совпавшим blocking-ключам.
    history_events = pair_events_from_history_lookup(
        packet_lookup_index,
        artifacts["historical_index"],
        exclude_profile_ids=packet_profile_ids,
        # Защита от шумного profile_id: ограничиваем число исторических
        # кандидатов одного входного профиля до расчета pair-признаков.
        max_candidates_per_profile=config.get("max_history_candidates_per_profile", 300),
        # Защита всего запроса: ограничиваем общее число пар с историей,
        # которые перейдут к расчету признаков и XGBoost.
        max_total_candidate_pairs=config.get("max_candidate_pairs_per_request", 300_000),
    )
    LOGGER.info("build packet internal candidate events")
    # Отдельно строим кандидатов между профилями внутри входного пакета.
    packet_events = pair_events_from_internal_index(packet_internal_index)
    LOGGER.info("packet internal events=%s", f"{len(packet_events):,}")
    pair_events = pd.concat([history_events, packet_events], ignore_index=True)
    LOGGER.info("total pair events=%s", f"{len(pair_events):,}")

    LOGGER.info("build value maps")
    # Similarity требует значения и из истории, и из нового пакета.
    combined_values = pd.concat([artifacts["historical_values"], packet_values], ignore_index=True).drop_duplicates()
    value_maps = make_value_maps(combined_values)
    LOGGER.info("aggregate evidence and similarity")
    pair_evidence = aggregate_pair_events(pair_events, value_maps)
    if pair_evidence.empty:
        # Если кандидатов нет, каждый профиль считается новым клиентом.
        assignment = packet_core[["profile_id"]].copy()
        assignment["decision"] = "new_client"
        assignment["predicted_entity_id"] = pd.NA
        assignment["best_match_score"] = 0.0
        return assignment, pair_evidence, {
            "profiles": len(packet_core),
            "candidate_pairs": 0,
            "accepted_edges": 0,
            "rule_usage": summarize_rule_usage(pair_evidence, pair_evidence),
        }

    feature_cols = artifacts["feature_cols"]
    x = pair_evidence.copy()
    # Совместимость с артефактами: отсутствующие признаки заполняем нулями.
    for col in feature_cols:
        if col not in x.columns:
            x[col] = 0
    x = x[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    LOGGER.info("score candidate pairs rows=%s features=%s", f"{len(x):,}", f"{len(feature_cols):,}")
    # Модель возвращает вероятность, что пара относится к одному клиенту.
    pair_evidence["score"] = artifacts["model"].predict_proba(x)[:, 1]
    LOGGER.info(
        "score summary min=%.6f p50=%.6f p95=%.6f max=%.6f threshold=%.6f",
        float(pair_evidence["score"].min()),
        float(pair_evidence["score"].quantile(0.50)),
        float(pair_evidence["score"].quantile(0.95)),
        float(pair_evidence["score"].max()),
        threshold,
    )
    # Из вероятностей делаем рёбра графа: порог + mutual top-K.
    edges = apply_mutual_top_k(pair_evidence, "score", threshold, top_k)
    LOGGER.info("accepted graph edges=%s top_k=%s", f"{len(edges):,}", top_k)
    rule_usage = summarize_rule_usage(pair_evidence, edges)
    LOGGER.info(
        "rule usage candidate_rules=%s accepted_rules=%s",
        f"{len(rule_usage['candidate_rules']):,}",
        f"{len(rule_usage['accepted_rules']):,}",
    )

    LOGGER.info("build assignment table")
    # Connected components превращают принятые рёбра в финальные группы.
    assignment = build_assignment_table(packet_core, artifacts["historical_core"], edges, pair_evidence)
    metrics = {
        "profiles": len(packet_core),
        "candidate_pairs": len(pair_evidence),
        "accepted_edges": len(edges),
        "threshold": threshold,
        "graph_top_k": top_k,
        "matched_existing": int(assignment["decision"].eq("matched_existing").sum()),
        "new_clients": int(assignment["decision"].eq("new_client").sum()),
        "ambiguous": int(assignment["decision"].eq("ambiguous").sum()),
        "model_mode": config.get("policy_name", "graph_table_edge_policy"),
        "rule_usage": rule_usage,
    }
    return assignment, pair_evidence.sort_values("score", ascending=False).reset_index(drop=True), metrics


# Превращает принятые рёбра графа в решение по каждому профилю пакета.
def build_assignment_table(
    packet_core: pd.DataFrame,
    historical_core: pd.DataFrame,
    edges: pd.DataFrame,
    pair_scores: pd.DataFrame,
) -> pd.DataFrame:
    # Граф содержит профили из пакета и исторические профили, найденные как кандидаты.
    packet_ids = set(packet_core["profile_id"].astype(str))
    historical = historical_core.copy()
    historical["profile_id"] = historical["profile_id"].astype(str)
    # entity_id берём только у истории: на новом пакете в production его нет.
    historical_entity = dict(zip(historical["profile_id"], historical.get("entity_id", pd.Series(dtype="object"))))

    graph = nx.Graph()
    graph.add_nodes_from(packet_ids)
    # Принятые пары становятся рёбрами графа.
    graph.add_edges_from(edges[["profile_id_l", "profile_id_r"]].itertuples(index=False, name=None))

    # Для отчёта по каждому профилю сохраняем лучший score среди его пар.
    best_scores = {}
    for row in pair_scores.itertuples(index=False):
        best_scores[row.profile_id_l] = max(float(row.score), best_scores.get(row.profile_id_l, 0.0))
        best_scores[row.profile_id_r] = max(float(row.score), best_scores.get(row.profile_id_r, 0.0))

    rows = []
    for component in nx.connected_components(graph):
        # Компонента связности = предполагаемая группа одного клиента.
        component_ids = sorted(map(str, component))
        packet_component_ids = [profile_id for profile_id in component_ids if profile_id in packet_ids]
        historical_ids = [profile_id for profile_id in component_ids if profile_id not in packet_ids]
        entity_ids = sorted({str(historical_entity[p]) for p in historical_ids if p in historical_entity and pd.notna(historical_entity[p])})
        if not entity_ids:
            # Исторических клиентов рядом нет: считаем новым клиентом.
            decision = "new_client"
            predicted_entity_id = pd.NA
        elif len(entity_ids) == 1:
            # В компоненте ровно один исторический клиент: приклеиваемся к нему.
            decision = "matched_existing"
            predicted_entity_id = entity_ids[0]
        else:
            # Несколько исторических клиентов в одной компоненте: конфликт, автосклейка опасна.
            decision = "ambiguous"
            predicted_entity_id = "|".join(entity_ids)
        for profile_id in packet_component_ids:
            rows.append(
                {
                    "profile_id": profile_id,
                    "decision": decision,
                    "predicted_entity_id": predicted_entity_id,
                    "best_match_score": best_scores.get(profile_id, 0.0),
                    "n_component_profiles": len(component_ids),
                    "matched_historical_profile_ids": "|".join(historical_ids),
                    "component_profile_ids": "|".join(component_ids),
                }
            )
    return pd.DataFrame(rows).sort_values(["decision", "best_match_score", "profile_id"], ascending=[True, False, True])


# Удобная обёртка для сервисного вызова по плоскому dataframe.
def run_graph_tree_table_inference(
    df: pd.DataFrame,
    split: str | None = None,
    score_threshold: float | None = None,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Обёртка для сервисного режима: принимает плоский dataframe и возвращает assignment.
    work = df.copy()
    if split is not None and "split" in work.columns:
        # split полезен для локальных проверок на подготовленных тестовых пакетах.
        work = work[work["split"].astype(str).eq(str(split))].copy()
    packet_core = build_profile_core_from_flat(work)
    packet_values = build_packet_profile_values_from_flat(work)
    artifacts = load_artifacts(artifact_dir)
    assignment, pair_scores, metrics = score_packet(packet_core, packet_values, artifacts, score_threshold)
    metrics["pairs"] = len(pair_scores)
    return assignment, metrics
