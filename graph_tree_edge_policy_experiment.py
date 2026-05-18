from __future__ import annotations

"""Graph + decision tree edge policy experiment.

Этот скрипт проверяет финальную для текущего исследования архитектуру:

1. Берём `blocking_index` из 03 ноутбука.
2. Оставляем только рекомендованные blocking-правила.
3. Внутри каждого блока генерируем candidate pairs.
4. Для каждой пары считаем evidence:
   - через какие blocking-правила пара была найдена;
   - насколько маленькими/точными были эти блоки;
   - какие семейства правил сработали.
5. Добавляем similarity-признаки по значениям профилей:
   - пересечения identity/fs/geo значений;
   - Jaccard similarity;
   - флаги сильных identity-совпадений.
6. В этой версии дерево обучается только на минимальном наборе объяснимых
   признаков: blocking evidence + similarity. Score базовой pair-модели и
   расширенные graph-rank признаки не используются как признаки дерева.
7. Обучаем DecisionTreeClassifier не как финальную ER-модель, а как edge policy:
   дерево решает, ставить ли ребро между двумя profile_id.
8. NetworkX собирает connected components; каждая компонента = найденный клиент.

Важно: дерево работает только на candidate pairs, которые пришли из blocking.
Если blocking не поставил настоящую пару в кандидаты, дерево и граф её уже не увидят.
"""

import hashlib
import pickle
from datetime import datetime
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


BASE = Path("data/processed")
MART_DIR = BASE / "er_profile_mart_multivalue"
PAIR_MODEL_DIR = BASE / "er_baseline_pair_model"
OUT_DIR = BASE / "er_graph_tree_policy"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    """Print a timestamped progress message and flush it immediately."""
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)



def apply_mutual_top_k(pair_scores: pd.DataFrame, score_col: str, threshold: float, top_k: int | None) -> pd.DataFrame:
    """Превратить scored candidate pairs в графовые ребра.

    Сначала оставляем пары со score >= threshold.
    Затем, если top_k задан, применяем mutual top-K:
    ребро остаётся только если левый профиль входит в top-K соседей правого
    и правый профиль входит в top-K соседей левого.

    Для текущего эксперимента top_k=1. Это консервативная защита от ситуаций,
    когда один популярный/шумный профиль притягивает много слабых соседей.
    """
    edge_df = pair_scores[pair_scores[score_col].ge(threshold)].copy()
    if top_k is None or edge_df.empty:
        return edge_df
    left_view = edge_df[["profile_id_l", "profile_id_r", score_col]].rename(
        columns={"profile_id_l": "profile_id", "profile_id_r": "neighbor_id", score_col: "edge_score"}
    )
    right_view = edge_df[["profile_id_r", "profile_id_l", score_col]].rename(
        columns={"profile_id_r": "profile_id", "profile_id_l": "neighbor_id", score_col: "edge_score"}
    )
    directed = pd.concat([left_view, right_view], ignore_index=True)
    directed = (
        directed.sort_values(["profile_id", "edge_score", "neighbor_id"], ascending=[True, False, True])
        .groupby("profile_id", sort=False)
        .head(top_k)
    )
    directed_pairs = set(zip(directed["profile_id"], directed["neighbor_id"]))
    keep = [
        ((left, right) in directed_pairs) and ((right, left) in directed_pairs)
        for left, right in edge_df[["profile_id_l", "profile_id_r"]].itertuples(index=False, name=None)
    ]
    return edge_df.loc[keep].copy()


def graph_components(edge_df: pd.DataFrame, split_profiles: list[str], profile_to_entity: dict[str, str]) -> pd.DataFrame:
    """Собрать граф и вернуть компоненты связности.

    Вершины графа: все profile_id выбранного split.
    Ребра графа: пары, которые прошли edge policy.

    NetworkX connected components дают финальное предсказание:
    если P1 связан с P2, а P2 связан с P3, все три профиля попадают
    в одну найденную группу, даже если прямого ребра P1-P3 нет.
    """
    graph = nx.Graph()
    graph.add_nodes_from(split_profiles)
    graph.add_edges_from(edge_df[["profile_id_l", "profile_id_r"]].itertuples(index=False, name=None))
    rows = []
    for component_id, component in enumerate(nx.connected_components(graph)):
        for profile_id in component:
            rows.append((profile_id, component_id, profile_to_entity[profile_id]))
    return pd.DataFrame(rows, columns=["profile_id", "graph_component_id", "entity_id"])


def entity_decision_metrics(components: pd.DataFrame, split_core: pd.DataFrame) -> dict[str, float]:
    """Посчитать entity-level метрики по компонентам графа.

    Это главные метрики для текущей постановки, потому что итоговое решение
    принимается не по отдельной паре, а по клиентской группе.

    Метрики:
    - duplicate_entity_recall: доля настоящих клиентов с несколькими профилями,
      у которых граф нашёл хотя бы два профиля вместе;
    - singleton_entity_precision: доля настоящих одиночек, которых граф оставил
      одиночками;
    - false_merge_rate: доля найденных непустых склеек, где внутри оказались
      разные entity_id;
    - overall_entity_decision_acc: общий summary по duplicate + singleton entity.
    """
    split_eval = split_core.merge(components[["profile_id", "graph_component_id"]], on="profile_id", how="left")
    split_eval["graph_component_id"] = split_eval["graph_component_id"].fillna(split_eval["profile_id"])
    entity_profiles = split_eval.groupby("entity_id")["profile_id"].agg(list).to_dict()
    entity_components = split_eval.groupby("entity_id")["graph_component_id"].agg(list).to_dict()
    component_profiles = split_eval.groupby("graph_component_id")["profile_id"].agg(list).to_dict()
    component_entities = split_eval.groupby("graph_component_id")["entity_id"].agg(list).to_dict()

    duplicate_entities = {e: ps for e, ps in entity_profiles.items() if len(ps) >= 2}
    singleton_entities = {e: ps for e, ps in entity_profiles.items() if len(ps) == 1}

    found_duplicate_entities = 0
    for entity_id, comps in entity_components.items():
        if entity_id not in duplicate_entities:
            continue
        if pd.Series(comps).value_counts().max() >= 2:
            found_duplicate_entities += 1

    component_size = {cid: len(ps) for cid, ps in component_profiles.items()}
    profile_to_component = dict(zip(split_eval["profile_id"], split_eval["graph_component_id"]))
    correct_singletons = 0
    for _, profiles in singleton_entities.items():
        if component_size[profile_to_component[profiles[0]]] == 1:
            correct_singletons += 1

    non_singleton_components = {cid: ents for cid, ents in component_entities.items() if len(ents) >= 2}
    false_merge_components = sum(1 for ents in non_singleton_components.values() if len(set(ents)) > 1)

    existing_clients_total = len(duplicate_entities)
    existing_clients_found = found_duplicate_entities
    existing_clients_missed = existing_clients_total - existing_clients_found
    new_clients_total = len(singleton_entities)
    new_clients_correct = correct_singletons
    new_clients_wrong = new_clients_total - new_clients_correct
    predicted_merge_groups = len(non_singleton_components)
    wrong_merge_groups = false_merge_components
    total_clients = existing_clients_total + new_clients_total
    correct_client_decisions = existing_clients_found + new_clients_correct

    return {
        "clients_total": total_clients,
        "existing_clients_total": existing_clients_total,
        "existing_clients_found": existing_clients_found,
        "existing_clients_found_pct": 100 * existing_clients_found / max(existing_clients_total, 1),
        "existing_clients_missed": existing_clients_missed,
        "existing_clients_missed_pct": 100 * existing_clients_missed / max(existing_clients_total, 1),
        "new_clients_total": new_clients_total,
        "new_clients_correct": new_clients_correct,
        "new_clients_correct_pct": 100 * new_clients_correct / max(new_clients_total, 1),
        "new_clients_wrongly_attached": new_clients_wrong,
        "new_clients_wrongly_attached_pct": 100 * new_clients_wrong / max(new_clients_total, 1),
        "predicted_merge_groups": predicted_merge_groups,
        "wrong_merge_groups": wrong_merge_groups,
        "wrong_merge_groups_pct": 100 * wrong_merge_groups / max(predicted_merge_groups, 1),
        "correct_client_decisions": correct_client_decisions,
        "correct_client_decisions_pct": 100 * correct_client_decisions / max(total_clients, 1),
    }


def evaluate_edge_policy(
    pair_scores: pd.DataFrame,
    split_core: pd.DataFrame,
    split_profiles: list[str],
    profile_to_entity: dict[str, str],
    score_col: str,
    threshold: float,
    policy_name: str,
) -> dict[str, float]:
    """Оценить один вариант edge policy на valid/test split.

    На вход подаём candidate pairs со score дерева или baseline-модели.
    Функция:
    1. применяет threshold и mutual top-K;
    2. строит граф;
    3. считает edge-level числа;
    4. считает entity-level метрики по connected components.
    """
    edge_df = apply_mutual_top_k(pair_scores, score_col, threshold, GRAPH_TOP_K)
    components = graph_components(edge_df, split_profiles, profile_to_entity)
    row = {
        "policy": policy_name,
        "threshold": threshold,
        "score_col": score_col,
        "candidate_pairs": len(pair_scores),
        "graph_edges": len(edge_df),
    }
    row.update(entity_decision_metrics(components, split_core))
    return row



def make_value_maps(profile_values: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    """Подготовить быстрые lookup-таблицы значений профиля.

    Из long-слоя `profile_value_summary_long` строим словари:
    `source__feature -> profile_id -> set(value_norm)`.

    Это нужно, чтобы быстро считать пересечения значений для пары профилей,
    не делая тяжёлые join-ы для каждой пары.
    """
    value_maps = {}
    for source, feature in PAIR_FEATURES:
        key = f"{source}__{feature}"
        sub = profile_values.loc[
            (profile_values["source"].eq(source)) & (profile_values["feature"].eq(feature)),
            ["profile_id", "value_norm"],
        ].drop_duplicates()
        value_maps[key] = sub.groupby("profile_id", observed=True)["value_norm"].agg(lambda values: set(map(str, values))).to_dict()
    return value_maps


def score_split_candidate_pairs(
    split_name: str,
    profile_core: pd.DataFrame,
    selected_index_all: pd.DataFrame,
    block_size_lookup: pd.DataFrame,
    profile_to_entity: dict[str, str],
    value_maps: dict[str, dict[str, set[str]]],
    model,
    feature_cols: list[str],
    cache_key: str = "",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Построить scored candidate pairs для одного split.

    Это центральная функция скрипта.

    Что она делает:
    1. Берёт только профили нужного split.
    2. Берёт `blocking_index` только для этих профилей.
    3. Генерирует все пары внутри каждого блока.
    4. Агрегирует повторные срабатывания одной пары в evidence-признаки.
    5. Для пар, прошедших quality gate, считает baseline pair score.
    6. Добавляет similarity-признаки и graph-rank признаки.
    7. Сохраняет результат в parquet cache.

    На выходе одна строка = одна candidate pair.
    """
    cache_suffix = f"_{cache_key}" if cache_key else ""
    cache_path = OUT_DIR / f"{split_name}_pair_evidence_scored{cache_suffix}.parquet"
    keep_cols = [
        "profile_id_l",
        "profile_id_r",
        "pair_key",
        "label",
        "rules_key",
        "score",
        *TREE_FEATURES,
    ]
    if cache_path.exists() and not force_rebuild:
        cached = pd.read_parquet(cache_path)
        missing_cols = [col for col in keep_cols if col not in cached.columns]
        if not missing_cols:
            log(f"{split_name}: loaded pair evidence cache rows={len(cached):,}")
            return cached[keep_cols].copy()
        log(f"{split_name}: cache is missing columns, rebuilding pair evidence")

    split_profiles = set(profile_core.loc[profile_core["split"].eq(split_name), "profile_id"].astype(str))
    split_index = selected_index_all[selected_index_all["profile_id"].isin(split_profiles)].copy()
    split_index = split_index.merge(
        block_size_lookup,
        on=["block_family", "block_rule", "block_value"],
        how="left",
    )

    log(
        f"{split_name}: building pair events "
        f"profiles={len(split_profiles):,} index_rows={len(split_index):,}"
    )
    pair_events = all_pairs_from_blocks(split_index)
    pair_events["pair_key"] = pair_events["profile_id_l"] + "|" + pair_events["profile_id_r"]
    log(f"{split_name}: pair events rows={len(pair_events):,}")

    # label нужен только для обучения/оценки: 1 если оба profile_id имеют один entity_id.
    # В production такого поля не будет.
    pair_events["label"] = pair_events["profile_id_l"].map(profile_to_entity).eq(
        pair_events["profile_id_r"].map(profile_to_entity)
    ).astype("int8")

    log(f"{split_name}: building pair evidence")
    # Здесь схлопываем несколько строк одной пары в одну строку.
    # Если пара нашлась через 5 разных block_rule/block_value, это превращается
    # в признаки вроде n_block_rules=5, min_block_size, sum_block_weight и т.д.
    pair_evidence = (
        pair_events.groupby(["profile_id_l", "profile_id_r", "pair_key", "label"], sort=False)
        .agg(
            # Сколько разных правил нашли пару. Чем больше независимых правил,
            # тем сильнее evidence.
            n_block_rules=("block_rule", "nunique"),

            # Сколько разных семейств нашли пару. Полезно отличать ситуацию
            # "три похожих context-правила" от "context + behavior + identity".
            n_block_families=("block_family", "nunique"),

            # Размеры блоков, через которые пара была найдена.
            # Маленький блок обычно точнее большого: если в блоке 2 профиля,
            # совпадение гораздо сильнее, чем в блоке на 900 профилей.
            min_block_size=("block_size", "min"),

            # block_weight = 1 / log1p(block_size), считается ниже в main().
            # Чем меньше блок, тем больше вес. sum/max/mean показывают силу
            # сработавших blocking-сигналов.
            sum_block_weight=("block_weight", "sum"),

            # Флаг технического fallback. Fallback нужен для покрытия, но это
            # слабый сигнал для склейки.
            hit_coverage_fallback=("block_family", lambda s: int((s == "coverage_fallback").any())),
            rules=("block_rule", lambda s: frozenset(s)),
            families=("block_family", lambda s: frozenset(s)),
        )
        .reset_index()
    )
    log(f"{split_name}: unique candidate pairs={len(pair_evidence):,}")
    pair_evidence["is_fallback_only"] = (
        pair_evidence["hit_coverage_fallback"].eq(1) & pair_evidence["n_block_families"].eq(1)
    ).astype("int8")
    pair_evidence["has_non_fallback_signal"] = (
        pair_evidence["n_block_families"].gt(pair_evidence["hit_coverage_fallback"])
    ).astype("int8")

    for col, default_value in {
        "score": 0.0,
        "fs_total_jaccard": 0.0,
        "geo_total_jaccard": 0.0,
        "fs_shared_count": 0,
        "identity_email_match": 0,
        "identity_phone_match": 0,
        "identity_strong_match": 0,
    }.items():
        pair_evidence[col] = default_value

    log(f"{split_name}: scoring gated pairs")
    # Quality gate: baseline pair model скорим только для пар, которые нашли
    # минимум два blocking-правила. Это экономит ресурсы и снижает шум.
    # Негейтнутые пары остаются в таблице со score=0, чтобы дерево видело их
    # как слабые кандидаты.
    gate = pair_evidence["n_block_rules"].ge(PAIR_QUALITY_GATE_MIN_RULES)
    gated = pair_evidence.loc[gate].copy()
    total_chunks = int(np.ceil(len(gated) / FULL_PAIR_SCORE_CHUNK_SIZE)) if len(gated) else 0
    log(
        f"{split_name}: gated pairs={len(gated):,} "
        f"chunks={total_chunks:,} chunk_size={FULL_PAIR_SCORE_CHUNK_SIZE:,}"
    )
    scores = []
    for chunk_no, start in enumerate(range(0, len(gated), FULL_PAIR_SCORE_CHUNK_SIZE), start=1):
        log(f"{split_name}: scoring chunk {chunk_no}/{total_chunks}")
        chunk = gated.iloc[start : start + FULL_PAIR_SCORE_CHUNK_SIZE]
        features = build_pair_features(chunk, value_maps)
        for col in feature_cols:
            if col not in features.columns:
                features[col] = 0
        scores.append(model.predict_proba(features[feature_cols])[:, 1])

        # Эти признаки считаются в build_pair_features из value_maps.
        # Они описывают уже не blocking evidence, а содержательное сходство
        # значений двух профилей.
        fs_intersect_cols = [col for col in features.columns if col.startswith("fs__") and col.endswith("__intersect_count")]
        pair_evidence.loc[chunk.index, "fs_total_jaccard"] = features["fs_total_jaccard"].to_numpy()
        pair_evidence.loc[chunk.index, "geo_total_jaccard"] = features["geo_total_jaccard"].to_numpy()
        pair_evidence.loc[chunk.index, "fs_shared_count"] = features[fs_intersect_cols].sum(axis=1).to_numpy()
        pair_evidence.loc[chunk.index, "identity_email_match"] = features["identity__email__intersect_count"].gt(0).astype("int8").to_numpy()
        pair_evidence.loc[chunk.index, "identity_phone_match"] = features["identity__phone__intersect_count"].gt(0).astype("int8").to_numpy()
        pair_evidence.loc[chunk.index, "identity_strong_match"] = (
            features["identity__email__intersect_count"].gt(0) | features["identity__phone__intersect_count"].gt(0)
        ).astype("int8").to_numpy()

    if scores:
        pair_evidence.loc[gate, "score"] = np.concatenate(scores)

    pair_evidence["has_strong_family"] = pair_evidence["families"].map(lambda x: int(bool(set(x) & STRONG_FAMILIES)))
    pair_evidence["has_behavior"] = pair_evidence["families"].map(lambda x: int("behavior" in x))
    pair_evidence["has_behavior_context"] = pair_evidence["families"].map(
        lambda x: int(bool(set(x) & {"behavior_context", "behavior_context_device"}))
    )
    pair_evidence["has_context"] = pair_evidence["families"].map(lambda x: int("context" in x))
    pair_evidence["has_coverage_compound"] = pair_evidence["families"].map(lambda x: int("coverage_compound" in x))
    pair_evidence["only_weak_families"] = pair_evidence["families"].map(lambda x: int(set(x).issubset(WEAK_FAMILIES)))
    pair_evidence["rules_key"] = pair_evidence["rules"].map(lambda x: "|".join(sorted(x)))

    out = pair_evidence[keep_cols].copy()
    out.to_parquet(cache_path, index=False)
    log(f"{split_name}: saved pair evidence cache rows={len(out):,}")
    return out


def main() -> None:
    """Запустить полный эксперимент tree edge policy."""
    log("start graph tree edge policy experiment")
    pd.DataFrame(
        [
            {"feature": feature, "description": TREE_FEATURE_DESCRIPTIONS.get(feature, "")}
            for feature in TREE_FEATURES
        ]
    ).to_csv(OUT_DIR / "tree_feature_descriptions.csv", index=False)

    # Артефакты из 03 ноутбука:
    # - profile_core: список профилей и ground truth entity_id;
    # - profile_value_summary_long: значения признаков для similarity;
    # - blocking_index: первый слой candidate generation;
    # - recommended_blocking_rules: правила, отобранные после positive_recall диагностики.
    profile_core = pd.read_parquet(MART_DIR / "profile_core.parquet")
    profile_values = pd.read_parquet(MART_DIR / "profile_value_summary_long.parquet")
    blocking_index = pd.read_parquet(MART_DIR / "blocking_index.parquet")
    recommended_rules = pd.read_csv(MART_DIR / "recommended_blocking_rules.csv")
    log(
        "loaded inputs "
        f"profiles={len(profile_core):,} values={len(profile_values):,} "
        f"blocking_rows={len(blocking_index):,} rules={len(recommended_rules):,}"
    )

    for df in [profile_core, profile_values, blocking_index]:
        for col in ["profile_id", "entity_id", "block_rule", "block_family", "block_value"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # Split строим на уровне entity_id, чтобы профили одного клиента не протекали
    # одновременно в valid и test.
    profile_core["split"] = profile_core["entity_id"].map(stable_entity_split)
    profile_to_entity = dict(zip(profile_core["profile_id"], profile_core["entity_id"]))

    # Baseline pair-модель из 04 ноутбука.
    # В этой версии её score НЕ входит в признаки дерева. Он остаётся только
    # для baseline-сравнения старого графового подхода и guardrail-вариантов.
    with open(PAIR_MODEL_DIR / "baseline_assignment_model.pkl", "rb") as f:
        model_bundle = pickle.load(f)
    pair_model = model_bundle["model"]
    pair_feature_cols = model_bundle["feature_cols"]

    recommended_rule_names = set(
        recommended_rules.loc[
            recommended_rules["recommended_for_next_step"].astype(str).str.lower().eq("true"),
            "block_rule",
        ].astype(str)
    )
    log(f"recommended rules selected={len(recommended_rule_names):,}")
    rules_cache_key = hashlib.md5("|".join(sorted(recommended_rule_names)).encode("utf-8")).hexdigest()[:8]
    log(f"pair evidence cache key={rules_cache_key}")

    # Оставляем только рекомендованные blocking-правила. Это и есть candidate
    # universe текущего эксперимента: дерево не ищет пары вне этих блоков.
    selected_index_all = blocking_index[blocking_index["block_rule"].isin(recommended_rule_names)].copy()
    log(f"selected blocking index rows={len(selected_index_all):,}")

    # Размер блока нужен как признак качества evidence.
    # Вес блока меньше для больших блоков: совпадение в блоке на 2 профиля
    # сильнее, чем совпадение в блоке на 900 профилей.
    block_size_lookup = (
        selected_index_all.groupby(["block_family", "block_rule", "block_value"], observed=True)["profile_id"]
        .nunique()
        .rename("block_size")
        .reset_index()
    )
    block_size_lookup["block_weight"] = 1 / np.log1p(block_size_lookup["block_size"])

    value_maps = make_value_maps(profile_values)
    log(f"value maps ready features={len(value_maps):,}")

    valid_core = profile_core[profile_core["split"].eq("valid")].copy()
    test_core = profile_core[profile_core["split"].eq("test")].copy()
    valid_profiles = set(valid_core["profile_id"].astype(str))
    test_profiles = set(test_core["profile_id"].astype(str))
    log(f"splits valid_profiles={len(valid_profiles):,} test_profiles={len(test_profiles):,}")

    valid_pairs = score_split_candidate_pairs(
        "valid",
        profile_core,
        selected_index_all,
        block_size_lookup,
        profile_to_entity,
        value_maps,
        pair_model,
        pair_feature_cols,
        cache_key=rules_cache_key,
    )
    test_pairs = score_split_candidate_pairs(
        "test",
        profile_core,
        selected_index_all,
        block_size_lookup,
        profile_to_entity,
        value_maps,
        pair_model,
        pair_feature_cols,
        cache_key=rules_cache_key,
    )

    rows = []
    tree_rows = []
    tree_rules = []
    feature_importance_rows = []
    log(f"candidate pairs valid={len(valid_pairs):,} test={len(test_pairs):,}")

    valid_guardrail = valid_pairs[valid_pairs["only_weak_families"].eq(0)].copy()
    test_guardrail = test_pairs[test_pairs["only_weak_families"].eq(0)].copy()
    valid_small_block = valid_pairs[
        valid_pairs["only_weak_families"].eq(0) | valid_pairs["min_block_size"].le(2)
    ].copy()
    test_small_block = test_pairs[
        test_pairs["only_weak_families"].eq(0) | test_pairs["min_block_size"].le(2)
    ].copy()

    log("evaluating score baselines")
    for split_name, pairs, core, profiles in [
        ("valid", valid_pairs, valid_core, valid_profiles),
        ("test", test_pairs, test_core, test_profiles),
    ]:
        row = evaluate_edge_policy(pairs, core, profiles, profile_to_entity, "score", 0.8, "pair_score_baseline_080")
        row["split"] = split_name
        rows.append(row)

    log("evaluating guardrail baselines")
    for split_name, pairs, core, profiles, policy in [
        ("valid", valid_guardrail, valid_core, valid_profiles, "guardrail_only_weak_blocked"),
        ("test", test_guardrail, test_core, test_profiles, "guardrail_only_weak_blocked"),
        ("valid", valid_small_block, valid_core, valid_profiles, "guardrail_small_block2"),
        ("test", test_small_block, test_core, test_profiles, "guardrail_small_block2"),
    ]:
        row = evaluate_edge_policy(pairs, core, profiles, profile_to_entity, "score", 0.8, policy)
        row["split"] = split_name
        rows.append(row)

    x_valid = valid_pairs[TREE_FEATURES].fillna(0)
    y_valid = valid_pairs["label"].astype(int)
    x_test = test_pairs[TREE_FEATURES].fillna(0)

    threshold_grid = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    total_plain_policies = len([2, 3, 4, 5, 6]) * len([50, 200, 1000])
    policy_no = 0

    for max_depth in [2, 3, 4, 5, 6]:
        for min_leaf in [50, 200, 1000]:
            policy_no += 1
            policy_name = f"tree_depth{max_depth}_leaf{min_leaf}"
            log(f"plain tree policy {policy_no}/{total_plain_policies}: {policy_name}")
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                class_weight="balanced",
                random_state=42,
            )
            tree.fit(x_valid, y_valid)
            valid_pairs[f"{policy_name}_proba"] = tree.predict_proba(x_valid)[:, 1]
            test_pairs[f"{policy_name}_proba"] = tree.predict_proba(x_test)[:, 1]

            valid_policy_rows = []
            for threshold in threshold_grid:
                row = evaluate_edge_policy(
                    valid_pairs,
                    valid_core,
                    valid_profiles,
                    profile_to_entity,
                    f"{policy_name}_proba",
                    float(threshold),
                    policy_name,
                )
                row["split"] = "valid"
                row["max_depth"] = max_depth
                row["min_samples_leaf"] = min_leaf
                valid_policy_rows.append(row)

            valid_report = pd.DataFrame(valid_policy_rows)
            selected = valid_report.sort_values(
                ["correct_client_decisions_pct", "wrong_merge_groups_pct", "existing_clients_found_pct"],
                ascending=[False, True, False],
            ).iloc[0]

            test_row = evaluate_edge_policy(
                test_pairs,
                test_core,
                test_profiles,
                profile_to_entity,
                f"{policy_name}_proba",
                float(selected["threshold"]),
                policy_name,
            )
            test_row["split"] = "test"
            test_row["max_depth"] = max_depth
            test_row["min_samples_leaf"] = min_leaf
            test_row["selected_on_valid_threshold"] = float(selected["threshold"])

            rows.extend(valid_policy_rows)
            rows.append(test_row)
            tree_rows.append(
                {
                    "policy": policy_name,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "selected_threshold": float(selected["threshold"]),
                    "valid_correct_client_decisions_pct": float(selected["correct_client_decisions_pct"]),
                    "valid_existing_clients_found_pct": float(selected["existing_clients_found_pct"]),
                    "valid_wrong_merge_groups_pct": float(selected["wrong_merge_groups_pct"]),
                    "test_clients_total": int(test_row["clients_total"]),
                    "test_existing_clients_total": int(test_row["existing_clients_total"]),
                    "test_existing_clients_found": int(test_row["existing_clients_found"]),
                    "test_existing_clients_found_pct": float(test_row["existing_clients_found_pct"]),
                    "test_existing_clients_missed": int(test_row["existing_clients_missed"]),
                    "test_existing_clients_missed_pct": float(test_row["existing_clients_missed_pct"]),
                    "test_new_clients_total": int(test_row["new_clients_total"]),
                    "test_new_clients_correct": int(test_row["new_clients_correct"]),
                    "test_new_clients_correct_pct": float(test_row["new_clients_correct_pct"]),
                    "test_new_clients_wrongly_attached": int(test_row["new_clients_wrongly_attached"]),
                    "test_new_clients_wrongly_attached_pct": float(test_row["new_clients_wrongly_attached_pct"]),
                    "test_predicted_merge_groups": int(test_row["predicted_merge_groups"]),
                    "test_wrong_merge_groups": int(test_row["wrong_merge_groups"]),
                    "test_wrong_merge_groups_pct": float(test_row["wrong_merge_groups_pct"]),
                    "test_graph_edges": int(test_row["graph_edges"]),
                    "test_correct_client_decisions_pct": float(test_row["correct_client_decisions_pct"]),
                }
            )
            tree_rules.append(
                "\n".join(
                    [
                        "=" * 100,
                        policy_name,
                        f"selected_threshold={float(selected['threshold'])}",
                        export_text(tree, feature_names=TREE_FEATURES, max_depth=4),
                    ]
                )
            )
            feature_importance_rows.extend(
                {"policy": policy_name, "feature": feature, "importance": float(importance)}
                for feature, importance in zip(TREE_FEATURES, tree.feature_importances_)
            )
            log(
                f"{policy_name}: threshold={float(selected['threshold'])} "
                f"found_existing={int(test_row['existing_clients_found'])}/{int(test_row['existing_clients_total'])} "
                f"new_correct={int(test_row['new_clients_correct'])}/{int(test_row['new_clients_total'])} "
                f"wrong_merges={int(test_row['wrong_merge_groups'])}/{int(test_row['predicted_merge_groups'])}"
            )

    constrained_datasets = [
        ("guardrail", valid_guardrail.copy(), test_guardrail.copy()),
        ("guardrail_smallblock2", valid_small_block.copy(), test_small_block.copy()),
    ]

    total_constrained_policies = len(constrained_datasets) * len([2, 3, 4, 5, 6]) * len([50, 200, 1000])
    constrained_policy_no = 0
    for dataset_name, train_pairs, eval_pairs in constrained_datasets:
        log(
            f"constrained dataset={dataset_name} "
            f"train_pairs={len(train_pairs):,} eval_pairs={len(eval_pairs):,}"
        )
        x_train = train_pairs[TREE_FEATURES].fillna(0)
        y_train = train_pairs["label"].astype(int)
        x_eval = eval_pairs[TREE_FEATURES].fillna(0)

        for max_depth in [2, 3, 4, 5, 6]:
            for min_leaf in [50, 200, 1000]:
                constrained_policy_no += 1
                policy_name = f"tree_{dataset_name}_depth{max_depth}_leaf{min_leaf}"
                log(f"constrained tree policy {constrained_policy_no}/{total_constrained_policies}: {policy_name}")
                tree = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    class_weight="balanced",
                    random_state=42,
                )
                tree.fit(x_train, y_train)
                train_pairs[f"{policy_name}_proba"] = tree.predict_proba(x_train)[:, 1]
                eval_pairs[f"{policy_name}_proba"] = tree.predict_proba(x_eval)[:, 1]

                valid_policy_rows = []
                for threshold in threshold_grid:
                    row = evaluate_edge_policy(
                        train_pairs,
                        valid_core,
                        valid_profiles,
                        profile_to_entity,
                        f"{policy_name}_proba",
                        float(threshold),
                        policy_name,
                    )
                    row["split"] = "valid"
                    row["max_depth"] = max_depth
                    row["min_samples_leaf"] = min_leaf
                    row["constraint_dataset"] = dataset_name
                    valid_policy_rows.append(row)

                valid_report = pd.DataFrame(valid_policy_rows)
                selected = valid_report.sort_values(
                    ["correct_client_decisions_pct", "wrong_merge_groups_pct", "existing_clients_found_pct"],
                    ascending=[False, True, False],
                ).iloc[0]

                test_row = evaluate_edge_policy(
                    eval_pairs,
                    test_core,
                    test_profiles,
                    profile_to_entity,
                    f"{policy_name}_proba",
                    float(selected["threshold"]),
                    policy_name,
                )
                test_row["split"] = "test"
                test_row["max_depth"] = max_depth
                test_row["min_samples_leaf"] = min_leaf
                test_row["constraint_dataset"] = dataset_name
                test_row["selected_on_valid_threshold"] = float(selected["threshold"])

                rows.extend(valid_policy_rows)
                rows.append(test_row)
                tree_rows.append(
                    {
                        "policy": policy_name,
                        "constraint_dataset": dataset_name,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_leaf,
                        "selected_threshold": float(selected["threshold"]),
                        "valid_correct_client_decisions_pct": float(selected["correct_client_decisions_pct"]),
                        "valid_existing_clients_found_pct": float(selected["existing_clients_found_pct"]),
                        "valid_wrong_merge_groups_pct": float(selected["wrong_merge_groups_pct"]),
                        "test_clients_total": int(test_row["clients_total"]),
                        "test_existing_clients_total": int(test_row["existing_clients_total"]),
                        "test_existing_clients_found": int(test_row["existing_clients_found"]),
                        "test_existing_clients_found_pct": float(test_row["existing_clients_found_pct"]),
                        "test_existing_clients_missed": int(test_row["existing_clients_missed"]),
                        "test_existing_clients_missed_pct": float(test_row["existing_clients_missed_pct"]),
                        "test_new_clients_total": int(test_row["new_clients_total"]),
                        "test_new_clients_correct": int(test_row["new_clients_correct"]),
                        "test_new_clients_correct_pct": float(test_row["new_clients_correct_pct"]),
                        "test_new_clients_wrongly_attached": int(test_row["new_clients_wrongly_attached"]),
                        "test_new_clients_wrongly_attached_pct": float(test_row["new_clients_wrongly_attached_pct"]),
                        "test_predicted_merge_groups": int(test_row["predicted_merge_groups"]),
                        "test_wrong_merge_groups": int(test_row["wrong_merge_groups"]),
                        "test_wrong_merge_groups_pct": float(test_row["wrong_merge_groups_pct"]),
                        "test_graph_edges": int(test_row["graph_edges"]),
                        "test_correct_client_decisions_pct": float(test_row["correct_client_decisions_pct"]),
                    }
                )
                tree_rules.append(
                    "\n".join(
                        [
                            "=" * 100,
                            policy_name,
                            f"selected_threshold={float(selected['threshold'])}",
                            export_text(tree, feature_names=TREE_FEATURES, max_depth=4),
                        ]
                    )
                )
                feature_importance_rows.extend(
                    {"policy": policy_name, "feature": feature, "importance": float(importance)}
                    for feature, importance in zip(TREE_FEATURES, tree.feature_importances_)
                )
                log(
                    f"{policy_name}: threshold={float(selected['threshold'])} "
                    f"found_existing={int(test_row['existing_clients_found'])}/{int(test_row['existing_clients_total'])} "
                    f"new_correct={int(test_row['new_clients_correct'])}/{int(test_row['new_clients_total'])} "
                    f"wrong_merges={int(test_row['wrong_merge_groups'])}/{int(test_row['predicted_merge_groups'])}"
                )

    full_report = pd.DataFrame(rows)
    full_report.to_csv(OUT_DIR / "tree_policy_full_threshold_report.csv", index=False)

    selected_report = pd.DataFrame(tree_rows).sort_values(
        ["test_correct_client_decisions_pct", "test_wrong_merge_groups_pct"],
        ascending=[False, True],
    )
    selected_report.to_csv(OUT_DIR / "tree_policy_selected_report.csv", index=False)
    feature_importance_report = pd.DataFrame(feature_importance_rows)
    feature_importance_report.to_csv(OUT_DIR / "tree_policy_feature_importance.csv", index=False)

    with open(OUT_DIR / "tree_policy_rules.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(tree_rules))

    log(f"saved outputs to {OUT_DIR}")
    log("top selected policies:")
    print(selected_report.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
