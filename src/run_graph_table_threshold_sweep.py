from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SRC_DIR / "graph_table_test_data" / "raw_packet_test_1000_profiles.parquet"
DEFAULT_ARTIFACT_DIR = SRC_DIR / "graph_table_artifacts"
DEFAULT_OUT_DIR = ROOT_DIR / "reports" / "model_eval"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graph_table_pipeline import (
    apply_mutual_top_k,
    build_assignment_table,
    build_packet_profile_values_from_flat,
    build_profile_core_from_flat,
    load_artifacts,
    score_packet,
)
from graph_table_utils import prepare_graph_table_input_df
from run_graph_table_inference import evaluate_assignment_if_labels_available


def calculate_final_pr_auc(result: pd.DataFrame) -> float:
    """Считает PR-AUC для итоговых объединений после полного алгоритма."""
    recall_col = "final_merge_recall"
    precision_col = "final_merge_precision"
    curve = (
        result[[recall_col, precision_col]]
        .dropna()
        .sort_values([recall_col, precision_col])
    )
    if curve.empty:
        return float("nan")

    # Точка "ничего не объединили": полнота равна нулю, точность условно равна единице.
    full_curve = pd.concat(
        [
            pd.DataFrame([{recall_col: 0.0, precision_col: 1.0}]),
            curve,
        ],
        ignore_index=True,
    ).drop_duplicates(subset=[recall_col], keep="last")
    full_curve = full_curve.sort_values(recall_col)

    recalls = full_curve[recall_col].to_numpy()
    precisions = full_curve[precision_col].to_numpy()
    pr_auc = 0.0
    for idx in range(1, len(recalls)):
        pr_auc += max(recalls[idx] - recalls[idx - 1], 0.0) * precisions[idx]

    return float(pr_auc)


def save_final_pr_curves(result: pd.DataFrame, png_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recall_col = "final_merge_recall"
    precision_col = "final_merge_precision"
    plt.figure(figsize=(7, 5))
    for top_k, rows in result.groupby("graph_top_k", sort=True):
        curve = rows[["threshold", recall_col, precision_col]].dropna().sort_values(recall_col)
        plt.plot(curve[recall_col], curve[precision_col], marker="o", linewidth=2, label=f"top-K={top_k}")
    plt.xlabel("Полнота найденных объединений")
    plt.ylabel("Точность объединений")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("run_graph_table_threshold_sweep")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_top_k(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item <= 0 for item in values):
        raise ValueError("--graph-top-k должен содержать положительные целые значения")
    return values


def component_size_stats(assignment: pd.DataFrame) -> dict[str, int]:
    """Считает число уникальных групп профилей разных размеров."""
    components = assignment[["component_profile_ids", "n_component_profiles"]].drop_duplicates()
    sizes = components["n_component_profiles"]
    return {
        "merge_groups_total": int(sizes.ge(2).sum()),
        "merge_groups_size_2": int(sizes.eq(2).sum()),
        "merge_groups_size_3_plus": int(sizes.ge(3).sum()),
        "max_group_size": int(sizes.max()) if not sizes.empty else 0,
    }


def extend_confident_pairs_by_two_links(
    base_edges: pd.DataFrame,
    pair_scores: pd.DataFrame,
    extension_threshold: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Добавляет профиль к уверенной паре только при двух сильных связях с ней."""
    if base_edges.empty:
        return base_edges.copy(), {"extension_profiles_added": 0, "extension_edges_added": 0}

    base_graph = nx.Graph()
    base_graph.add_edges_from(base_edges[["profile_id_l", "profile_id_r"]].itertuples(index=False, name=None))
    confident_pairs = [
        set(map(str, component))
        for component in nx.connected_components(base_graph)
        if len(component) == 2
    ]
    already_linked = set(map(str, base_graph.nodes))
    strong_pairs = pair_scores[pair_scores["score"].ge(extension_threshold)].copy()
    group_by_member = {
        member: group_no
        for group_no, group in enumerate(confident_pairs)
        for member in group
    }

    # Для каждого свободного профиля сохраняем группы, с которыми у него есть
    # по одной сильной связи к обоим участникам уже принятой пары.
    links: dict[tuple[str, int], dict[str, object]] = {}
    for edge in strong_pairs.itertuples(index=False):
        left = str(edge.profile_id_l)
        right = str(edge.profile_id_r)
        for member, outsider in [(left, right), (right, left)]:
            group_no = group_by_member.get(member)
            if group_no is None or outsider in already_linked:
                continue
            key = (outsider, group_no)
            current = links.setdefault(key, {})
            if member not in current or float(edge.score) > float(current[member].score):
                current[member] = edge

    offers: dict[str, list[pd.DataFrame]] = {}
    for (outsider, _), member_edges in links.items():
        if len(member_edges) == 2:
            offers.setdefault(outsider, []).append(
                pd.DataFrame([edge._asdict() for edge in member_edges.values()])
            )

    # Если профиль одновременно подходит к нескольким группам, автоматическое
    # объединение опасно: оставляем его без нового ребра.
    additions = [choices[0] for choices in offers.values() if len(choices) == 1]
    if not additions:
        return base_edges.copy(), {"extension_profiles_added": 0, "extension_edges_added": 0}
    extended = pd.concat([base_edges, *additions], ignore_index=True).drop_duplicates(
        ["profile_id_l", "profile_id_r"]
    )
    return extended, {
        "extension_profiles_added": int(len(additions)),
        "extension_edges_added": int(len(extended) - len(base_edges)),
    }


def build_quality_row(
    strategy: str,
    top_k: int,
    threshold: float,
    extension_threshold: float | None,
    edges: pd.DataFrame,
    packet_core: pd.DataFrame,
    pair_scores: pd.DataFrame,
    raw_df: pd.DataFrame,
    artifacts: dict,
    artifact_dir: Path,
    extension_stats: dict[str, int] | None = None,
) -> dict[str, object]:
    """Строит итоговый граф и собирает метрики одного проверяемого режима."""
    assignment = build_assignment_table(packet_core, artifacts["historical_core"], edges, pair_scores)
    evaluation = evaluate_assignment_if_labels_available(assignment, raw_df, artifact_dir)
    if evaluation is None:
        raise ValueError("Для расчёта метрик на тесте входной пакет должен содержать entity_id.")
    entity = evaluation["entity_level"]
    profile = evaluation["profile_level"]
    final_graph_quality = evaluation.get("final_graph_quality", {})
    linked_pair_quality = final_graph_quality.get("pair_level", {})
    final_merge_quality = final_graph_quality.get("final_merge_level", {})
    return {
        "strategy": strategy,
        "graph_top_k": top_k,
        "threshold": threshold,
        "extension_threshold": extension_threshold,
        "candidate_pairs": int(len(pair_scores)),
        "accepted_edges": int(len(edges)),
        **(extension_stats or {"extension_profiles_added": 0, "extension_edges_added": 0}),
        **component_size_stats(assignment),
        "existing_entities_found": entity.get("existing_entities_found"),
        "existing_entities_total": entity.get("existing_entities_total"),
        "existing_entities_found_pct": entity.get("existing_entities_found_pct"),
        "new_entities_correct": entity.get("new_entities_correct"),
        "new_entities_total": entity.get("new_entities_total"),
        "new_entities_correct_pct": entity.get("new_entities_correct_pct"),
        "new_entities_wrongly_attached": entity.get("new_entities_wrongly_attached"),
        "real_existing_profiles_found_pct": profile.get("real_existing_profiles_found_pct"),
        "real_new_profiles_correct_pct": profile.get("real_new_profiles_correct_pct"),
        "real_new_profiles_wrongly_attached": profile.get("real_new_profiles_wrongly_attached"),
        "wrong_existing_attachment_profiles": profile.get("wrong_existing_attachment_profiles"),
        "linked_profile_pairs_tp": linked_pair_quality.get("tp"),
        "linked_profile_pairs_fp": linked_pair_quality.get("fp"),
        "linked_profile_pairs_fn": linked_pair_quality.get("fn"),
        "linked_profile_pairs_recall": linked_pair_quality.get("recall"),
        "final_merge_tp": final_merge_quality.get("tp"),
        "final_merge_fp": final_merge_quality.get("fp"),
        "final_merge_fn": final_merge_quality.get("fn"),
        "final_merge_precision": final_merge_quality.get("precision"),
        "final_merge_recall": final_merge_quality.get("recall"),
        "final_merge_f1": final_merge_quality.get("f1"),
    }


def make_threshold_grid(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--threshold-step должен быть больше 0")
    if stop < start:
        raise ValueError("--threshold-stop должен быть не меньше --threshold-start")
    values = []
    current = start
    while current <= stop + step / 10:
        values.append(round(current, 6))
        current += step
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить качество итоговых объединений при разных порогах модели.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--pair-score-cache",
        type=Path,
        default=None,
        help="Необязательный файл уже посчитанных оценок пар для повторного запуска на том же пакете и тех же артефактах.",
    )
    parser.add_argument("--thresholds", type=str, default=None, help="Список порогов через запятую; заменяет диапазон параметров ниже.")
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-stop", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--graph-top-k",
        type=str,
        default=None,
        help="Список значений top-K через запятую. По умолчанию используется значение из policy_config.json.",
    )
    parser.add_argument(
        "--two-link-extension-thresholds",
        type=str,
        default=None,
        help="Пороги строгого добавления третьего профиля: нужны две сильные связи с уже принятой парой.",
    )
    args = parser.parse_args()

    logger = setup_logger()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds) if args.thresholds else make_threshold_grid(
        args.threshold_start,
        args.threshold_stop,
        args.threshold_step,
    )
    extension_thresholds = parse_thresholds(args.two_link_extension_thresholds) if args.two_link_extension_thresholds else []

    logger.info("читаем входной пакет=%s", args.input)
    raw_df = pd.read_parquet(args.input)
    work_df = prepare_graph_table_input_df(raw_df)
    packet_core = build_profile_core_from_flat(work_df)
    packet_values = build_packet_profile_values_from_flat(work_df)

    logger.info("загружаем артефакты=%s", args.artifact_dir)
    artifacts = load_artifacts(args.artifact_dir)
    runtime_top_k = int(artifacts["config"].get("graph_top_k", 1))
    top_k_values = parse_top_k(args.graph_top_k) if args.graph_top_k else [runtime_top_k]
    policy_name = artifacts["config"].get("policy_name", "graph_table_edge_policy")
    runtime_threshold = float(artifacts["config"].get("score_threshold", 0.95))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_policy = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(policy_name))
    csv_path = args.out_dir / f"final_quality_threshold_sweep_{safe_policy}_{timestamp}.csv"
    json_path = args.out_dir / f"final_quality_threshold_sweep_summary_{safe_policy}_{timestamp}.json"
    final_merge_pr_png_path = args.out_dir / f"final_merge_pr_curve_{safe_policy}_{timestamp}.png"
    generated_pair_score_path = args.out_dir / f"pair_scores_{args.input.stem}_{safe_policy}_{timestamp}.parquet"
    pair_score_cache = args.pair_score_cache

    if pair_score_cache is not None and pair_score_cache.exists():
        logger.info("загружаем готовые оценки пар=%s", pair_score_cache)
        pair_scores = pd.read_parquet(pair_score_cache)
    else:
        logger.info("оцениваем пары один раз; проверяемые пороги=%s", thresholds)
        _, pair_scores, _ = score_packet(packet_core, packet_values, artifacts, score_threshold=min(thresholds))
        logger.info("сохраняем рассчитанные оценки пар=%s", generated_pair_score_path)
        pair_scores.to_parquet(generated_pair_score_path, index=False)
    logger.info(
        "оценки пар готовы candidate_pairs=%s score_min=%.6f score_p95=%.6f score_max=%.6f",
        f"{len(pair_scores):,}",
        float(pair_scores["score"].min()),
        float(pair_scores["score"].quantile(0.95)),
        float(pair_scores["score"].max()),
    )

    rows = []
    for top_k in top_k_values:
        for threshold in thresholds:
            logger.info("проверяем graph_top_k=%s порог=%.4f", top_k, threshold)
            edges = apply_mutual_top_k(pair_scores, "score", threshold, top_k)
            rows.append(
                build_quality_row(
                    "mutual_top_k",
                    top_k,
                    threshold,
                    None,
                    edges,
                    packet_core,
                    pair_scores,
                    raw_df,
                    artifacts,
                    args.artifact_dir,
                )
            )
            for extension_threshold in extension_thresholds:
                logger.info(
                    "проверяем строгое расширение graph_top_k=%s порог=%.4f порог_добавления=%.4f",
                    top_k,
                    threshold,
                    extension_threshold,
                )
                extended_edges, extension_stats = extend_confident_pairs_by_two_links(
                    edges,
                    pair_scores,
                    extension_threshold,
                )
                rows.append(
                    build_quality_row(
                        "two_link_extension",
                        top_k,
                        threshold,
                        extension_threshold,
                        extended_edges,
                        packet_core,
                        pair_scores,
                        raw_df,
                        artifacts,
                        args.artifact_dir,
                        extension_stats,
                    )
                )
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = pd.DataFrame(rows)
    result.to_csv(csv_path, index=False)
    baseline_result = result[result["strategy"].eq("mutual_top_k")]
    pr_auc_by_top_k = {
        str(top_k): calculate_final_pr_auc(rows)
        for top_k, rows in baseline_result.groupby("graph_top_k", sort=True)
    }
    save_final_pr_curves(
        baseline_result,
        final_merge_pr_png_path,
        f"Качество итоговых объединений: {policy_name}",
    )

    def best_with_guardrail(min_new_correct_pct: float) -> list[dict]:
        candidates = result[result["new_entities_correct_pct"].ge(min_new_correct_pct)].sort_values(
            ["existing_entities_found_pct", "new_entities_correct_pct"],
            ascending=[False, False],
        )
        return candidates.head(1).to_dict(orient="records")

    runtime_point = result.loc[
        result["strategy"].eq("mutual_top_k")
        &
        result["graph_top_k"].eq(runtime_top_k)
        & result["threshold"].sub(runtime_threshold).abs().le(1e-12)
    ]
    summary = {
        "policy_name": policy_name,
        "artifact_dir": str(args.artifact_dir),
        "input": str(args.input),
        "candidate_pairs": int(len(pair_scores)),
        "pair_scores_path": str(pair_score_cache if pair_score_cache is not None and pair_score_cache.exists() else generated_pair_score_path),
        "thresholds": thresholds,
        "threshold_start": args.threshold_start,
        "threshold_stop": args.threshold_stop,
        "threshold_step": args.threshold_step,
        "runtime_threshold": runtime_threshold,
        "runtime_graph_top_k": runtime_top_k,
        "checked_graph_top_k": top_k_values,
        "checked_two_link_extension_thresholds": extension_thresholds,
        "runtime_threshold_metrics": runtime_point.to_dict(orient="records"),
        "pr_auc_scope": "итоговые объединения после порога модели, взаимного top-K и построения графа",
        "pr_auc_is_full_threshold_range": bool(min(thresholds) <= 0.0 and max(thresholds) >= 1.0),
        "best_with_new_entities_correct_ge_85": best_with_guardrail(85.0),
        "best_with_new_entities_correct_ge_90": best_with_guardrail(90.0),
        "best_with_new_entities_correct_ge_95": best_with_guardrail(95.0),
        "final_merge_pr_auc_by_graph_top_k": pr_auc_by_top_k,
        "final_merge_pr_curve_png": str(final_merge_pr_png_path),
        "csv_path": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== КАЧЕСТВО ИТОГОВЫХ ОБЪЕДИНЕНИЙ ПО ПОРОГАМ ===")
    print(
        result[
            [
                "threshold",
                "strategy",
                "graph_top_k",
                "extension_threshold",
                "accepted_edges",
                "extension_profiles_added",
                "merge_groups_size_2",
                "merge_groups_size_3_plus",
                "max_group_size",
                "linked_profile_pairs_tp",
                "linked_profile_pairs_fp",
                "linked_profile_pairs_recall",
                "final_merge_precision",
                "final_merge_recall",
                "final_merge_f1",
                "existing_entities_found_pct",
                "new_entities_correct_pct",
            ]
        ].to_string(index=False)
    )
    print(
        "\nPR-AUC итоговых объединений после полного алгоритма по graph_top_k: "
        f"{pr_auc_by_top_k}"
    )
    if min(thresholds) > 0.0 or max(thresholds) < 1.0:
        print("ВНИМАНИЕ: PR-AUC рассчитан только по указанному диапазону порогов; для финального отчёта используйте диапазон 0..1.")
    print(f"\ncsv={csv_path}")
    print(f"summary={json_path}")


if __name__ == "__main__":
    main()
