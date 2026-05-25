from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_PACKET = Path(__file__).resolve().parent / "graph_table_test_data" / "raw_packet_test_1000_profiles.parquet"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "graph_table_artifacts"
DEFAULT_OUT_DIR = ROOT_DIR / "reports" / "graph_table_inference"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_table_html_report import save_html_report


# Настраивает логирование inference в консоль и файл.
def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_graph_table_inference_{timestamp}.log"

    logger = logging.getLogger("run_graph_table_inference")
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


# Закрывает файловый лог сразу после расчёта, что особенно важно для долгоживущего Streamlit-процесса.
def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


# Считает F1 по precision и recall.
def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# Возвращает канонический ключ пары profile_id.
def _pair(left: str, right: str) -> tuple[str, str]:
    left = str(left)
    right = str(right)
    return (left, right) if left < right else (right, left)


# Разбирает список profile_id из строки компоненты графа.
def _component_ids(value: object, fallback_profile_id: str) -> list[str]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return [str(fallback_profile_id)]
    return sorted({part for part in str(value).split("|") if part})


# Считает качество пар и финальных объединений, если во входе есть entity_id.
def evaluate_final_graph_if_labels_available(
    assignment: pd.DataFrame,
    raw_df: pd.DataFrame,
    artifact_dir: Path,
) -> dict | None:
    """Считает качество финального графа по парам и финальным объединениям."""
    if "entity_id" not in raw_df.columns:
        return None
    hist_core_path = artifact_dir / "historical_profile_core.parquet"
    if not hist_core_path.exists():
        return None

    packet_profiles = raw_df[["profile_id", "entity_id"]].dropna().drop_duplicates().copy()
    packet_profiles["profile_id"] = packet_profiles["profile_id"].astype(str)
    packet_profiles["entity_id"] = packet_profiles["entity_id"].astype(str)
    packet_ids = set(packet_profiles["profile_id"])

    historical_core = pd.read_parquet(hist_core_path, columns=["profile_id", "entity_id"]).dropna().drop_duplicates()
    historical_core["profile_id"] = historical_core["profile_id"].astype(str)
    historical_core["entity_id"] = historical_core["entity_id"].astype(str)

    universe = pd.concat([historical_core, packet_profiles], ignore_index=True).drop_duplicates(["profile_id", "entity_id"])
    profile_to_entity = dict(zip(universe["profile_id"], universe["entity_id"]))

    true_pairs: set[tuple[str, str]] = set()
    true_entity_profiles: dict[str, set[str]] = {}
    for entity_id, grp in universe.groupby("entity_id", sort=False):
        profiles = sorted(grp["profile_id"].astype(str).unique())
        if not set(profiles) & packet_ids:
            continue
        true_entity_profiles[str(entity_id)] = set(profiles)
        if len(profiles) < 2:
            continue
        for left, right in combinations(profiles, 2):
            if left in packet_ids or right in packet_ids:
                true_pairs.add(_pair(left, right))

    predicted_pairs: set[tuple[str, str]] = set()
    predicted_final_merges: dict[str, set[str]] = {}
    for row in assignment[["profile_id", "component_profile_ids"]].itertuples(index=False):
        ids = set(_component_ids(row.component_profile_ids, row.profile_id))
        if not ids & packet_ids:
            continue
        key = "|".join(sorted(ids))
        predicted_final_merges[key] = ids
        if len(ids) < 2:
            continue
        for left, right in combinations(sorted(ids), 2):
            if left in packet_ids or right in packet_ids:
                predicted_pairs.add(_pair(left, right))

    pair_tp = len(predicted_pairs & true_pairs)
    pair_fp = len(predicted_pairs - true_pairs)
    pair_fn = len(true_pairs - predicted_pairs)
    pair_precision = pair_tp / max(pair_tp + pair_fp, 1)
    pair_recall = pair_tp / max(pair_tp + pair_fn, 1)

    final_merge_tp = 0
    final_merge_fp = 0
    recovered_entities: set[str] = set()
    for ids in predicted_final_merges.values():
        if len(ids) < 2:
            continue
        known_entities = {profile_to_entity[p] for p in ids if p in profile_to_entity}
        if len(known_entities) == 1:
            entity_id = next(iter(known_entities))
            same_entity_profiles = ids & true_entity_profiles.get(entity_id, set())
            if len(same_entity_profiles) >= 2:
                final_merge_tp += 1
                recovered_entities.add(entity_id)
            else:
                final_merge_fp += 1
        else:
            final_merge_fp += 1

    true_duplicate_entities = {
        entity_id
        for entity_id, profiles in true_entity_profiles.items()
        if len(profiles) >= 2 and bool(profiles & packet_ids)
    }
    final_merge_fn = len(true_duplicate_entities - recovered_entities)
    final_merge_precision = final_merge_tp / max(final_merge_tp + final_merge_fp, 1)
    final_merge_recall = final_merge_tp / max(final_merge_tp + final_merge_fn, 1)

    return {
        "pair_level": {
            "scope": "final_graph_pairs_with_packet_profile",
            "tp": pair_tp,
            "fp": pair_fp,
            "fn": pair_fn,
            "precision": pair_precision,
            "recall": pair_recall,
            "f1": _f1(pair_precision, pair_recall),
            "predicted_positive_pairs": len(predicted_pairs),
            "true_positive_pairs": len(true_pairs),
        },
        "final_merge_level": {
            "scope": "final_profile_merges_with_packet_profile",
            "tp": final_merge_tp,
            "fp": final_merge_fp,
            "fn": final_merge_fn,
            "precision": final_merge_precision,
            "recall": final_merge_recall,
            "f1": _f1(final_merge_precision, final_merge_recall),
            "predicted_final_merges": final_merge_tp + final_merge_fp,
            "true_duplicate_entities": len(true_duplicate_entities),
        },
    }


# Считает бизнес-метрики assignment по новым и найденным существующим клиентам.
def evaluate_assignment_if_labels_available(
    assignment: pd.DataFrame,
    raw_df: pd.DataFrame,
    artifact_dir: Path,
) -> dict | None:
    if "entity_id" not in raw_df.columns:
        return None
    hist_core_path = artifact_dir / "historical_profile_core.parquet"
    if not hist_core_path.exists():
        return None

    packet_profiles = raw_df[["profile_id", "entity_id"]].dropna().drop_duplicates().copy()
    packet_profiles["profile_id"] = packet_profiles["profile_id"].astype(str)
    packet_profiles["entity_id"] = packet_profiles["entity_id"].astype(str)
    packet_ids = set(packet_profiles["profile_id"])

    historical_core = pd.read_parquet(hist_core_path, columns=["profile_id", "entity_id"]).dropna().drop_duplicates()
    historical_core["profile_id"] = historical_core["profile_id"].astype(str)
    historical_core["entity_id"] = historical_core["entity_id"].astype(str)
    historical_external = historical_core[~historical_core["profile_id"].isin(packet_ids)]

    external_counts = historical_external.groupby("entity_id")["profile_id"].nunique().rename("external_profiles")
    packet_entity_counts = packet_profiles.groupby("entity_id")["profile_id"].nunique().rename("packet_profiles")
    entity_truth = pd.concat([packet_entity_counts, external_counts], axis=1).fillna(0).reset_index()
    entity_truth = entity_truth[entity_truth["packet_profiles"].gt(0)].copy()
    entity_truth["should_match_existing"] = entity_truth["external_profiles"].gt(0)
    entity_truth["should_be_new"] = ~entity_truth["should_match_existing"]

    scored = assignment.copy()
    scored["profile_id"] = scored["profile_id"].astype(str)
    scored = scored.merge(packet_profiles, on="profile_id", how="left")
    scored = scored.merge(entity_truth[["entity_id", "should_match_existing", "should_be_new"]], on="entity_id", how="left")
    scored["predicted_entity_id_str"] = scored["predicted_entity_id"].astype("string")
    scored["correct_existing_profile"] = (
        scored["decision"].eq("matched_existing")
        & scored["predicted_entity_id_str"].eq(scored["entity_id"].astype("string"))
    )
    scored["is_new_profile_decision"] = scored["decision"].eq("new_client")

    profile_existing_total = int(scored["should_match_existing"].sum())
    profile_existing_found = int((scored["should_match_existing"] & scored["correct_existing_profile"]).sum())
    profile_new_total = int(scored["should_be_new"].sum())
    profile_new_correct = int((scored["should_be_new"] & scored["is_new_profile_decision"]).sum())
    wrong_existing_attachment_profiles = int((scored["decision"].eq("matched_existing") & ~scored["correct_existing_profile"]).sum())

    entity_found = scored.groupby("entity_id")["correct_existing_profile"].any().rename("found_correct_existing").reset_index()
    entity_decision = entity_truth.merge(entity_found, on="entity_id", how="left").fillna({"found_correct_existing": False})
    existing_entities_total = int(entity_decision["should_match_existing"].sum())
    existing_entities_found = int((entity_decision["should_match_existing"] & entity_decision["found_correct_existing"]).sum())

    new_entity_profile_ok = scored.groupby("entity_id")["is_new_profile_decision"].all().rename("all_profiles_new").reset_index()
    entity_decision = entity_decision.merge(new_entity_profile_ok, on="entity_id", how="left").fillna({"all_profiles_new": False})
    new_entities_total = int(entity_decision["should_be_new"].sum())
    new_entities_correct = int((entity_decision["should_be_new"] & entity_decision["all_profiles_new"]).sum())
    final_graph_quality = evaluate_final_graph_if_labels_available(assignment, raw_df, artifact_dir)

    result = {
        "profile_level": {
            "profiles_total": int(len(scored)),
            "real_existing_profiles_total": profile_existing_total,
            "real_existing_profiles_found": profile_existing_found,
            "real_existing_profiles_found_pct": 100 * profile_existing_found / max(profile_existing_total, 1),
            "real_existing_profiles_missed": profile_existing_total - profile_existing_found,
            "real_existing_profiles_missed_pct": 100 * (profile_existing_total - profile_existing_found) / max(profile_existing_total, 1),
            "real_new_profiles_total": profile_new_total,
            "real_new_profiles_correct": profile_new_correct,
            "real_new_profiles_correct_pct": 100 * profile_new_correct / max(profile_new_total, 1),
            "real_new_profiles_wrongly_attached": profile_new_total - profile_new_correct,
            "wrong_existing_attachment_profiles": wrong_existing_attachment_profiles,
            "ambiguous_profiles": int(scored["decision"].eq("ambiguous").sum()),
        },
        "entity_level": {
            "entities_total": int(len(entity_decision)),
            "existing_entities_total": existing_entities_total,
            "existing_entities_found": existing_entities_found,
            "existing_entities_found_pct": 100 * existing_entities_found / max(existing_entities_total, 1),
            "existing_entities_missed": existing_entities_total - existing_entities_found,
            "existing_entities_missed_pct": 100 * (existing_entities_total - existing_entities_found) / max(existing_entities_total, 1),
            "new_entities_total": new_entities_total,
            "new_entities_correct": new_entities_correct,
            "new_entities_correct_pct": 100 * new_entities_correct / max(new_entities_total, 1),
            "new_entities_wrongly_attached": new_entities_total - new_entities_correct,
        },
        "decision_counts": scored["decision"].value_counts(dropna=False).to_dict(),
        "best_match_score": scored["best_match_score"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
    }
    if final_graph_quality is not None:
        result["final_graph_quality"] = final_graph_quality
    return result


# Печатает основные метрики inference в удобном текстовом виде.
def print_main_metrics(metrics: dict, evaluation: dict | None) -> None:
    print("\n=== MAIN RUNTIME METRICS ===")
    print(f"profiles: {metrics.get('profiles', 0):,}")
    print(f"candidate_pairs: {metrics.get('candidate_pairs', 0):,}")
    print(f"accepted_edges: {metrics.get('accepted_edges', 0):,}")
    print(f"matched_existing: {metrics.get('matched_existing', 0):,}")
    print(f"new_clients: {metrics.get('new_clients', 0):,}")
    print(f"ambiguous: {metrics.get('ambiguous', 0):,}")
    print(f"threshold: {metrics.get('threshold')}")
    print(f"model_mode: {metrics.get('model_mode')}")
    accepted_rules = metrics.get("rule_usage", {}).get("accepted_rules", [])
    if accepted_rules:
        print("\n=== TOP ACCEPTED BLOCKING RULES ===")
        for row in accepted_rules[:10]:
            print(f"{row['block_rule']}: {row['pair_count']:,} accepted pair hits")
    if not evaluation:
        return

    entity = evaluation["entity_level"]
    profile = evaluation["profile_level"]
    print("\n=== MAIN QUALITY METRICS BY ENTITY_ID ===")
    print(
        "existing clients found: "
        f"{entity['existing_entities_found']:,}/{entity['existing_entities_total']:,} "
        f"({entity['existing_entities_found_pct']:.2f}%)"
    )
    print(
        "existing clients missed: "
        f"{entity['existing_entities_missed']:,}/{entity['existing_entities_total']:,} "
        f"({entity['existing_entities_missed_pct']:.2f}%)"
    )
    print(
        "new clients correct: "
        f"{entity['new_entities_correct']:,}/{entity['new_entities_total']:,} "
        f"({entity['new_entities_correct_pct']:.2f}%)"
    )
    print(f"new clients wrongly attached: {entity['new_entities_wrongly_attached']:,}")
    print(f"wrong existing attachment profiles: {profile['wrong_existing_attachment_profiles']:,}")
    print(f"ambiguous profiles: {profile['ambiguous_profiles']:,}")
    final_quality = evaluation.get("final_graph_quality", {})
    if final_quality:
        pair = final_quality.get("pair_level", {})
        final_merge = final_quality.get("final_merge_level", {})
        print("\n=== FINAL GRAPH QUALITY ===")
        print(
            "pair precision/recall/F1: "
            f"{pair.get('precision', 0):.3f} / {pair.get('recall', 0):.3f} / {pair.get('f1', 0):.3f}"
        )
        print(
            "final merge precision/recall/F1: "
            f"{final_merge.get('precision', 0):.3f} / {final_merge.get('recall', 0):.3f} / {final_merge.get('f1', 0):.3f}"
        )


# Читает параметры запуска inference из командной строки.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph-table production inference for a parquet packet.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    return parser.parse_args()


# Читает parquet-пакет, распаковывает нужные признаки и собирает core/value таблицы.
def build_packet_frames(args: argparse.Namespace, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from graph_table_pipeline import (
        build_packet_profile_values_from_flat,
        build_profile_core_from_flat,
    )
    from graph_table_utils import prepare_graph_table_input_df

    logger.info("read input parquet")
    raw_df = pd.read_parquet(args.input)
    logger.info("raw rows=%s columns=%s", f"{len(raw_df):,}", f"{raw_df.shape[1]:,}")
    if args.max_rows is not None:
        raw_df = raw_df.head(args.max_rows).copy()
        logger.info("limited raw rows=%s", f"{len(raw_df):,}")
    logger.info("prepare input dataframe")
    work_df = prepare_graph_table_input_df(raw_df)
    logger.info("prepared rows=%s columns=%s profiles=%s", f"{len(work_df):,}", f"{work_df.shape[1]:,}", f"{work_df['profile_id'].nunique():,}")

    logger.info("build packet core/value frames")
    if args.split is not None and "split" in work_df.columns:
        work_df = work_df[work_df["split"].astype(str).eq(str(args.split))].copy()
        logger.info("filtered split=%s rows=%s profiles=%s", args.split, f"{len(work_df):,}", f"{work_df['profile_id'].nunique():,}")
    packet_core = build_profile_core_from_flat(work_df)
    packet_values = build_packet_profile_values_from_flat(work_df)
    return raw_df, packet_core, packet_values


# Загружает артефакты и запускает graph-table scoring.
def run_scoring(
    packet_core: pd.DataFrame,
    packet_values: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    from graph_table_pipeline import load_artifacts, score_packet

    artifacts = load_artifacts(args.artifact_dir)
    assignment, pair_scores, metrics = score_packet(packet_core, packet_values, artifacts, args.score_threshold)
    metrics["pairs"] = len(pair_scores)
    return assignment, metrics


# Сохраняет минимальный набор результата inference: assignment, метрики, evaluation и HTML.
def save_inference_outputs(
    assignment: pd.DataFrame,
    metrics: dict,
    raw_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    assignment_path = args.out_dir / f"assignment_{timestamp}.parquet"
    assignment_csv_path = args.out_dir / f"assignment_{timestamp}.csv"
    metrics_path = args.out_dir / f"metrics_{timestamp}.json"

    assignment.to_parquet(assignment_path, index=False)
    assignment.to_csv(assignment_csv_path, index=False)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    evaluation = evaluate_assignment_if_labels_available(assignment, raw_df, args.artifact_dir)
    evaluation_path = None
    if evaluation is not None:
        evaluation_path = args.out_dir / f"evaluation_{timestamp}.json"
        evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    html_report_path = save_html_report(assignment, metrics, evaluation, args.out_dir, timestamp)
    return {
        "timestamp": timestamp,
        "assignment_path": assignment_path,
        "assignment_csv_path": assignment_csv_path,
        "metrics_path": metrics_path,
        "evaluation": evaluation,
        "evaluation_path": evaluation_path,
        "html_report_path": html_report_path,
    }


# Печатает ключевые метрики и пути к файлам в лог и консоль.
def log_and_print_outputs(metrics: dict, outputs: dict[str, object], logger: logging.Logger) -> None:
    evaluation = outputs["evaluation"]
    logger.info(
        "summary profiles=%s candidate_pairs=%s accepted_edges=%s matched_existing=%s new_clients=%s ambiguous=%s",
        metrics.get("profiles"),
        metrics.get("candidate_pairs"),
        metrics.get("accepted_edges"),
        metrics.get("matched_existing"),
        metrics.get("new_clients"),
        metrics.get("ambiguous"),
    )
    logger.info("assignment_parquet=%s", outputs["assignment_path"])
    logger.info("assignment_csv=%s", outputs["assignment_csv_path"])
    logger.info("metrics_json=%s", outputs["metrics_path"])
    if outputs["evaluation_path"] is not None:
        logger.info("evaluation_json=%s", outputs["evaluation_path"])
    logger.info("html_report=%s", outputs["html_report_path"])
    logger.info("done")

    print_main_metrics(metrics, evaluation)
    print(f"assignment_parquet={outputs['assignment_path']}")
    print(f"assignment_csv={outputs['assignment_csv_path']}")
    print(f"metrics_json={outputs['metrics_path']}")
    print(f"html_report={outputs['html_report_path']}")


# Как работает скрипт run_graph_table_inference.py:
# 1. parse_args() читает путь к входному parquet, каталог артефактов, порог score и ограничения.
# 2. build_packet_frames() читает входной пакет и локально распаковывает нужные признаки,
#    затем строит packet_core и packet_values в формате, который понимает graph_table_pipeline.
# 3. run_scoring() загружает production-артефакты и вызывает score_packet().
#    Внутри score_packet строится blocking index, генерируются пары, считаются признаки пары,
#    XGBoost скорит пары, mutual top-K оставляет рёбра, NetworkX собирает компоненты графа.
# 4. save_inference_outputs() сохраняет assignment, метрики, проверку качества и HTML-отчёт.
# 5. log_and_print_outputs() выводит основные числа и пути к файлам в лог и консоль.
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir)
    try:
        logger.info("start graph-table inference")
        logger.info("input=%s", args.input)
        logger.info("artifact_dir=%s", args.artifact_dir)
        logger.info("score_threshold=%s max_rows=%s split=%s", args.score_threshold, args.max_rows, args.split)

        raw_df, packet_core, packet_values = build_packet_frames(args, logger)
        assignment, metrics = run_scoring(packet_core, packet_values, args)
        outputs = save_inference_outputs(assignment, metrics, raw_df, args)
        log_and_print_outputs(metrics, outputs, logger)
    finally:
        close_logger(logger)


if __name__ == "__main__":
    main()
