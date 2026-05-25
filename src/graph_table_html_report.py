from __future__ import annotations

from pathlib import Path

import pandas as pd

from graph_table_report_template import render_report_template


# Форматирует целое число для HTML-отчета.
def format_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(value):,}".replace(",", " ")


# Форматирует долю, которая уже хранится в процентах.
def format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1f}%"


# Форматирует score модели, который хранится как число от 0 до 1.
def format_score(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


# Собирает уникальные группы profile_id, которые граф предлагает объединить.
def merge_group_rows(assignment: pd.DataFrame, threshold: float | None, top_n: int = 50) -> list[dict[str, object]]:
    if assignment.empty:
        return []
    threshold = 0.95 if threshold is None else float(threshold)
    rows = assignment.copy()
    rows["component_profile_ids"] = rows["component_profile_ids"].fillna("").astype(str)
    rows = rows[rows["component_profile_ids"].str.count(r"\|").ge(1)].copy()
    if rows.empty:
        return []

    rows["best_match_score"] = pd.to_numeric(rows["best_match_score"], errors="coerce").fillna(0.0)
    rows = (
        rows.groupby("component_profile_ids", as_index=False)
        .agg(best_match_score=("best_match_score", "max"))
        .sort_values(["best_match_score", "component_profile_ids"], ascending=[False, True])
        .head(top_n)
    )

    result = []
    for group_no, row in enumerate(rows.itertuples(index=False), start=1):
        score = float(getattr(row, "best_match_score", 0.0))
        if score >= max(0.98, threshold):
            recommendation = "объединить"
        else:
            recommendation = "проверить перед объединением"
        result.append(
            {
                "group_no": group_no,
                "profile_ids": str(getattr(row, "component_profile_ids", "")).split("|"),
                "best_match_score": format_score(score),
                "recommendation": recommendation,
            }
        )
    return result


# Берет из metrics только самые частые правила, которые реально дали связи графа.
def accepted_rule_rows(metrics: dict, limit: int = 8) -> list[dict[str, object]]:
    rows = metrics.get("rule_usage", {}).get("accepted_rules", [])
    return rows[:limit] if isinstance(rows, list) else []


# Генерирует компактный HTML-отчет по результату inference.
def save_html_report(
    assignment: pd.DataFrame,
    metrics: dict,
    evaluation: dict | None,
    out_dir: Path,
    timestamp: str,
) -> Path:
    path = out_dir / f"inference_result_report_{timestamp}.html"
    profile = evaluation.get("profile_level", {}) if evaluation else {}
    final_quality = evaluation.get("final_graph_quality", {}) if evaluation else {}
    pair_quality = final_quality.get("pair_level", {})
    final_merge_quality = final_quality.get("final_merge_level", {})

    merge_groups = merge_group_rows(assignment, metrics.get("threshold"))
    linked_input_profiles = int(
        assignment["component_profile_ids"].fillna("").astype(str).str.count(r"\|").ge(1).sum()
    ) if "component_profile_ids" in assignment.columns else 0
    found_pct = float(profile.get("real_existing_profiles_found_pct", 0.0))
    new_pct = float(profile.get("real_new_profiles_correct_pct", 0.0))
    html_text = render_report_template(
        "graph_table_inference_report.html",
        model_mode=metrics.get("model_mode", "unknown"),
        threshold=metrics.get("threshold"),
        graph_top_k=metrics.get("graph_top_k"),
        timestamp=timestamp,
        profiles=format_int(metrics.get("profiles")),
        candidate_pairs=format_int(metrics.get("candidate_pairs")),
        accepted_edges=format_int(metrics.get("accepted_edges")),
        merge_groups=format_int(len(merge_groups)),
        linked_profiles=format_int(linked_input_profiles),
        profiles_without_links=format_int(int(metrics.get("profiles", 0)) - linked_input_profiles),
        found_pct_bar=f"{found_pct:.2f}",
        found_pct=format_pct(found_pct),
        new_pct_bar=f"{new_pct:.2f}",
        new_pct=format_pct(new_pct),
        pair_precision=pair_quality.get("precision", "n/a") if pair_quality else "n/a",
        pair_recall=pair_quality.get("recall", "n/a") if pair_quality else "n/a",
        pair_f1=pair_quality.get("f1", "n/a") if pair_quality else "n/a",
        final_precision=final_merge_quality.get("precision", "n/a") if final_merge_quality else "n/a",
        final_recall=final_merge_quality.get("recall", "n/a") if final_merge_quality else "n/a",
        final_f1=final_merge_quality.get("f1", "n/a") if final_merge_quality else "n/a",
        merge_groups_rows=merge_groups,
        rule_rows=[
            {
                **row,
                "pair_count": format_int(row.get("pair_count", 0)),
            }
            for row in accepted_rule_rows(metrics)
        ],
    )
    path.write_text(html_text, encoding="utf-8")
    return path
