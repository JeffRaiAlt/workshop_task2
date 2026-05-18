from __future__ import annotations

"""Build an interactive HTML report for graph/tree ER experiment outputs.

This script intentionally lives outside `graph_tree_edge_policy_experiment.py`.
The model script builds data and metrics; this script only reads saved CSV files
and renders a human-friendly report.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


BASE = Path("data/processed")
GRAPH_DIR = BASE / "er_graph_tree_policy"
MART_DIR = BASE / "er_profile_mart_multivalue"
OUT_PATH = GRAPH_DIR / "tree_policy_visual_report.html"


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(require_file(GRAPH_DIR / "tree_policy_selected_report.csv"))
    feature_importance = pd.read_csv(require_file(GRAPH_DIR / "tree_policy_feature_importance.csv"))
    recommended_rules = pd.read_csv(require_file(MART_DIR / "recommended_blocking_rules.csv"))
    selected = normalize_selected_report(selected)
    return selected, feature_importance, recommended_rules


def normalize_selected_report(selected: pd.DataFrame) -> pd.DataFrame:
    """Support both the new compact metric schema and older saved reports."""
    if "test_correct_client_decisions_pct" in selected.columns:
        return selected

    full_path = GRAPH_DIR / "tree_policy_full_threshold_report.csv"
    if not full_path.exists():
        raise ValueError(
            "tree_policy_selected_report.csv uses the old metric schema, "
            "and tree_policy_full_threshold_report.csv is not available to recover counts. "
            "Run graph_tree_edge_policy_experiment.py first."
        )

    full = pd.read_csv(full_path)
    test_rows = full[full["split"].astype(str).eq("test")].copy()
    recovered = []
    for row in selected.itertuples(index=False):
        policy = getattr(row, "policy")
        threshold = getattr(row, "selected_threshold")
        match = test_rows[test_rows["policy"].astype(str).eq(str(policy))]
        if "threshold" in match.columns:
            match = match[match["threshold"].round(10).eq(round(float(threshold), 10))]
        if "constraint_dataset" in selected.columns and "constraint_dataset" in match.columns:
            constraint = getattr(row, "constraint_dataset", None)
            if pd.notna(constraint):
                match = match[match["constraint_dataset"].astype(str).eq(str(constraint))]
        if match.empty:
            recovered.append({})
            continue
        m = match.iloc[0]
        duplicate_total = int(m.get("duplicate_entities", 0))
        duplicate_found = int(m.get("found_duplicate_entities", 0))
        singleton_total = int(m.get("singleton_entities", 0))
        singleton_correct = int(m.get("correct_singleton_entities", 0))
        predicted_groups = int(m.get("predicted_non_singleton_components", 0))
        wrong_groups = int(m.get("false_merge_components", 0))
        recovered.append(
            {
                "test_clients_total": duplicate_total + singleton_total,
                "test_existing_clients_total": duplicate_total,
                "test_existing_clients_found": duplicate_found,
                "test_existing_clients_found_pct": 100 * float(m.get("duplicate_entity_recall", 0.0)),
                "test_existing_clients_missed": duplicate_total - duplicate_found,
                "test_existing_clients_missed_pct": 100 * (1 - float(m.get("duplicate_entity_recall", 0.0))),
                "test_new_clients_total": singleton_total,
                "test_new_clients_correct": singleton_correct,
                "test_new_clients_correct_pct": 100 * float(m.get("singleton_entity_precision", 0.0)),
                "test_new_clients_wrongly_attached": singleton_total - singleton_correct,
                "test_new_clients_wrongly_attached_pct": 100 * (1 - float(m.get("singleton_entity_precision", 0.0))),
                "test_predicted_merge_groups": predicted_groups,
                "test_wrong_merge_groups": wrong_groups,
                "test_wrong_merge_groups_pct": 100 * float(m.get("false_merge_rate", 0.0)),
                "test_graph_edges": int(m.get("edges_after_threshold_topk", 0)),
                "test_correct_client_decisions_pct": 100 * float(m.get("overall_entity_decision_acc", 0.0)),
            }
        )

    recovered_df = pd.DataFrame(recovered)
    return pd.concat([selected.reset_index(drop=True), recovered_df], axis=1)


def sort_policies(report: pd.DataFrame) -> pd.DataFrame:
    return report.sort_values(
        ["test_correct_client_decisions_pct", "test_wrong_merge_groups_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_summary_figure(best: pd.Series) -> go.Figure:
    labels = [
        "Нашли существующих",
        "Пропустили существующих",
        "Новых верно",
        "Новых ошибочно приклеили",
        "Ошибочные группы склейки",
    ]
    values = [
        best["test_existing_clients_found_pct"],
        best["test_existing_clients_missed_pct"],
        best["test_new_clients_correct_pct"],
        best["test_new_clients_wrongly_attached_pct"],
        best["test_wrong_merge_groups_pct"],
    ]
    counts = [
        f"{int(best['test_existing_clients_found']):,} / {int(best['test_existing_clients_total']):,}",
        f"{int(best['test_existing_clients_missed']):,} / {int(best['test_existing_clients_total']):,}",
        f"{int(best['test_new_clients_correct']):,} / {int(best['test_new_clients_total']):,}",
        f"{int(best['test_new_clients_wrongly_attached']):,} / {int(best['test_new_clients_total']):,}",
        f"{int(best['test_wrong_merge_groups']):,} / {int(best['test_predicted_merge_groups']):,}",
    ]
    colors = ["#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#991b1b"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%<br>{c}" for v, c in zip(values, counts)],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Лучший вариант: {best['policy']}",
        xaxis_title="Процент",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=180, r=30, t=50, b=25),
    )
    return fig


def build_policy_comparison(top: pd.DataFrame) -> go.Figure:
    labels = top["policy"].astype(str)
    if "constraint_dataset" in top.columns:
        labels = labels + "<br>" + top["constraint_dataset"].fillna("").astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Нашли существующих",
            x=labels,
            y=top["test_existing_clients_found_pct"],
            marker_color="#2563eb",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Новых верно",
            x=labels,
            y=top["test_new_clients_correct_pct"],
            marker_color="#16a34a",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Ошибочные склейки",
            x=labels,
            y=top["test_wrong_merge_groups_pct"],
            marker_color="#dc2626",
        )
    )
    fig.update_layout(
        title="Топ политик: основные проценты",
        yaxis_title="Процент",
        barmode="group",
        height=430,
        margin=dict(l=60, r=30, t=55, b=135),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def build_feature_importance(feature_importance: pd.DataFrame) -> go.Figure:
    top_features = (
        feature_importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_features["importance"],
            y=top_features["feature"],
            orientation="h",
            marker_color="#7c3aed",
            hovertemplate="%{y}<br>mean importance=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Средняя важность признаков дерева",
        xaxis_title="Mean importance",
        yaxis_title="",
        height=430,
        margin=dict(l=220, r=30, t=55, b=25),
    )
    return fig


def build_recommended_rules_table(recommended_rules: pd.DataFrame) -> go.Figure:
    rules = recommended_rules[
        recommended_rules["recommended_for_next_step"].astype(str).str.lower().eq("true")
    ].copy()
    rules = rules.sort_values(["block_family", "positive_recall", "candidate_pairs"], ascending=[True, False, True])

    columns = [
        "block_family",
        "block_rule",
        "positive_recall",
        "positive_pairs_captured",
        "candidate_pairs",
        "pairs_per_positive_captured",
        "max_block_size",
        "profile_coverage",
    ]
    table = rules[[col for col in columns if col in rules.columns]].copy()
    if "positive_recall" in table.columns:
        table["positive_recall"] = table["positive_recall"].map(lambda x: f"{100 * x:.1f}%")
    if "profile_coverage" in table.columns:
        table["profile_coverage"] = table["profile_coverage"].map(lambda x: f"{100 * x:.1f}%")
    if "candidate_pairs" in table.columns:
        table["candidate_pairs"] = table["candidate_pairs"].map(lambda x: f"{int(x):,}")
    if "positive_pairs_captured" in table.columns:
        table["positive_pairs_captured"] = table["positive_pairs_captured"].map(lambda x: f"{int(x):,}")
    if "pairs_per_positive_captured" in table.columns:
        table["pairs_per_positive_captured"] = table["pairs_per_positive_captured"].map(lambda x: f"{x:,.1f}")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(table.columns), fill_color="#f3f4f6", align="left"),
                cells=dict(values=[table[col] for col in table.columns], align="left"),
                columnwidth=[110, 520, 95, 140, 140, 170, 110, 110],
            )
        ]
    )
    fig.update_layout(
        title="Recommended blocking rules: правила и основные показатели",
        height=32 * len(table) + 95,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


def build_recommended_rules_explanation() -> go.Figure:
    rows = [
        ("block_family", "Семейство правила. Показывает общий тип сигнала: гео/context, поведение, композитный сигнал или identity-rescue."),
        ("block_rule", "Конкретное правило, по которому строится блок. Например: гео + устройство + ОС или точное совпадение телефона."),
        ("positive_recall", "Доля настоящих пар-дублей, которые это правило смогло поймать среди всех true pairs в датасете."),
        ("positive_pairs_captured", "Сколько настоящих пар-дублей попало в кандидаты через это правило."),
        ("candidate_pairs", "Сколько всего пар-кандидатов создаёт правило. Чем больше число, тем тяжелее и потенциально шумнее правило."),
        ("pairs_per_positive_captured", "Сколько candidate pairs приходится на одну найденную true pair. Меньше значит чище правило."),
        ("max_block_size", "Максимальный размер блока: сколько profile_id оказалось в самом большом значении этого правила."),
        ("profile_coverage", "Доля профилей, которые вообще попали хотя бы в один блок этого правила."),
        ("context", "Гео/context правила: город, geoname, subdivision. Хороши для покрытия, но сами по себе могут быть широкими."),
        ("coverage_compound", "Композиты для покрытия: несколько простых context/device признаков вместе, чтобы сузить широкий блок."),
        ("behavior", "Одиночные поведенческие fs/site-id признаки. Часто дают recall, но могут быть шумными."),
        ("behavior_context", "Комбинация гео/context + behavior. Обычно точнее одиночного behavior."),
        ("behavior_context_device", "Комбинация гео/context + behavior + устройство/ОС. Ещё сильнее сужает область поиска."),
        ("identity_rescue", "Узкие identity-правила, например точный телефон. Покрытие маленькое, но сигнал обычно очень точный."),
        ("coverage_fallback", "Технический fallback для полного покрытия профилей. Полезен как страховка, но слаб как основание для склейки."),
    ]
    table = pd.DataFrame(rows, columns=["термин", "что означает"])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(table.columns), fill_color="#f3f4f6", align="left"),
                cells=dict(values=[table[col] for col in table.columns], align="left"),
                columnwidth=[170, 720],
            )
        ]
    )
    fig.update_layout(
        title="Расшифровка колонок и семейств blocking-правил",
        height=30 * len(table) + 95,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


def write_html(figures: list[go.Figure], output_path: Path) -> None:
    parts = [
        "<!doctype html>",
        '<html lang="ru">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Graph tree ER report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;color:#111827;background:#fff;}",
        "h1{margin-bottom:4px;} .muted{color:#6b7280;margin-top:0;}",
        ".section{margin-top:10px;}",
        ".section .plotly-graph-div{margin:0!important;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Graph + tree ER report</h1>",
        f'<p class="muted">Сгенерировано: {datetime.now():%Y-%m-%d %H:%M:%S}</p>',
    ]

    for i, fig in enumerate(figures):
        parts.append('<div class="section">')
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
        parts.append("</div>")

    parts.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    selected_report, feature_importance, recommended_rules = load_inputs()
    sorted_report = sort_policies(selected_report)
    top = sorted_report.head(10).copy()
    best = sorted_report.iloc[0]

    figures = [
        build_summary_figure(best),
        build_policy_comparison(top),
        build_recommended_rules_table(recommended_rules),
        build_recommended_rules_explanation(),
        build_feature_importance(feature_importance),
    ]
    write_html(figures, OUT_PATH)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
