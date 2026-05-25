from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SRC_DIR / "graph_table_artifacts"
DEFAULT_OUT_DIR = ROOT_DIR / "reports" / "model_eval" / "shap"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_table_definitions import MODEL_FEATURE_DESCRIPTIONS
from graph_table_pipeline import apply_mutual_top_k
from graph_table_report_template import render_report_template


def find_latest_pair_scores() -> Path:
    """Находит последний сохранённый набор оценённых пар-кандидатов."""
    candidates = sorted(
        (ROOT_DIR / "reports" / "model_eval").glob("pair_scores_*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Не найден reports/model_eval/pair_scores_*.parquet. Сначала выполните перебор порогов.")
    return candidates[0]


def infer_packet_path(pair_scores_path: Path) -> Path | None:
    """Находит исходный тестовый пакет по имени файла оценённых пар."""
    stem = pair_scores_path.stem
    marker = "pair_scores_"
    policy_marker = "_xgb_"
    if not stem.startswith(marker) or policy_marker not in stem:
        return None
    packet_stem = stem[len(marker) : stem.index(policy_marker)]
    candidate = SRC_DIR / "graph_table_test_data" / f"{packet_stem}.parquet"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Построить SHAP-отчёт для текущей graph-table модели XGBoost.")
    parser.add_argument("--pair-scores", type=Path, default=None, help="Файл оценённых пар-кандидатов из перебора порогов.")
    parser.add_argument("--packet", type=Path, default=None, help="Размеченный тестовый пакет, на котором оценивались пары.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Порог связей для раздела о принятых ребрах; по умолчанию используется policy_config.json.",
    )
    parser.add_argument(
        "--max-shap-rows",
        type=int,
        default=0,
        help="0 = объяснить все оценённые пары; другое число = взять стабильную выборку указанного размера.",
    )
    parser.add_argument("--plot-rows", type=int, default=5000, help="Число строк только для отображения диаграммы распределения SHAP.")
    return parser.parse_args()


def read_config_and_features(artifact_dir: Path) -> tuple[dict, list[str], object]:
    """Загружает рабочую конфигурацию, список признаков и обученную модель."""
    config = json.loads((artifact_dir / "policy_config.json").read_text(encoding="utf-8"))
    features = json.loads((artifact_dir / "feature_cols.json").read_text(encoding="utf-8"))
    model = joblib.load(artifact_dir / config["model_file"])
    return config, features, model


def label_pairs(pair_scores: pd.DataFrame, packet_path: Path | None, artifact_dir: Path) -> pd.DataFrame:
    """Добавляет правильный ответ для пары, если доступен размеченный тестовый пакет."""
    result = pair_scores.copy()
    if packet_path is None or not packet_path.exists():
        return result
    packet = pd.read_parquet(packet_path, columns=["profile_id", "entity_id"]).dropna().drop_duplicates()
    history = pd.read_parquet(
        artifact_dir / "historical_profile_core.parquet", columns=["profile_id", "entity_id"]
    ).dropna().drop_duplicates()
    entities = pd.concat([history, packet], ignore_index=True).drop_duplicates("profile_id")
    mapping = dict(zip(entities["profile_id"].astype(str), entities["entity_id"].astype(str)))
    result["is_true_pair"] = (
        result["profile_id_l"].astype(str).map(mapping).eq(result["profile_id_r"].astype(str).map(mapping))
    )
    return result


def calculate_shap_values(
    pair_scores: pd.DataFrame,
    features: list[str],
    model: object,
    max_rows: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Считает SHAP для оценки модели на всех или на выбранном числе пар."""
    explained = pair_scores if max_rows <= 0 or len(pair_scores) <= max_rows else pair_scores.sample(
        n=max_rows, random_state=42
    )
    explained = explained.copy().reset_index(drop=True)
    x = explained[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x))
    if values.ndim == 3:
        values = values[:, :, 1]
    return explained, values


def build_importance(
    explained: pd.DataFrame,
    shap_values: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    """Собирает рейтинг признаков по абсолютному влиянию на оценку модели."""
    accepted_mask = explained["accepted_edge"].to_numpy(dtype=bool)
    true_accepted_mask = accepted_mask & explained.get("is_true_pair", pd.Series(False, index=explained.index)).to_numpy()
    false_accepted_mask = accepted_mask & ~explained.get("is_true_pair", pd.Series(False, index=explained.index)).to_numpy()
    rows = []
    for index, feature in enumerate(features):
        column = shap_values[:, index]
        rows.append(
            {
                "feature": feature,
                "description": MODEL_FEATURE_DESCRIPTIONS.get(feature, ""),
                "mean_abs_shap_all_pairs": float(np.abs(column).mean()),
                "mean_shap_all_pairs": float(column.mean()),
                "mean_abs_shap_accepted_edges": float(np.abs(column[accepted_mask]).mean()) if accepted_mask.any() else np.nan,
                "mean_shap_accepted_edges": float(column[accepted_mask].mean()) if accepted_mask.any() else np.nan,
                "mean_shap_true_accepted_edges": float(column[true_accepted_mask].mean()) if true_accepted_mask.any() else np.nan,
                "mean_shap_false_accepted_edges": float(column[false_accepted_mask].mean()) if false_accepted_mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap_all_pairs", ascending=False).reset_index(drop=True)


def save_bar_chart(importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    """Сохраняет основной график влияния признаков на оценку модели."""
    selected = importance.head(top_n).sort_values("mean_abs_shap_all_pairs")
    plt.figure(figsize=(10, 6.2))
    plt.barh(selected["feature"], selected["mean_abs_shap_all_pairs"], color="#1b7f79")
    plt.title("SHAP: влияние признаков на оценку XGBoost")
    plt.xlabel("Среднее абсолютное влияние SHAP")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_beeswarm(
    explained: pd.DataFrame,
    shap_values: np.ndarray,
    features: list[str],
    path: Path,
    plot_rows: int,
) -> None:
    """Сохраняет график направления влияния признаков."""
    if len(explained) > plot_rows:
        selected = explained.sample(n=plot_rows, random_state=43).index.to_numpy()
    else:
        selected = explained.index.to_numpy()
    x = explained.loc[selected, features].replace([np.inf, -np.inf], np.nan).fillna(0)
    shap.summary_plot(shap_values[selected], x, feature_names=features, max_display=15, show=False)
    plt.title("SHAP: направление влияния признаков на оценку модели")
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def image_data_uri(path: Path) -> str:
    """Встраивает PNG внутрь HTML, чтобы отчет оставался одним переносимым файлом."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def format_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100 * value:.2f}%"


def save_html_report(
    path: Path,
    importance: pd.DataFrame,
    bar_path: Path,
    beeswarm_path: Path,
    metadata: dict,
) -> None:
    """Передает рассчитанные значения в отдельный HTML-шаблон SHAP-отчета."""
    accepted_text = (
        f"{metadata['accepted_edges']:,} связей, из них правильных {metadata['accepted_true_edges']:,} "
        f"и ошибочных {metadata['accepted_false_edges']:,}"
        if metadata["labels_available"]
        else f"{metadata['accepted_edges']:,} связей; разметка пары недоступна"
    )
    html_text = render_report_template(
        "graph_table_shap_report.html",
        policy_name=metadata["policy_name"],
        packet_name=metadata["packet_name"],
        threshold=metadata["threshold"],
        candidate_pairs=f"{metadata['candidate_pairs']:,}",
        explained_pairs=f"{metadata['explained_pairs']:,}",
        accepted_edges=f"{metadata['accepted_edges']:,}",
        accepted_precision=format_pct(metadata["accepted_precision"]),
        accepted_text=accepted_text,
        top_rows=importance.head(12).to_dict(orient="records"),
        bar_image=image_data_uri(bar_path),
        beeswarm_image=image_data_uri(beeswarm_path),
    )
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    pair_scores_path = args.pair_scores or find_latest_pair_scores()
    packet_path = args.packet or infer_packet_path(pair_scores_path)
    config, features, model = read_config_and_features(args.artifact_dir)
    threshold = float(args.score_threshold if args.score_threshold is not None else config.get("score_threshold", 0.95))
    top_k = config.get("graph_top_k", 1)

    print(f"читаем оценки пар={pair_scores_path}")
    pair_scores = pd.read_parquet(pair_scores_path)
    missing_features = [feature for feature in features if feature not in pair_scores.columns]
    if missing_features:
        raise ValueError(f"В файле оценённых пар отсутствуют признаки модели: {missing_features}")
    pair_scores = label_pairs(pair_scores, packet_path, args.artifact_dir)
    accepted_edges = apply_mutual_top_k(pair_scores, "score", threshold, top_k)
    accepted_keys = set(
        zip(accepted_edges["profile_id_l"].astype(str), accepted_edges["profile_id_r"].astype(str))
    )
    pair_scores["accepted_edge"] = [
        (str(left), str(right)) in accepted_keys
        for left, right in pair_scores[["profile_id_l", "profile_id_r"]].itertuples(index=False, name=None)
    ]

    print(f"считаем SHAP candidate_pairs={len(pair_scores):,} accepted_edges={len(accepted_edges):,}")
    explained, shap_values = calculate_shap_values(pair_scores, features, model, args.max_shap_rows)
    importance = build_importance(explained, shap_values, features)

    threshold_tag = str(threshold).replace(".", "_")
    report_dir = args.out_dir / f"{pair_scores_path.stem}_threshold_{threshold_tag}"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "shap_feature_importance.csv"
    bar_path = report_dir / "shap_feature_importance.png"
    beeswarm_path = report_dir / "shap_summary_beeswarm.png"
    html_path = report_dir / "shap_report.html"

    importance.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_bar_chart(importance, bar_path)
    save_beeswarm(explained, shap_values, features, beeswarm_path, args.plot_rows)

    labels_available = "is_true_pair" in pair_scores.columns
    accepted_true = int(accepted_edges["is_true_pair"].sum()) if labels_available else 0
    accepted_false = int(len(accepted_edges) - accepted_true) if labels_available else 0
    metadata = {
        "policy_name": config.get("policy_name", "unknown"),
        "packet_name": packet_path.name if packet_path is not None else "not provided",
        "threshold": threshold,
        "candidate_pairs": int(len(pair_scores)),
        "explained_pairs": int(len(explained)),
        "accepted_edges": int(len(accepted_edges)),
        "labels_available": labels_available,
        "accepted_true_edges": accepted_true,
        "accepted_false_edges": accepted_false,
        "accepted_precision": accepted_true / len(accepted_edges) if labels_available and len(accepted_edges) else np.nan,
    }
    save_html_report(html_path, importance, bar_path, beeswarm_path, metadata)

    print("\n=== ГЛАВНЫЕ ПРИЗНАКИ ПО SHAP ===")
    print(importance[["feature", "mean_abs_shap_all_pairs", "mean_shap_accepted_edges"]].head(15).to_string(index=False))
    print(f"\nhtml={html_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
