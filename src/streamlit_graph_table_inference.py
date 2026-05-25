from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_graph_table_inference import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_PACKET,
    build_packet_frames,
    close_logger,
    log_and_print_outputs,
    run_scoring,
    save_inference_outputs,
    setup_logger,
)


# Открывает стандартный диалог выбора parquet-файла на машине, где запущен Streamlit.
def choose_parquet_file(initial_path: Path) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        st.warning(f"Не удалось открыть системный диалог выбора файла: {exc}")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Выберите входной parquet",
        initialdir=str(initial_path.parent if initial_path.parent.exists() else Path.cwd()),
        filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


# Открывает стандартный диалог выбора каталога на машине, где запущен Streamlit.
def choose_directory(initial_path: Path, title: str) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        st.warning(f"Не удалось открыть системный диалог выбора каталога: {exc}")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        title=title,
        initialdir=str(initial_path if initial_path.exists() else Path.cwd()),
    )
    root.destroy()
    return Path(selected) if selected else None


# Обновляет путь к parquet через callback, чтобы не нарушать правила session_state Streamlit.
def select_input_path() -> None:
    current_path = Path(st.session_state.get("input_path", DEFAULT_PACKET))
    selected = choose_parquet_file(current_path)
    if selected is not None:
        st.session_state["input_path"] = str(selected)


# Обновляет путь к каталогу артефактов через callback.
def select_artifact_dir() -> None:
    current_path = Path(st.session_state.get("artifact_dir", DEFAULT_ARTIFACT_DIR))
    selected = choose_directory(current_path, "Выберите каталог production-артефактов")
    if selected is not None:
        st.session_state["artifact_dir"] = str(selected)


# Обновляет путь к каталогу результата через callback.
def select_out_dir() -> None:
    current_path = Path(st.session_state.get("out_dir", DEFAULT_OUT_DIR))
    selected = choose_directory(current_path, "Выберите каталог для результатов inference")
    if selected is not None:
        st.session_state["out_dir"] = str(selected)


# Собирает объект с параметрами в том же формате, что и CLI-скрипт inference.
def make_args(
    input_path: Path,
    artifact_dir: Path,
    out_dir: Path,
    score_threshold: float | None,
    max_rows: int | None,
    split: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        input=input_path,
        artifact_dir=artifact_dir,
        out_dir=out_dir,
        score_threshold=score_threshold,
        max_rows=max_rows,
        split=split,
    )


# Берет только понятные колонки assignment для просмотра в интерфейсе.
def compact_assignment_view(assignment: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "profile_id",
        "decision",
        "predicted_entity_id",
        "best_match_score",
        "component_profile_ids",
    ]
    existing_columns = [column for column in columns if column in assignment.columns]
    result = assignment[existing_columns].copy()
    if "best_match_score" in result.columns:
        result["best_match_score"] = pd.to_numeric(result["best_match_score"], errors="coerce").round(4)
    return result


# Показывает основные runtime-метрики крупными числами.
def show_runtime_metrics(metrics: dict) -> None:
    first_row = st.columns(4)
    first_row[0].metric("Профилей", f"{metrics.get('profiles', 0):,}".replace(",", " "))
    first_row[1].metric("Пар после блокинга", f"{metrics.get('candidate_pairs', 0):,}".replace(",", " "))
    first_row[2].metric("Связей принято", f"{metrics.get('accepted_edges', 0):,}".replace(",", " "))
    first_row[3].metric("Найдено в истории", f"{metrics.get('matched_existing', 0):,}".replace(",", " "))

    second_row = st.columns(3)
    second_row[0].metric("Новые клиенты", f"{metrics.get('new_clients', 0):,}".replace(",", " "))
    second_row[1].metric("Неоднозначно", f"{metrics.get('ambiguous', 0):,}".replace(",", " "))
    second_row[2].metric("Порог score", metrics.get("threshold", "из артефактов"))


# Показывает контрольные метрики, если во входном пакете есть entity_id.
def show_quality_metrics(evaluation: dict | None) -> None:
    if not evaluation:
        st.info("Во входном пакете нет entity_id или historical_profile_core.parquet. Проверку качества пропускаем.")
        return

    entity = evaluation.get("entity_level", {})
    profile = evaluation.get("profile_level", {})
    final_quality = evaluation.get("final_graph_quality", {})
    final_merge = final_quality.get("final_merge_level", {})

    cols = st.columns(4)
    cols[0].metric("Существующие найдены", f"{entity.get('existing_entities_found_pct', 0):.2f}%")
    cols[1].metric("Новые верно", f"{entity.get('new_entities_correct_pct', 0):.2f}%")
    cols[2].metric("Ошибочно приклеены", f"{entity.get('new_entities_wrongly_attached', 0):,}".replace(",", " "))
    cols[3].metric("Неоднозначно", f"{profile.get('ambiguous_profiles', 0):,}".replace(",", " "))

    if final_merge:
        st.caption(
            "Качество финальных объединений: "
            f"precision={final_merge.get('precision', 0):.3f}, "
            f"recall={final_merge.get('recall', 0):.3f}, "
            f"F1={final_merge.get('f1', 0):.3f}"
        )


# Запускает тот же inference-пайплайн, что и CLI-скрипт.
def run_inference(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, dict[str, object]]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir)
    try:
        logger.info("start streamlit graph-table inference")
        logger.info("input=%s", args.input)
        logger.info("artifact_dir=%s", args.artifact_dir)
        logger.info("score_threshold=%s max_rows=%s split=%s", args.score_threshold, args.max_rows, args.split)

        raw_df, packet_core, packet_values = build_packet_frames(args, logger)
        assignment, metrics = run_scoring(packet_core, packet_values, args)
        outputs = save_inference_outputs(assignment, metrics, raw_df, args)
        log_and_print_outputs(metrics, outputs, logger)
        return assignment, metrics, outputs
    finally:
        close_logger(logger)


# Основной Streamlit-экран: параметры запуска, кнопка запуска и отображение результата.
def main() -> None:
    st.set_page_config(page_title="ER inference", layout="wide")
    st.title("Entity Resolution: локальный inference")
    st.caption("Интерфейс запускает тот же graph-table pipeline, что и run_graph_table_inference.py.")

    st.session_state.setdefault("input_path", str(DEFAULT_PACKET))
    st.session_state.setdefault("artifact_dir", str(DEFAULT_ARTIFACT_DIR))
    st.session_state.setdefault("out_dir", str(DEFAULT_OUT_DIR))

    with st.sidebar:
        st.header("Параметры запуска")
        input_path = Path(st.text_input("Входной parquet", key="input_path"))
        st.button("Выбрать parquet", on_click=select_input_path)

        artifact_dir = Path(st.text_input("Каталог артефактов", key="artifact_dir"))
        st.button("Выбрать каталог артефактов", on_click=select_artifact_dir)

        out_dir = Path(st.text_input("Каталог результата", key="out_dir"))
        st.button("Выбрать каталог результата", on_click=select_out_dir)

        use_artifact_threshold = st.checkbox("Использовать порог из артефактов", value=True)
        score_threshold = None
        if not use_artifact_threshold:
            score_threshold = st.slider("score_threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.01)

        limit_rows = st.checkbox("Ограничить число строк", value=False)
        max_rows = None
        if limit_rows:
            max_rows = st.number_input("max_rows", min_value=1, value=1000, step=100)

        split_value = st.text_input("split, если нужен", value="")
        split = split_value.strip() or None

        run_button = st.button("Запустить inference", type="primary")

    if not run_button:
        st.info("Выберите параметры слева и нажмите «Запустить inference».")
        return

    if not input_path.exists():
        st.error(f"Не найден входной parquet: {input_path}")
        return
    if not artifact_dir.exists():
        st.error(f"Не найден каталог артефактов: {artifact_dir}")
        return

    args = make_args(input_path, artifact_dir, out_dir, score_threshold, max_rows, split)
    with st.spinner("Считаем inference. Для больших пакетов это может занять время."):
        assignment, metrics, outputs = run_inference(args)

    st.success("Inference завершён.")
    show_runtime_metrics(metrics)

    st.subheader("Проверка качества")
    show_quality_metrics(outputs.get("evaluation"))

    st.subheader("Найденные профили")
    st.dataframe(compact_assignment_view(assignment), use_container_width=True, height=420)

    st.subheader("HTML-отчёт")
    html_path = Path(outputs["html_report_path"])
    html_text = html_path.read_text(encoding="utf-8")
    components.html(html_text, height=1000, scrolling=True)

    st.subheader("Файлы результата")
    st.write(f"assignment parquet: `{outputs['assignment_path']}`")
    st.write(f"assignment csv: `{outputs['assignment_csv_path']}`")
    st.write(f"metrics json: `{outputs['metrics_path']}`")
    if outputs["evaluation_path"] is not None:
        st.write(f"evaluation json: `{outputs['evaluation_path']}`")
    st.write(f"html report: `{outputs['html_report_path']}`")

    with open(outputs["assignment_csv_path"], "rb") as file:
        st.download_button("Скачать assignment CSV", file, file_name=Path(outputs["assignment_csv_path"]).name)
    with open(html_path, "rb") as file:
        st.download_button("Скачать HTML-отчёт", file, file_name=html_path.name)


if __name__ == "__main__":
    main()
