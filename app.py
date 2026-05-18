import html
import io
import time

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from services import inference, artifacts_status
from report import build_markdown_report


def safe_read_parquet(file) -> pd.DataFrame:
    try:
        data = file.read()
        buf = io.BytesIO(data)
        table = pq.read_table(buf)
        return table.to_pandas()
    except Exception as e:
        raise ValueError(f"Ошибка чтения parquet: {e}")


st.set_page_config(page_title="Profile Matching Production App", layout="wide")
st.title("🎯 Система матчинга и дедупликации профилей клиентов")

# Статус артефактов в раскрывающемся блоке
with st.expander("📦 Статус артефактов", expanded=True):
    status = artifacts_status()
    st.json(status)

# Вкладки вместо сайдбара
tab1, tab2, tab3 = st.tabs(["📁 Загрузка файла", "⚙️ Режим обработки", "📊 Настройки анализа"])

with tab1:
    uploaded_file = st.file_uploader(
        "Загрузите файл в формате parquet", 
        type=["parquet"],
        help="Поддерживаются только .parquet файлы"
    )

with tab2:
    mode_label = st.radio(
        "Режим обработки",
        options=["Блокинг", "Графовый"],
        index=0,
        horizontal=True,
        help="Блокинг - быстрый, Графовый - точнее но требует больше ресурсов"
    )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        max_rows_input = st.number_input(
            "Максимальное количество строк для анализа",
            min_value=10,
            max_value=500_000,
            value=5000,
            step=100,
            help="Ограничение на количество загружаемых строк"
        )
    with col2:
        score_threshold = st.slider(
            "Порог score", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="Минимальный score для включения в результаты"
        )

# Определяем mode для дальнейшего использования
mode = "graph" if "Графовый" in mode_label else "table"

if uploaded_file is None:
    st.info("📁 Загрузите parquet-файл во вкладке 'Загрузка файла', чтобы начать.")
    st.stop()

st.write("### 📊 Предпросмотр исходных данных")
try:
    df_raw = safe_read_parquet(uploaded_file)
    st.write(f"✅ Файл успешно прочитан. Размер: **{df_raw.shape[0]}** строк, **{df_raw.shape[1]}** колонок.")
    st.dataframe(df_raw.head(10), use_container_width=True)
except ValueError as e:
    st.error(str(e))
    st.stop()

# Настройки анализа
st.write("### ⚙️ Настройки анализа")

col1, col2 = st.columns([2, 1])

with col1:
    rows_count = st.slider(
        "Количество строк для обработки",
        min_value=10,
        max_value=int(min(max_rows_input, max(10, len(df_raw)))),
        value=int(min(1000, len(df_raw), max_rows_input)),
        step=10,
        help="Сколько строк данных будет обработано"
    )

with col2:
    st.write("")
    st.write("")
    run_inference = st.button(
        "🚀 Запустить inference",
        type="primary",
        use_container_width=True
    )

st.divider()

# Запуск обработки только после нажатия кнопки
if run_inference:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🔄 Обрабатываю данные..."):
        try:
            # Имитация прогресса (можно убрать если inference сам показывает прогресс)
            for i in range(100):
                time.sleep(0.01)  # Убрать в реальном коде
                progress_bar.progress(i + 1)
                status_text.text(f"Шаг {i + 1}/100")
            
            predict_df, notes = inference(
                df_raw,
                max_rows=rows_count,
                mode=mode,
                score_threshold=score_threshold,
            )
            md_report = build_markdown_report(predict_df, notes=notes, mode=mode)
            
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.exception(e)
            st.stop()
        else:
            st.success("✅ Inference успешно завершен!")
            
            # Результаты
            st.subheader("📊 Результаты обработки")
            st.write(f"Найдено пар: **{len(predict_df)}**")
            st.dataframe(predict_df.head(500), use_container_width=True)
            
            st.subheader("📝 Markdown-отчет")
            st.markdown(md_report)
            
            # Улучшенная кнопка копирования
            escaped_md = html.escape(md_report)
            copy_button = f'''
            <textarea id="md_to_copy" style="position:absolute;left:-9999px;">{escaped_md}</textarea>
            <button onclick="navigator.clipboard.writeText(document.getElementById('md_to_copy').value); this.innerText='✅ Скопировано'; setTimeout(() => this.innerText='📋 Копировать markdown', 2000);" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
                📋 Копировать markdown
            </button>
            '''
            st.components.v1.html(copy_button, height=50)
else:
    st.info("👆 Нажмите кнопку **Запустить inference**, чтобы начать обработку данных")