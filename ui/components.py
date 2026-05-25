import streamlit as st
import pandas as pd

def render_file_uploader() -> pd.DataFrame | None:
    uploaded_file = st.file_uploader(
        "Загрузите parquet-файл",
        type=["parquet"],
    )

    if uploaded_file is None:
        return None

    try:
        df = pd.read_parquet(uploaded_file)
    except Exception as exc:
        st.error(f"Не удалось прочитать parquet: {exc}")
        return None

    return df

def render_inference_controls() -> tuple[str, float]:
    mode = st.selectbox(
        "Mode",
        options=["graph", "knn"],
        index=0,
    )

    score_threshold = st.slider(
        "Score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    return mode, score_threshold