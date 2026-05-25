import pandas as pd

from core.preprocess import prepare_input_df
from core.pipeline import run_inference_pipeline, artifacts_status as _artifacts_status


def artifacts_status():
    return _artifacts_status()


def inference(df: pd.DataFrame, max_rows: int = 5000, mode: str = 'table', score_threshold: float = 0.5):
    work_df = df.head(max_rows).copy()
    work_df = prepare_input_df(work_df)
    split_values = set(work_df['split'].astype(str)) if 'split' in work_df.columns else {'test'}
    split = 'test' if 'test' in split_values else sorted(split_values)[0]
    pred_df, metrics = run_inference_pipeline(work_df, split=split, score_threshold=score_threshold)
    if mode == "graph":
        pred_df, metrics = run_inference_pipeline(
            work_df,
            split=split,
            score_threshold=score_threshold,
        )
        if not pred_df.empty:
            pred_df = pred_df.rename(
                columns={
                    "profile_id_1": "source",
                    "profile_id_2": "target",
                    "score": "weight",
                }
            )

    elif mode == "table":
        raise NotImplementedError("Table pipeline пока не реализован")

    else:
        raise ValueError(f"Неизвестный mode: {mode}")

    notes = "\n".join([
        f"- Обработано строк: {len(work_df)}",
        f"- Использованный split: {split}",
        f"- Количество агрегированных профилей: {metrics.get('profiles', 0)}",
        f"- Количество пар после threshold: {metrics.get('pairs', 0)}",
        f"- Режим модели: {metrics.get('model_mode', 'unknown')}",
        f"- Порог score: {metrics.get('threshold', score_threshold)}"
    ])
    return pred_df, notes