from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

FS_FLAG_FEATURES = {
    "is_gmail",
    "is_man",
    "is_phone",
    "is_woman",
    "is_yandex",
    "was_phone_lead",
}


# Маленькие технические функции без бизнес-логики.
# Они нужны в разных частях pipeline: чтение JSON, нормализация,
# hash-bucket и временные bucket-ключи.

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_key_value_items(value: Any) -> dict[str, Any]:
    """Распаковывает список значений вида `feature:value` из сырого входного пакета."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    # Parquet возвращает array-поля как numpy.ndarray, а не как обычный list.
    # Без этого np/fs значения теряются до формирования blocking-правил.
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        return {}

    parsed: dict[str, Any] = {}
    for item in value:
        text = str(item)
        if ":" in text:
            key, item_value = text.split(":", 1)
            parsed[key.strip()] = item_value.strip()
        else:
            parsed[text] = None
    return parsed


def parse_json_features(value: Any) -> dict[str, Any]:
    """Распаковывает realtime-признаки, которые приходят словарём или JSON-строкой."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _expand_feature_source(
    df: pd.DataFrame,
    raw_column: str,
    prefix: str,
    parser: Any,
    *,
    flag_features: set[str] | None = None,
) -> pd.DataFrame:
    """Создаёт колонки с префиксом источника из вложенной сырой колонки."""
    if raw_column not in df.columns:
        return df

    parsed = df[raw_column].map(parser)
    features = sorted({feature for values in parsed for feature in values})
    flags = flag_features or set()
    for feature in features:
        target = f"{prefix}_{feature}"
        if feature in flags:
            df[target] = parsed.map(lambda values: feature in values)
        else:
            df[target] = parsed.map(lambda values: values.get(feature))
    return df


def prepare_graph_table_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Минимально готовит сырой пакет для blocking, признаков пары и inference."""
    work = df.copy()
    if "created_at" in work.columns:
        work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")

    # Сохраняем исходные имена `np_*`, `rt_*`, `fs_*`: внешние алиасы pipeline не нужны.
    work = _expand_feature_source(work, "non_processing_features", "np", parse_key_value_items)
    work = _expand_feature_source(work, "realtime_features", "rt", parse_json_features)
    work = _expand_feature_source(
        work,
        "fs_features",
        "fs",
        parse_key_value_items,
        flag_features=FS_FLAG_FEATURES,
    )
    return work


def normalize_value(value: Any, *, feature: str | None = None) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return None
    if feature == "phone":
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits or None
    return text


def stable_hash_bucket(value: Any, n_buckets: int) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    bucket = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % n_buckets
    return str(bucket)


def registration_60m_bucket_values(profile_id: Any, created_at: Any) -> list[dict[str, str]]:
    timestamp = pd.to_datetime(created_at, errors="coerce")
    if pd.isna(timestamp):
        return []
    bucket_start = timestamp.floor("30min")
    rows = []
    for offset in [0, -1, -2]:
        bucket = bucket_start + pd.to_timedelta(offset * 30, unit="m")
        rows.append(
            {
                "profile_id": str(profile_id),
                "source": "derived",
                "feature": "registration_60m_bucket",
                "value_norm": bucket.strftime("%Y-%m-%d %H:%M"),
            }
        )
    return rows


def prefix6(value: Any) -> str | None:
    return value[:6] if isinstance(value, str) and len(value) >= 6 else None


def daypart_bucket(value: Any) -> str | None:
    try:
        hour = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    if hour < 0 or hour > 23:
        return None
    if hour <= 5:
        return "sleep_00_05"
    if hour <= 10:
        return "morning_06_10"
    if hour <= 17:
        return "work_11_17"
    return "evening_18_23"


def weekend_bucket(value: Any) -> str | None:
    try:
        day = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    if day < 0 or day > 6:
        return None
    return "weekend" if day in {5, 6} else "weekday"

