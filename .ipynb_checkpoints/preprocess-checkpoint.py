import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def safe_convert(x):
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if not isinstance(x, (list, tuple, set)):
        return {}
    result = {}
    for item in x:
        if ':' in str(item):
            parts = str(item).split(':', 1)
            if len(parts) == 2:
                key, value = parts
                result[key.strip()] = value.strip()
            else:
                result[str(item)] = None
        else:
            result[str(item)] = None
    return result


def json_convert(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def expand_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'non_processing_features' in df.columns:
        df['non_processing_features'] = df['non_processing_features'].apply(safe_convert)
        np_keys = set()
        for d in df['non_processing_features'].dropna():
            if isinstance(d, dict):
                np_keys.update(d.keys())
        for key in np_keys:
            df[f'np_{key}'] = df['non_processing_features'].apply(lambda x: x.get(key) if isinstance(x, dict) else None)
        rename_map = {
            'np_device': 'device',
            'np_osfamily': 'osfamily',
            'np_browser': 'browser',
            'np_geoname_id': 'geoname_id',
            'np_subdivision_1_iso_code': 'region_code',
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
    if 'realtime_features' in df.columns:
        df['realtime_features'] = df['realtime_features'].apply(json_convert)
        rt_keys = set()
        for d in df['realtime_features'].dropna():
            if isinstance(d, dict):
                rt_keys.update(d.keys())
        for key in rt_keys:
            df[f'rt_{key}'] = df['realtime_features'].apply(lambda x: x.get(key) if isinstance(x, dict) else None)
        rename_map_rt = {
            'rt_country': 'country',
            'rt_is_million': 'is_million_city',
            'rt_tz_offset': 'tz_offset',
            'rt_geoname': 'city_name',
            'rt_geoid': 'city_geoid',
            'rt_local_hour': 'local_hour',
            'rt_day': 'day_of_week_from_rt',
            'rt_population': 'city_population',
            'rt_visit_count': 'visit_count',
        }
        for old, new in rename_map_rt.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
    if 'fs_features' in df.columns:
        df['fs_features'] = df['fs_features'].apply(safe_convert)
        fs_keys = set()
        for d in df['fs_features'].dropna():
            if isinstance(d, dict):
                fs_keys.update(d.keys())
        flags = {'is_gmail', 'is_man', 'is_phone', 'is_woman', 'is_yandex', 'was_phone_lead'}
        for key in fs_keys:
            col = f'fs_{key}'
            if key in flags:
                df[col] = df['fs_features'].apply(lambda x: key in x if isinstance(x, dict) else False)
            else:
                df[col] = df['fs_features'].apply(lambda x: x.get(key) if isinstance(x, dict) else None)
    return df


def ensure_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'split' in df.columns:
        return df
    if 'entity_id' in df.columns:
        entity_df = df.groupby('entity_id', as_index=False).agg(row_count=('entity_id', 'size'))
        if len(entity_df) >= 3:
            train_entity_df, test_entity_df = train_test_split(entity_df, test_size=0.2, random_state=42)
            train_entities = set(train_entity_df['entity_id'])
            df['split'] = np.where(df['entity_id'].isin(train_entities), 'train', 'test')
        else:
            df['split'] = 'test'
    else:
        df['split'] = 'test'
    return df


def prepare_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
    if 'email' in df.columns and 'email_domain' not in df.columns:
        df['email_domain'] = df['email'].astype(str).str.split('@').str[1]
    df = expand_all_features(df)
    df = ensure_split(df)
    return df
