import json
import pickle
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'


def to_set(series):
    vals = series.dropna().astype(str).unique().tolist()
    vals = [v for v in vals if v not in ['nan', 'None', '']]
    return set(vals)


def mode_or_nan(series):
    s = series.dropna().astype(str)
    s = s[~s.isin(['nan', 'None', ''])]
    if len(s) == 0:
        return np.nan
    return s.mode().iat[0]


def same_any(a, b):
    return int(len(a & b) > 0)


def intersect_size(a, b):
    return len(a & b)


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def overlap_coef(a, b):
    denom = min(len(a), len(b))
    return len(a & b) / denom if denom else 0.0


def artifacts_status():
    expected = [
        'catboost_pair_model.cbm',
        'feature_cols.json',
        'graph_schema.pkl',
        'graphsage_state_dict.pt',
        'summary.json',
    ]
    status = {name: (ARTIFACTS_DIR / name).exists() for name in expected}
    status['artifacts_dir'] = str(ARTIFACTS_DIR)
    status['catboost_available'] = CatBoostClassifier is not None
    return status


def load_feature_cols():
    path = ARTIFACTS_DIR / 'feature_cols.json'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ['feature_cols', 'features', 'columns']:
                if key in data and isinstance(data[key], list):
                    return data[key]
    return None


def load_catboost_model():
    model_path = ARTIFACTS_DIR / 'catboost_pair_model.cbm'
    if not model_path.exists() or CatBoostClassifier is None:
        return None
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def build_profile_df(df: pd.DataFrame, split: str = 'test') -> pd.DataFrame:
    df_part = df[df['split'] == split].copy()
    cast_cols = ['profile_id', 'entity_id', 'email', 'phone', 'email_domain', 'device', 'city_name', 'region_code', 'browser', 'osfamily', 'sex']
    for col in cast_cols:
        if col in df_part.columns:
            df_part[col] = df_part[col].astype(str)
            df_part.loc[df_part[col].isin(['nan', 'None']), col] = np.nan
    agg_dict = {
        'entity_id': ('entity_id', mode_or_nan) if 'entity_id' in df_part.columns else ('profile_id', mode_or_nan),
        'row_count': ('profile_id', 'size'),
        'email_set': ('email', to_set) if 'email' in df_part.columns else ('profile_id', lambda x: set()),
        'phone_set': ('phone', to_set) if 'phone' in df_part.columns else ('profile_id', lambda x: set()),
        'domain_set': ('email_domain', to_set) if 'email_domain' in df_part.columns else ('profile_id', lambda x: set()),
        'device_set': ('device', to_set) if 'device' in df_part.columns else ('profile_id', lambda x: set()),
        'city_set': ('city_name', to_set) if 'city_name' in df_part.columns else ('profile_id', lambda x: set()),
        'region_set': ('region_code', to_set) if 'region_code' in df_part.columns else ('profile_id', lambda x: set()),
        'browser_set': ('browser', to_set) if 'browser' in df_part.columns else ('profile_id', lambda x: set()),
        'os_set': ('osfamily', to_set) if 'osfamily' in df_part.columns else ('profile_id', lambda x: set()),
        'sex_set': ('sex', to_set) if 'sex' in df_part.columns else ('profile_id', lambda x: set()),
    }
    optional_numeric = ['local_hour', 'visit_count', 'city_population', 'fs_is_phone', 'fs_is_gmail', 'fs_is_yandex', 'fs_is_man', 'fs_is_woman']
    for col in optional_numeric:
        if col in df_part.columns:
            agg_dict[f'{col}_mean'] = (col, 'mean')
    profile_df = df_part.groupby('profile_id').agg(**agg_dict).reset_index()
    for src, dst in [('email_set', 'n_email'), ('phone_set', 'n_phone'), ('domain_set', 'n_domain'), ('device_set', 'n_device'), ('city_set', 'n_city'), ('region_set', 'n_region'), ('browser_set', 'n_browser'), ('os_set', 'n_os')]:
        profile_df[dst] = profile_df[src].apply(len)
    return profile_df


def generate_candidate_pairs_2hop(profile_df: pd.DataFrame, attribute_cols: dict, max_group_size: int = 200):
    candidate_pairs = set()
    for attr_type, set_col in attribute_cols.items():
        if set_col not in profile_df.columns:
            continue
        inv_index = defaultdict(list)
        for _, row in profile_df[['profile_id', set_col]].iterrows():
            p = row['profile_id']
            for v in row[set_col]:
                inv_index[(attr_type, v)].append(p)
        for _, prof_list in inv_index.items():
            prof_list = sorted(set(prof_list))
            if 2 <= len(prof_list) <= max_group_size:
                for a, b in combinations(prof_list, 2):
                    candidate_pairs.add((a, b))
    return candidate_pairs


def generate_candidate_pairs_knn(profile_df: pd.DataFrame, numeric_cols: list, n_neighbors=20):
    use_cols = [c for c in numeric_cols if c in profile_df.columns]
    if not use_cols or len(profile_df) < 2:
        return set()
    X = profile_df[use_cols].fillna(0.0).values
    neighbors = min(n_neighbors, len(profile_df))
    nn = NearestNeighbors(n_neighbors=neighbors, metric='cosine')
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    profiles = profile_df['profile_id'].tolist()
    candidate_pairs = set()
    for i, neigh_idx in enumerate(indices):
        p1 = profiles[i]
        for j in neigh_idx[1:]:
            p2 = profiles[j]
            a, b = sorted([p1, p2])
            candidate_pairs.add((a, b))
    return candidate_pairs


def build_pair_features(pairs_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    pairs_df = pairs_df.merge(profile_df.add_suffix('_1'), left_on='profile_id_1', right_on='profile_id_1', how='left')
    pairs_df = pairs_df.merge(profile_df.add_suffix('_2'), left_on='profile_id_2', right_on='profile_id_2', how='left')
    pairs_df['same_phone_any'] = pairs_df.apply(lambda r: same_any(r.phone_set_1, r.phone_set_2), axis=1)
    pairs_df['same_email_any'] = pairs_df.apply(lambda r: same_any(r.email_set_1, r.email_set_2), axis=1)
    pairs_df['same_domain_any'] = pairs_df.apply(lambda r: same_any(r.domain_set_1, r.domain_set_2), axis=1)
    pairs_df['same_device_any'] = pairs_df.apply(lambda r: same_any(r.device_set_1, r.device_set_2), axis=1)
    pairs_df['same_city_any'] = pairs_df.apply(lambda r: same_any(r.city_set_1, r.city_set_2), axis=1)
    pairs_df['same_region_any'] = pairs_df.apply(lambda r: same_any(r.region_set_1, r.region_set_2), axis=1)
    pairs_df['same_browser_any'] = pairs_df.apply(lambda r: same_any(r.browser_set_1, r.browser_set_2), axis=1)
    pairs_df['same_os_any'] = pairs_df.apply(lambda r: same_any(r.os_set_1, r.os_set_2), axis=1)
    pairs_df['same_sex_any'] = pairs_df.apply(lambda r: same_any(r.sex_set_1, r.sex_set_2), axis=1)
    pairs_df['common_phone_cnt'] = pairs_df.apply(lambda r: intersect_size(r.phone_set_1, r.phone_set_2), axis=1)
    pairs_df['common_email_cnt'] = pairs_df.apply(lambda r: intersect_size(r.email_set_1, r.email_set_2), axis=1)
    pairs_df['common_domain_cnt'] = pairs_df.apply(lambda r: intersect_size(r.domain_set_1, r.domain_set_2), axis=1)
    pairs_df['common_device_cnt'] = pairs_df.apply(lambda r: intersect_size(r.device_set_1, r.device_set_2), axis=1)
    pairs_df['common_city_cnt'] = pairs_df.apply(lambda r: intersect_size(r.city_set_1, r.city_set_2), axis=1)
    pairs_df['common_region_cnt'] = pairs_df.apply(lambda r: intersect_size(r.region_set_1, r.region_set_2), axis=1)
    pairs_df['domain_jaccard'] = pairs_df.apply(lambda r: jaccard(r.domain_set_1, r.domain_set_2), axis=1)
    pairs_df['device_jaccard'] = pairs_df.apply(lambda r: jaccard(r.device_set_1, r.device_set_2), axis=1)
    pairs_df['city_jaccard'] = pairs_df.apply(lambda r: jaccard(r.city_set_1, r.city_set_2), axis=1)
    pairs_df['phone_jaccard'] = pairs_df.apply(lambda r: jaccard(r.phone_set_1, r.phone_set_2), axis=1)
    pairs_df['email_jaccard'] = pairs_df.apply(lambda r: jaccard(r.email_set_1, r.email_set_2), axis=1)
    pairs_df['domain_overlap'] = pairs_df.apply(lambda r: overlap_coef(r.domain_set_1, r.domain_set_2), axis=1)
    pairs_df['device_overlap'] = pairs_df.apply(lambda r: overlap_coef(r.device_set_1, r.device_set_2), axis=1)
    pairs_df['city_overlap'] = pairs_df.apply(lambda r: overlap_coef(r.city_set_1, r.city_set_2), axis=1)
    numeric_base_cols = ['row_count', 'n_email', 'n_phone', 'n_domain', 'n_device', 'n_city', 'n_region', 'n_browser', 'n_os', 'local_hour_mean', 'visit_count_mean', 'city_population_mean', 'fs_is_phone_mean', 'fs_is_gmail_mean', 'fs_is_yandex_mean', 'fs_is_man_mean', 'fs_is_woman_mean']
    for col in numeric_base_cols:
        c1 = f'{col}_1'
        c2 = f'{col}_2'
        if c1 in pairs_df.columns and c2 in pairs_df.columns:
            pairs_df[f'{col}_diff'] = (pairs_df[c1] - pairs_df[c2]).abs()
            pairs_df[f'{col}_min'] = pairs_df[[c1, c2]].min(axis=1)
            pairs_df[f'{col}_max'] = pairs_df[[c1, c2]].max(axis=1)
    return pairs_df


def predict_pairs_catboost(cat_model, pairs_feat_df: pd.DataFrame, feature_cols: list):
    X = pairs_feat_df.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    pred = cat_model.predict_proba(X)[:, 1]
    out = pairs_feat_df[['profile_id_1', 'profile_id_2']].copy()
    out['score'] = pred
    return out.sort_values('score', ascending=False).reset_index(drop=True)


def run_inference_pipeline(df: pd.DataFrame, split: str = 'test', score_threshold: float = 0.5):
    profile_df = build_profile_df(df, split=split)
    if len(profile_df) < 2:
        return pd.DataFrame(columns=['profile_id_1', 'profile_id_2', 'score']), {'profiles': len(profile_df), 'pairs': 0, 'model_mode': 'none'}

    attribute_cols = {'domain': 'domain_set', 'phone': 'phone_set', 'device': 'device_set', 'city': 'city_set', 'region': 'region_set'}
    candidate_pairs = generate_candidate_pairs_2hop(profile_df, attribute_cols)
    numeric_cols = ['row_count', 'n_email', 'n_phone', 'n_domain', 'n_device', 'n_city', 'n_region', 'n_browser', 'n_os']
    candidate_pairs |= generate_candidate_pairs_knn(profile_df, numeric_cols=numeric_cols, n_neighbors=10)

    if not candidate_pairs:
        return pd.DataFrame(columns=['profile_id_1', 'profile_id_2', 'score']), {'profiles': len(profile_df), 'pairs': 0, 'model_mode': 'no_pairs'}

    pairs_df = pd.DataFrame(sorted(candidate_pairs), columns=['profile_id_1', 'profile_id_2'])
    pairs_feat_df = build_pair_features(pairs_df, profile_df)

    cat_model = load_catboost_model()
    feature_cols = load_feature_cols()

    cat_model = load_catboost_model()
    feature_cols = load_feature_cols()

    if cat_model is None or feature_cols is None:
        raise RuntimeError(
            "Не найден CatBoost или feature_cols.json. "
            "Положи артефакты в artifacts/ или включи эвристический fallback."
        )

    scored = predict_pairs_catboost(cat_model, pairs_feat_df, feature_cols)
    model_mode = "catboost"

    scored = scored[scored['score'] >= score_threshold].reset_index(drop=True)
    metrics = {
        'profiles': len(profile_df),
        'pairs': len(scored),
        'model_mode': model_mode,
        'threshold': score_threshold,
    }
    return scored, metrics
