import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"



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