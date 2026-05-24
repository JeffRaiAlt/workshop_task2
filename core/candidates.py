from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


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