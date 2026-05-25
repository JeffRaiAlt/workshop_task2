import numpy as np
import pandas as pd


def to_set(series):
    vals = series.dropna().astype(str).unique().tolist()
    return set(v for v in vals if v not in ('nan', 'None', ''))


def mode_or_nan(series):
    s = series.dropna().astype(str)
    s = s[~s.isin(('nan', 'None', ''))]
    return s.mode().iat[0] if len(s) else np.nan


def same_any(a, b):   return int(bool(a & b))
def intersect_size(a, b): return len(a & b)
def jaccard(a, b):
    u = len(a | b); return len(a & b) / u if u else 0.0
def overlap_coef(a, b):
    d = min(len(a), len(b)); return len(a & b) / d if d else 0.0


def build_profile_df(df: pd.DataFrame, split: str = 'test') -> pd.DataFrame:
    df_part = df[df['split'] == split].copy()

    cast_cols = ['profile_id', 'entity_id', 'email', 'phone', 'email_domain',
                 'device', 'city_name', 'region_code', 'browser', 'osfamily', 'sex']
    for col in cast_cols:
        if col in df_part.columns:
            df_part[col] = df_part[col].astype(str)
            df_part.loc[df_part[col].isin(('nan', 'None')), col] = np.nan

    set_map = {
        'email_set': 'email', 'phone_set': 'phone', 'domain_set': 'email_domain',
        'device_set': 'device', 'city_set': 'city_name', 'region_set': 'region_code',
        'browser_set': 'browser', 'os_set': 'osfamily', 'sex_set': 'sex',
    }
    agg_dict = {'row_count': ('profile_id', 'size')}
    if 'entity_id' in df_part.columns:
        agg_dict['entity_id'] = ('entity_id', mode_or_nan)
    for dst, src in set_map.items():
        if src in df_part.columns:
            agg_dict[dst] = (src, to_set)
        else:
            agg_dict[dst] = ('profile_id', lambda x: set())

    optional_numeric = ['local_hour', 'visit_count', 'city_population',
                        'fs_is_phone', 'fs_is_gmail', 'fs_is_yandex', 'fs_is_man', 'fs_is_woman']
    for col in optional_numeric:
        if col in df_part.columns:
            agg_dict[f'{col}_mean'] = (col, 'mean')

    profile_df = df_part.groupby('profile_id').agg(**agg_dict).reset_index()

    for src_col, cnt_col in [
        ('email_set','n_email'), ('phone_set','n_phone'), ('domain_set','n_domain'),
        ('device_set','n_device'), ('city_set','n_city'), ('region_set','n_region'),
        ('browser_set','n_browser'), ('os_set','n_os'),
    ]:
        if src_col in profile_df.columns:
            profile_df[cnt_col] = profile_df[src_col].apply(len)

    return profile_df


def build_pair_features(pairs_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    pairs_df = pairs_df.merge(profile_df.add_suffix('_1'), left_on='profile_id_1',
                               right_on='profile_id_1', how='left')
    pairs_df = pairs_df.merge(profile_df.add_suffix('_2'), left_on='profile_id_2',
                               right_on='profile_id_2', how='left')

    set_pairs = [
        ('phone', 'phone_set'), ('email', 'email_set'), ('domain', 'domain_set'),
        ('device', 'device_set'), ('city', 'city_set'), ('region', 'region_set'),
        ('browser', 'browser_set'), ('os', 'os_set'), ('sex', 'sex_set'),
    ]
    for name, col in set_pairs:
        c1, c2 = f'{col}_1', f'{col}_2'
        if c1 in pairs_df.columns and c2 in pairs_df.columns:
            pairs_df[f'same_{name}_any']    = pairs_df.apply(lambda r: same_any(r[c1], r[c2]), axis=1)
            pairs_df[f'common_{name}_cnt']  = pairs_df.apply(lambda r: intersect_size(r[c1], r[c2]), axis=1)

    for name, col in [('domain','domain_set'),('device','device_set'),('city','city_set'),
                      ('phone','phone_set'),('email','email_set')]:
        c1, c2 = f'{col}_1', f'{col}_2'
        if c1 in pairs_df.columns and c2 in pairs_df.columns:
            pairs_df[f'{name}_jaccard'] = pairs_df.apply(lambda r: jaccard(r[c1], r[c2]), axis=1)
            pairs_df[f'{name}_overlap'] = pairs_df.apply(lambda r: overlap_coef(r[c1], r[c2]), axis=1)

    numeric_base_cols = [
        'row_count', 'n_email', 'n_phone', 'n_domain', 'n_device',
        'n_city', 'n_region', 'n_browser', 'n_os',
        'local_hour_mean', 'visit_count_mean', 'city_population_mean',
        'fs_is_phone_mean', 'fs_is_gmail_mean', 'fs_is_yandex_mean',
        'fs_is_man_mean', 'fs_is_woman_mean',
    ]
    for col in numeric_base_cols:
        c1, c2 = f'{col}_1', f'{col}_2'
        if c1 in pairs_df.columns and c2 in pairs_df.columns:
            pairs_df[f'{col}_diff'] = (pairs_df[c1] - pairs_df[c2]).abs()
            pairs_df[f'{col}_min']  = pairs_df[[c1, c2]].min(axis=1)
            pairs_df[f'{col}_max']  = pairs_df[[c1, c2]].max(axis=1)

    return pairs_df
