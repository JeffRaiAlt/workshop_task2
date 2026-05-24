import pandas as pd

from core.features import build_profile_df, build_pair_features
from core.candidates import (
    generate_candidate_pairs_2hop,
    generate_candidate_pairs_knn,
)
from core.model import (
    load_catboost_model,
    load_feature_cols,
    predict_pairs_catboost,
)


def run_inference_pipeline(
    df: pd.DataFrame,
    split: str = "test",
    score_threshold: float = 0.5,
):
    profile_df = build_profile_df(df, split=split)

    if len(profile_df) < 2:
        return (
            pd.DataFrame(columns=["profile_id_1", "profile_id_2", "score"]),
            {
                "profiles": len(profile_df),
                "pairs": 0,
                "model_mode": "none",
            },
        )

    attribute_cols = {
        "domain": "domain_set",
        "phone": "phone_set",
        "device": "device_set",
        "city": "city_set",
        "region": "region_set",
    }

    candidate_pairs = generate_candidate_pairs_2hop(
        profile_df,
        attribute_cols,
    )

    numeric_cols = [
        "row_count",
        "n_email",
        "n_phone",
        "n_domain",
        "n_device",
        "n_city",
        "n_region",
        "n_browser",
        "n_os",
    ]

    candidate_pairs |= generate_candidate_pairs_knn(
        profile_df,
        numeric_cols=numeric_cols,
        n_neighbors=10,
    )

    if not candidate_pairs:
        return (
            pd.DataFrame(columns=["profile_id_1", "profile_id_2", "score"]),
            {
                "profiles": len(profile_df),
                "pairs": 0,
                "model_mode": "no_pairs",
            },
        )

    pairs_df = pd.DataFrame(
        sorted(candidate_pairs),
        columns=["profile_id_1", "profile_id_2"],
    )

    pairs_feat_df = build_pair_features(
        pairs_df,
        profile_df,
    )

    cat_model = load_catboost_model()
    feature_cols = load_feature_cols()

    if cat_model is None or feature_cols is None:
        raise RuntimeError(
            "Не найден CatBoost или feature_cols.json. "
            "Положи артефакты в artifacts/."
        )

    scored = predict_pairs_catboost(
        cat_model,
        pairs_feat_df,
        feature_cols,
    )

    scored = scored[scored["score"] >= score_threshold].reset_index(drop=True)

    metrics = {
        "profiles": len(profile_df),
        "pairs": len(scored),
        "model_mode": "catboost",
        "threshold": score_threshold,
    }

    return scored, metrics