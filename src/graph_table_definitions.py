from __future__ import annotations

# Здесь лежат статические настройки пайплайна: списки признаков,
# семейства правил и человекочитаемые описания.
# Основной pipeline импортирует их, чтобы не смешивать справочник с расчётом.

# Значения только для прямых similarity-признаков пары: Jaccard/overlap и
# совпадения identity. Это не список всех полей blocking-правил.
# Например, `postman_*` формирует кандидатов через blocking, а в модель
# передаётся как evidence срабатывания семейства `postman_context`.
PAIR_FEATURES = [
    ("identity", "email"),
    ("identity", "phone"),
    ("identity", "first_name"),
    ("identity", "last_name"),
    ("identity", "birthday"),
    ("identity", "sex"),
    ("np", "geoname_id"),
    ("np", "subdivision_1_iso_code"),
    ("np", "device"),
    ("np", "browser"),
    ("np", "osfamily"),
    ("rt", "geoid"),
    ("rt", "geoname"),
    ("rt", "country"),
    ("fs", "source_site_365"),
    ("fs", "source_site_30"),
    ("fs", "visited_30"),
    ("fs", "visited_365"),
    ("fs", "has_account"),
    ("fs", "has_click_365"),
    ("fs", "has_accept_365"),
]

# Прямые колонки профиля, которые не лежат внутри np_/rt_/fs_ namespace.
PACKET_IDENTITY_COLUMNS = ["email", "phone", "first_name", "last_name", "birthday", "sex"]


class BlockFamily:
    CONTEXT = "context"
    BEHAVIOR = "behavior"
    BEHAVIOR_CONTEXT = "behavior_context"
    BEHAVIOR_CONTEXT_DEVICE = "behavior_context_device"
    CLUSTER_CANDIDATE = "cluster_candidate"
    POSTMAN_CONTEXT = "postman_context"
    BEHAVIOR_DAYPART = "behavior_daypart"
    BEHAVIOR_DAYPART_DEVICE = "behavior_daypart_device"
    REGISTRATION_TIME_WINDOW = "registration_time_window"
    IDENTITY_RESCUE = "identity_rescue"
    COVERAGE_COMPOUND = "coverage_compound"
    COVERAGE_FALLBACK = "coverage_fallback"


class MatchScope:
    PACKET = "packet"
    HISTORY = "history"


class FeatureSource:
    IDENTITY = "identity"
    NP = "np"
    RT = "rt"
    FS = "fs"
    DERIVED = "derived"


class TransformName:
    PREFIX6 = "prefix6"
    DAYPART_BUCKET = "daypart_bucket"
    WEEKEND_BUCKET = "weekend_bucket"


class EvidenceColumn:
    PROFILE_ID_L = "profile_id_l"
    PROFILE_ID_R = "profile_id_r"
    PAIR_KEY = "pair_key"
    RULES_KEY = "rules_key"
    BLOCK_FAMILY = "block_family"
    BLOCK_RULE = "block_rule"
    BLOCK_VALUE = "block_value"
    BLOCK_SIZE = "block_size"
    BLOCK_WEIGHT = "block_weight"
    MATCH_SCOPE = "match_scope"
    MATCH_SCOPES = "match_scopes"
    MATCH_SCOPE_SET = "match_scope_set"
    RULES = "rules"
    FAMILIES = "families"

    IS_TIME_AWARE_FAMILY = "is_time_aware_family"
    IS_REGISTRATION_TIME_WINDOW = "is_registration_time_window"
    IS_BEHAVIOR_DAYPART_DEVICE = "is_behavior_daypart_device"
    IS_POSTMAN_CONTEXT = "is_postman_context"
    IS_STRONG_FAMILY = "is_strong_family"
    IS_WEAK_FAMILY = "is_weak_family"
    N_BLOCK_HITS = "n_block_hits"
    N_BLOCK_RULES = "n_block_rules"
    N_BLOCK_FAMILIES = "n_block_families"
    MIN_BLOCK_SIZE = "min_block_size"
    SUM_BLOCK_WEIGHT = "sum_block_weight"
    N_SMALL_BLOCKS_LE2 = "n_small_blocks_le2"
    N_SMALL_BLOCKS_LE5 = "n_small_blocks_le5"
    N_SMALL_BLOCKS_LE10 = "n_small_blocks_le10"
    HIT_COVERAGE_FALLBACK = "hit_coverage_fallback"
    N_STRONG_FAMILY_HITS = "n_strong_family_hits"
    N_WEAK_FAMILY_HITS = "n_weak_family_hits"
    N_TIME_AWARE_HITS = "n_time_aware_hits"
    N_REGISTRATION_TIME_WINDOW_HITS = "n_registration_time_window_hits"
    N_POSTMAN_CONTEXT_HITS = "n_postman_context_hits"
    HIT_BEHAVIOR_DAYPART_DEVICE = "hit_behavior_daypart_device"
    IS_FALLBACK_ONLY = "is_fallback_only"
    HAS_NON_FALLBACK_SIGNAL = "has_non_fallback_signal"
    HAS_SMALL_BLOCK_LE2 = "has_small_block_le2"
    HAS_SMALL_BLOCK_LE5 = "has_small_block_le5"
    HAS_SMALL_BLOCK_LE10 = "has_small_block_le10"
    SMALL_BLOCK_SHARE_LE5 = "small_block_share_le5"
    HAS_STRONG_FAMILY = "has_strong_family"
    STRONG_FAMILY_HIT_SHARE = "strong_family_hit_share"
    HAS_BEHAVIOR = "has_behavior"
    HAS_BEHAVIOR_CONTEXT = "has_behavior_context"
    HAS_POSTMAN_CONTEXT = "has_postman_context"
    POSTMAN_CONTEXT_HIT_SHARE = "postman_context_hit_share"
    HAS_TIME_AWARE_DEVICE_SIGNAL = "has_time_aware_device_signal"
    TIME_AWARE_HIT_SHARE = "time_aware_hit_share"
    HAS_REGISTRATION_TIME_WINDOW = "has_registration_time_window"
    REGISTRATION_TIME_WINDOW_HIT_SHARE = "registration_time_window_hit_share"
    REGISTRATION_TIME_WINDOW_ONLY = "registration_time_window_only"
    REGISTRATION_TIME_WINDOW_WITH_BEHAVIOR = "registration_time_window_with_behavior"
    HAS_CONTEXT = "has_context"
    HAS_COVERAGE_COMPOUND = "has_coverage_compound"
    ONLY_WEAK_FAMILIES = "only_weak_families"
    SMALL_BLOCK_WEAK_FAMILY_ONLY = "small_block_weak_family_only"
    FS_TOTAL_JACCARD = "fs_total_jaccard"
    GEO_TOTAL_JACCARD = "geo_total_jaccard"
    FS_SHARED_COUNT = "fs_shared_count"
    IDENTITY_EMAIL_MATCH = "identity_email_match"
    IDENTITY_PHONE_MATCH = "identity_phone_match"
    IDENTITY_STRONG_MATCH = "identity_strong_match"


BLOCKING_ATOMIC_RULES = [
    {
        "rule": "rule__context__np_geoname_id",
        "source": FeatureSource.NP,
        "feature": "geoname_id",
        "family": BlockFamily.CONTEXT,
        "min_len": 2,
    },
    {
        "rule": "rule__context__np_subdivision",
        "source": FeatureSource.NP,
        "feature": "subdivision_1_iso_code",
        "family": BlockFamily.CONTEXT,
        "min_len": 2,
    },
    {
        "rule": "rule__context__rt_geoid",
        "source": FeatureSource.RT,
        "feature": "geoid",
        "family": BlockFamily.CONTEXT,
        "min_len": 2,
    },
    {
        "rule": "rule__context__rt_geoname",
        "source": FeatureSource.RT,
        "feature": "geoname",
        "family": BlockFamily.CONTEXT,
        "min_len": 2,
    },
    {
        "rule": "rule__identity_rescue__phone_digits",
        "source": FeatureSource.IDENTITY,
        "feature": "phone",
        "family": BlockFamily.IDENTITY_RESCUE,
        "min_len": 7,
    },
    {
        "rule": "rule__identity_rescue__phone_prefix6",
        "source": FeatureSource.IDENTITY,
        "feature": "phone",
        "family": BlockFamily.IDENTITY_RESCUE,
        "transform": TransformName.PREFIX6,
        "min_len": 6,
    },
    {
        "rule": "rule__coverage_fallback__email_hash_bucket_1024",
        "source": FeatureSource.DERIVED,
        "feature": "email_hash_bucket_1024",
        "family": BlockFamily.COVERAGE_FALLBACK,
        "min_len": 1,
    },
]

BLOCKING_ATOMIC_RULE_GROUPS = [
    {
        "rule_template": "rule__behavior__fs_{feature}",
        "source": FeatureSource.FS,
        "family": BlockFamily.BEHAVIOR,
        "min_len": 1,
        "features": [
            "source_site_365",
            "source_site_30",
            "visited_30",
            "visited_365",
            "has_account",
            "has_click_365",
            "has_click_30",
            "has_accept_365",
            "has_accept_30",
            "has_order_365",
            "has_order_30",
            "has_view_90",
        ],
    },
]

BLOCKING_COMPOSITE_RULES = [
    {
        "rule": "rule__registration_time_window__np_geoname_id__registration_60m",
        "family": BlockFamily.REGISTRATION_TIME_WINDOW,
        "specs": [
            (FeatureSource.NP, "geoname_id", "geo", None, 2),
            (FeatureSource.DERIVED, "registration_60m_bucket", "reg_60m", None, 16),
        ],
    },
    {
        "rule": "rule__registration_time_window__np_subdivision__np_device__np_osfamily__registration_60m",
        "family": BlockFamily.REGISTRATION_TIME_WINDOW,
        "specs": [
            (FeatureSource.NP, "subdivision_1_iso_code", "subdivision", None, 2),
            (FeatureSource.NP, "device", "device", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
            (FeatureSource.DERIVED, "registration_60m_bucket", "reg_60m", None, 16),
        ],
    },
    {
        "rule": "rule__cluster_candidate__np_is_not_russia__np_device__np_osfamily__np_browser",
        "family": BlockFamily.CLUSTER_CANDIDATE,
        "specs": [
            (FeatureSource.NP, "is_not_russia", "is_not_russia", None, 1),
            (FeatureSource.NP, "device", "device", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
            (FeatureSource.NP, "browser", "browser", None, 1),
        ],
    },
    {
        "rule": "rule__cluster_candidate__fs_has_account__np_is_not_russia__np_device__np_osfamily__np_browser",
        "family": BlockFamily.CLUSTER_CANDIDATE,
        "specs": [
            (FeatureSource.FS, "has_account", "fs_value", None, 1),
            (FeatureSource.NP, "is_not_russia", "is_not_russia", None, 1),
            (FeatureSource.NP, "device", "device", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
            (FeatureSource.NP, "browser", "browser", None, 1),
        ],
    },
    {
        "rule": "rule__cluster_candidate__fs_postman_campaign_90__fs_postman_action_90__fs_postman_response_90__np_device__np_osfamily",
        "family": BlockFamily.CLUSTER_CANDIDATE,
        "specs": [
            (FeatureSource.FS, "postman_campaign_90", "campaign", None, 1),
            (FeatureSource.FS, "postman_action_90", "action", None, 1),
            (FeatureSource.FS, "postman_response_90", "response", None, 1),
            (FeatureSource.NP, "device", "device", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
        ],
    },
    {
        "rule": "rule__coverage__np_geoname_id__np_device__np_osfamily",
        "family": BlockFamily.COVERAGE_COMPOUND,
        "specs": [
            (FeatureSource.NP, "geoname_id", "geo", None, 2),
            (FeatureSource.NP, "device", "device", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
        ],
    },
    {
        "rule": "rule__coverage__np_geoname_id__np_device__np_osfamily__np_browser",
        "family": BlockFamily.COVERAGE_COMPOUND,
        "specs": [
            (FeatureSource.NP, "geoname_id", "geo", None, 2),
            (FeatureSource.NP, "device", "device", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
            (FeatureSource.NP, "browser", "browser", None, 1),
        ],
    },
    {
        "rule": "rule__coverage__np_subdivision__np_device__np_osfamily",
        "family": BlockFamily.COVERAGE_COMPOUND,
        "specs": [
            (FeatureSource.NP, "subdivision_1_iso_code", "subdivision", None, 2),
            (FeatureSource.NP, "device", "device", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
        ],
    },
    {
        "rule": "rule__coverage__email_domain__np_device__np_osfamily__sex",
        "family": BlockFamily.COVERAGE_COMPOUND,
        "specs": [
            (FeatureSource.DERIVED, "email_domain", "email_domain", None, 1),
            (FeatureSource.NP, "device", "device", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
            (FeatureSource.IDENTITY, "sex", "sex", None, 1),
        ],
    },
    {
        "rule": "rule__coverage__email_domain__email_initial2",
        "family": BlockFamily.COVERAGE_COMPOUND,
        "specs": [
            (FeatureSource.DERIVED, "email_domain", "email_domain", None, 1),
            (FeatureSource.DERIVED, "email_initial2", "email_initial2", None, 1),
        ],
    },
]

BLOCKING_COMPOSITE_RULE_GROUPS = [
    {
        "rule_template": "rule__behavior_context__np_geoname_id__fs_{feature}",
        "family": BlockFamily.BEHAVIOR_CONTEXT,
        "features": ["source_site_365", "has_account", "has_click_365", "has_accept_365", "visited_30"],
        "spec_template": [
            (FeatureSource.NP, "geoname_id", "geo", None, 2),
            (FeatureSource.FS, "{feature}", "fs_value", None, 1),
        ],
    },
    {
        "rule_template": "rule__postman_context__np_geoname_id__fs_{feature}",
        "family": BlockFamily.POSTMAN_CONTEXT,
        "features": [
            "postman_action_90",
            "postman_campaign_90",
            "postman_response_90",
            "postman_action_30",
            "postman_campaign_30",
            "postman_response_30",
        ],
        "spec_template": [
            (FeatureSource.NP, "geoname_id", "geo", None, 2),
            (FeatureSource.FS, "{feature}", "fs_value", None, 1),
        ],
    },
    {
        "rule_template": "rule__behavior_context_device__np_subdivision__fs_{feature}__np_osfamily",
        "family": BlockFamily.BEHAVIOR_CONTEXT_DEVICE,
        "features": ["source_site_365", "has_click_365", "has_accept_365"],
        "spec_template": [
            (FeatureSource.NP, "subdivision_1_iso_code", "subdivision", None, 2),
            (FeatureSource.FS, "{feature}", "fs_value", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
        ],
    },
    {
        "rule_template": "rule__behavior_daypart__np_geoname_id__fs_{feature}__rt_daypart__rt_weekpart",
        "family": BlockFamily.BEHAVIOR_DAYPART,
        "features": ["source_site_365", "has_click_365", "has_accept_365"],
        "spec_template": [
            (FeatureSource.NP, "geoname_id", "geo", None, 2),
            (FeatureSource.FS, "{feature}", "fs_value", None, 1),
            (FeatureSource.RT, "local_hour", "daypart", TransformName.DAYPART_BUCKET, 8),
            (FeatureSource.RT, "day", "weekpart", TransformName.WEEKEND_BUCKET, 7),
        ],
    },
    {
        "rule_template": "rule__behavior_daypart_device__np_subdivision__fs_{feature}__np_osfamily__rt_daypart__rt_weekpart",
        "family": BlockFamily.BEHAVIOR_DAYPART_DEVICE,
        "features": ["source_site_365", "has_click_365"],
        "spec_template": [
            (FeatureSource.NP, "subdivision_1_iso_code", "subdivision", None, 2),
            (FeatureSource.FS, "{feature}", "fs_value", None, 1),
            (FeatureSource.NP, "osfamily", "osfamily", None, 1),
            (FeatureSource.RT, "local_hour", "daypart", TransformName.DAYPART_BUCKET, 8),
            (FeatureSource.RT, "day", "weekpart", TransformName.WEEKEND_BUCKET, 7),
        ],
    },
    {
        "rule_template": "rule__registration_time_window__np_geoname_id__fs_{feature}__registration_60m",
        "family": BlockFamily.REGISTRATION_TIME_WINDOW,
        "features": ["source_site_365", "has_click_365", "has_accept_365"],
        "spec_template": [
            (FeatureSource.NP, "geoname_id", "geo", None, 2),
            (FeatureSource.FS, "{feature}", "fs_value", None, 1),
            (FeatureSource.DERIVED, "registration_60m_bucket", "reg_60m", None, 16),
        ],
    },
]

GEO_KEYS = {"np__geoname_id", "np__subdivision_1_iso_code", "rt__geoid", "rt__geoname"}
STRONG_FAMILIES = {
    BlockFamily.BEHAVIOR_CONTEXT,
    BlockFamily.BEHAVIOR_CONTEXT_DEVICE,
    BlockFamily.CLUSTER_CANDIDATE,
    BlockFamily.POSTMAN_CONTEXT,
    BlockFamily.BEHAVIOR_DAYPART,
    BlockFamily.BEHAVIOR_DAYPART_DEVICE,
    BlockFamily.REGISTRATION_TIME_WINDOW,
    BlockFamily.IDENTITY_RESCUE,
}
WEAK_FAMILIES = {BlockFamily.CONTEXT, BlockFamily.COVERAGE_COMPOUND, BlockFamily.COVERAGE_FALLBACK}
TIME_AWARE_FAMILIES = {
    BlockFamily.BEHAVIOR_DAYPART,
    BlockFamily.BEHAVIOR_DAYPART_DEVICE,
    BlockFamily.REGISTRATION_TIME_WINDOW,
}
REGISTRATION_TIME_FAMILY = BlockFamily.REGISTRATION_TIME_WINDOW
BEHAVIOR_EVIDENCE_FAMILIES = {
    BlockFamily.BEHAVIOR,
    BlockFamily.BEHAVIOR_CONTEXT,
    BlockFamily.BEHAVIOR_CONTEXT_DEVICE,
    BlockFamily.CLUSTER_CANDIDATE,
    BlockFamily.POSTMAN_CONTEXT,
    BlockFamily.BEHAVIOR_DAYPART,
    BlockFamily.BEHAVIOR_DAYPART_DEVICE,
}

# Признаки, на которых обучается модель ребра графа.
#
# Они описывают пару профилей, а не один профиль:
# - evidence из blocking: сколько правил нашло пару и насколько узкими были блоки;
# - семейства правил: какой тип сигнала породил пару;
# - similarity: сколько общих fs/geo/identity значений у двух профилей;
# - временные сигналы: нашли ли пару правила с частью суток, выходным или окном регистрации.
MODEL_FEATURES = [
    "n_block_rules",
    "n_block_families",
    "min_block_size",
    "n_small_blocks_le2",
    "n_small_blocks_le5",
    "n_small_blocks_le10",
    "has_small_block_le2",
    "has_small_block_le5",
    "has_small_block_le10",
    "small_block_share_le5",
    "hit_coverage_fallback",
    "is_fallback_only",
    "has_non_fallback_signal",
    "has_strong_family",
    "n_strong_family_hits",
    "n_weak_family_hits",
    "strong_family_hit_share",
    "has_behavior",
    "has_behavior_context",
    "has_postman_context",
    "postman_context_hit_share",
    "has_time_aware_device_signal",
    "n_time_aware_hits",
    "time_aware_hit_share",
    "has_registration_time_window",
    "n_registration_time_window_hits",
    "registration_time_window_hit_share",
    "registration_time_window_only",
    "registration_time_window_with_behavior",
    "has_context",
    "has_coverage_compound",
    "only_weak_families",
    "small_block_weak_family_only",
    "fs_total_jaccard",
    "geo_total_jaccard",
    "fs_shared_count",
    "identity_email_match",
    "identity_phone_match",
    "identity_strong_match",
]

MODEL_FEATURE_DESCRIPTIONS = {
    "n_block_rules": "Количество разных правил блокинга, которые нашли эту пару профилей. Если значение 1, пара появилась только по одному правилу. Если значение больше 1, несколько независимых правил указывают на одну и ту же пару, и это обычно делает связь убедительнее.",
    "n_block_families": "Количество разных семейств правил, которые нашли пару. Семейство - это тип сигнала: география, поведение, время регистрации, телефонный rescue и т.п. Этот признак показывает не просто число правил, а разнообразие источников evidence.",
    "min_block_size": "Минимальный размер блока среди всех блоков, через которые была найдена пара. Размер блока - это число профилей с одинаковым ключом по конкретному правилу. Маленький блок обычно более точный: совпасть в группе из 2-5 профилей сильнее, чем в группе из сотен профилей.",
    "n_small_blocks_le2": "Сколько сработавших блоков для этой пары имели размер ровно до 2 профилей. Это самый узкий тип совпадения: правило фактически соединило только эту пару.",
    "n_small_blocks_le5": "Сколько сработавших блоков для этой пары имели размер не больше 5 профилей. Такие блоки считаются узкими и обычно дают более надежный сигнал, чем широкие блоки.",
    "n_small_blocks_le10": "Сколько сработавших блоков для этой пары имели размер не больше 10 профилей. Это более мягкая версия признака узкого блока: сигнал все еще относительно специфичный, но уже менее точный, чем блок 2-5.",
    "has_small_block_le2": "Флаг 0/1: была ли пара найдена хотя бы через один блок размера не больше 2 профилей. Удобная бинарная версия признака n_small_blocks_le2.",
    "has_small_block_le5": "Флаг 0/1: была ли пара найдена хотя бы через один блок размера не больше 5 профилей. Показывает наличие хотя бы одного очень узкого правила.",
    "has_small_block_le10": "Флаг 0/1: была ли пара найдена хотя бы через один блок размера не больше 10 профилей. Показывает наличие хотя бы одного относительно узкого правила.",
    "small_block_share_le5": "Доля срабатываний пары, пришедшая из блоков размера не больше 5. Формула: n_small_blocks_le5 / n_block_hits. Высокое значение означает, что пару чаще подтверждали узкие, а не широкие блоки.",
    "hit_coverage_fallback": "Флаг 0/1: пару нашло техническое fallback-правило покрытия. Такие правила нужны, чтобы не потерять профили без сильных признаков, но сами по себе они слабые и могут создавать шум.",
    "is_fallback_only": "Флаг 0/1: пара найдена только fallback-правилом и больше ничем. Это рискованный случай: пара появилась из технического покрытия, без содержательного подтверждения.",
    "has_non_fallback_signal": "Флаг 0/1: кроме fallback есть хотя бы одно обычное правило блокинга. Если значение 1, пара имеет не только техническое, но и содержательное основание.",
    "has_strong_family": "Флаг 0/1: среди сработавших правил есть сильное семейство: behavior_context, behavior_context_device, cluster_candidate, postman_context, time-aware, registration_time_window или identity_rescue. Это грубый индикатор наличия более специфичного evidence.",
    "n_strong_family_hits": "Количество срабатываний, пришедших от сильных семейств правил. Чем выше значение, тем больше специфичных сигналов поддерживают пару.",
    "n_weak_family_hits": "Количество срабатываний, пришедших от слабых семейств: context, coverage_compound или coverage_fallback. Эти правила важны для recall, но по отдельности часто недостаточно точны.",
    "strong_family_hit_share": "Доля срабатываний сильных семейств среди всех срабатываний пары. Формула: n_strong_family_hits / n_block_hits. Помогает отличить пару, подтвержденную сильными сигналами, от пары, найденной в основном широкими coverage/context правилами.",
    "has_behavior": "Флаг 0/1: сработало атомарное поведенческое fs-правило без дополнительного контекста. Например, совпал source_site_365 или has_click_365. Само по себе такое совпадение может быть широким.",
    "has_behavior_context": "Флаг 0/1: сработало поведенческое правило с контекстом, например география + site-id или география + факт клика/акцепта. Такое правило обычно точнее атомарного behavior, потому что дополнительно сужает область поиска.",
    "has_postman_context": "Флаг 0/1: сработало правило вида география + postman-признак. Postman-признаки описывают коммуникационное поведение; используем их только в композите с географией, чтобы не создавать слишком широкие и шумные блоки.",
    "postman_context_hit_share": "Доля postman-context срабатываний среди всех срабатываний пары. Формула: n_postman_context_hits / n_block_hits. Важно не только наличие postman, но и его вес относительно остальных правил.",
    "has_time_aware_device_signal": "Флаг 0/1: сработало time-aware правило, дополненное device/os контекстом. Это более узкий временной сигнал: совпадает не только время, но и устройство или операционная система.",
    "n_time_aware_hits": "Количество time-aware правил, которые нашли пару. Показывает, насколько сильно пара поддержана временными композитами.",
    "time_aware_hit_share": "Доля time-aware срабатываний среди всех срабатываний пары. Формула: n_time_aware_hits / n_block_hits. Высокое значение означает, что решение сильно опирается на временной сигнал.",
    "has_context": "Флаг 0/1: сработало контекстное гео-правило, например geoname, geoname_id или subdivision. Это полезный recall-сигнал, но сам по себе часто слабый, потому что много разных людей могут быть в одной географии.",
    "has_coverage_compound": "Флаг 0/1: сработало составное coverage-правило, например география + устройство + OS family. Такие правила нужны для покрытия и сужения поиска, особенно когда нет сильных identity-признаков.",
    "only_weak_families": "Флаг 0/1: пару нашли только слабые семейства правил. Если значение 1, у пары нет strong-family evidence, поэтому модель должна быть осторожнее.",
    "fs_total_jaccard": "Jaccard similarity по fs-значениям пары. Формула: |общие fs-значения| / |все уникальные fs-значения двух профилей|. Значение 0 означает, что общих fs-значений нет; 1 означает полное совпадение множеств fs-значений.",
    "geo_total_jaccard": "Jaccard similarity по гео-значениям пары. Формула: |общие гео-значения| / |все уникальные гео-значения двух профилей|. Учитывает гео-ключи из np/rt, например geoname_id, geoname, subdivision, geoid.",
    "fs_shared_count": "Абсолютное число общих fs-значений между двумя профилями. В отличие от fs_total_jaccard, это не доля, а количество совпавших поведенческих значений.",
    "identity_email_match": "Флаг 0/1: у двух профилей совпал нормализованный email. В production это сильный признак, но в нашем датасете email часто разрежен, поэтому срабатывает редко.",
    "identity_phone_match": "Флаг 0/1: у двух профилей совпал нормализованный телефон. Телефон используется как high-precision rescue: покрытие маленькое, но совпадение обычно очень сильное.",
    "identity_strong_match": "Флаг 0/1: совпал email или телефон. Объединяет два самых сильных identity-сигнала в один простой признак.",
    "has_registration_time_window": "Флаг 0/1: пару нашло хотя бы одно правило окна регистрации. Окно регистрации строится через перекрывающиеся 30-минутные bucket-ключи, чтобы покрывать регистрации с разницей примерно до часа.",
    "n_registration_time_window_hits": "Количество правил окна регистрации, которые нашли пару. Чем выше значение, тем сильнее пара поддержана гипотезой близкой регистрации.",
    "registration_time_window_hit_share": "Доля срабатываний окна регистрации среди всех срабатываний пары. Формула: n_registration_time_window_hits / n_block_hits. Показывает, насколько решение зависит именно от регистрационного времени.",
    "registration_time_window_only": "Флаг 0/1: пару нашли только правила окна регистрации, без поддержки других семейств. Это рискованный случай, потому что близкое время регистрации само по себе не доказывает дубль.",
    "registration_time_window_with_behavior": "Флаг 0/1: окно регистрации поддержано поведенческим сигналом. То есть помимо близкого времени есть еще behavior/postman/behavior_context evidence.",
    "small_block_weak_family_only": "Флаг 0/1: пара найдена через маленький блок, но только слабыми семействами правил. Маленький блок полезен, но если он пришел только из слабых правил, это не всегда надежное совпадение.",
}

def rule_family_from_name(rule: str) -> str:
    parts = str(rule).split("__")
    return parts[1] if len(parts) > 2 and parts[0] == "rule" else "unknown"

