from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CATBOOST_MODEL_PATH = ARTIFACTS_DIR / "catboost_pair_model.cbm"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_cols.json"

DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_MAX_ROWS = 5_000

ID_COL = "id"
NAME_COL = "name"

RANDOM_STATE = 42