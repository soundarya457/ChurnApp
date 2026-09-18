"""
config.py — Centralised settings. All column lists are detected at runtime.
"""
import os
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
SQL_DIR    = BASE_DIR / "sql"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"

for d in [DATA_DIR, SQL_DIR, OUTPUT_DIR, UPLOAD_DIR]:
    d.mkdir(exist_ok=True)

# ── MySQL ──────────────────────────────────────────────────────git init
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASS",     "newpassword123"),
    "database": os.getenv("DB_NAME",     "churn_db"),
}
DB_URL = (
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ── ML Settings ────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# ── Runtime schema (populated by ingest.py after CSV is read) ──
# These are set dynamically — do NOT hardcode column names here.
SCHEMA = {
    "target_col":        None,   # e.g. "Attrition_Flag"
    "target_positive":   None,   # e.g. "Attrited Customer"
    "id_col":            None,   # e.g. "CLIENTNUM"
    "categorical_cols":  [],
    "numerical_cols":    [],
    "drop_cols":         [],
}
