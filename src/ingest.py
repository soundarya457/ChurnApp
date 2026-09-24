"""
ingest.py — Load any CSV → MySQL raw_data table.
Auto-detects schema with improved target column detection.
"""
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from src.config import DB_URL, SCHEMA
import logging

log = logging.getLogger(__name__)


# ── Schema detection ───────────────────────────────────────────

def detect_id_col(df: pd.DataFrame) -> str | None:
    """Return the column most likely to be a row identifier."""
    id_keywords = re.compile(
        r"(^id$|clientnum|customerid|cust.*id|account.*id|member.*id|rowid|row_id)",
        re.I
    )
    for col in df.columns:
        if id_keywords.search(col):
            if df[col].nunique() / len(df) > 0.90:
                return col
    # Integer column where every value is unique
    for col in df.select_dtypes(include="number").columns:
        if df[col].nunique() == len(df):
            return col
    return None


def detect_target_col(df: pd.DataFrame, id_col: str | None) -> tuple:
    """
    Return (target_col, positive_label).
    Priority:
      1. Columns with churn-related names
      2. Binary integer columns (0/1) — minority class = positive
      3. Binary string columns — minority class = positive
    Explicitly skips demographic columns (Gender, Geography etc.)
    """
    skip = {id_col} if id_col else set()

    # Columns that are almost certainly NOT the target
    demographic_pattern = re.compile(
        r"^(gender|sex|geography|country|region|state|city|name|surname|"
        r"firstname|lastname|email|phone|address|zip|postal)$",
        re.I
    )

    # Priority 1: column name strongly suggests churn/target
    churn_pattern = re.compile(
        r"(churn|attrition|exited|left|cancelled|canceled|churned|"
        r"target|label|class|outcome|default|fraud|converted)",
        re.I
    )

    # Check named churn columns first
    for col in df.columns:
        if col in skip or demographic_pattern.match(col):
            continue
        if churn_pattern.search(col):
            vals = df[col].dropna().unique()
            if len(vals) == 2:
                # For string binary: minority = positive
                counts = df[col].value_counts()
                positive = counts.index[-1]
                log.info(f"  Detected target (name match): '{col}' | positive='{positive}'")
                return col, str(positive)
            # Handle 0/1 numeric
            if set(map(int, vals)).issubset({0, 1}):
                log.info(f"  Detected binary target (name match): '{col}' | positive='1'")
                return col, "1"

    # Priority 2: binary integer 0/1 columns (skip demographics)
    for col in df.select_dtypes(include="number").columns:
        if col in skip or demographic_pattern.match(col):
            continue
        vals = set(df[col].dropna().unique())
        if vals.issubset({0, 1, 0.0, 1.0}):
            log.info(f"  Detected binary int target: '{col}' | positive='1'")
            return col, "1"

    # Priority 3: binary string columns (skip demographics)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col in skip or demographic_pattern.match(col):
            continue
        vals = df[col].dropna().unique()
        if len(vals) == 2:
            counts  = df[col].value_counts()
            positive = counts.index[-1]
            log.info(f"  Detected binary string target: '{col}' | positive='{positive}'")
            return col, str(positive)

    return None, None


def detect_col_types(df: pd.DataFrame, id_col: str | None, target_col: str | None):
    skip     = {c for c in [id_col, target_col] if c}
    cat_cols, num_cols, drop_cols = [], [], []

    for col in df.columns:
        if col in skip:
            continue
        n_unique = df[col].nunique()
        n_rows   = len(df)

        if n_unique == n_rows and df[col].dtype == "object":
            drop_cols.append(col)
            continue
        if n_unique <= 1:
            drop_cols.append(col)
            continue

        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            cat_cols.append(col)
        else:
            num_cols.append(col)

    return cat_cols, num_cols, drop_cols


def detect_schema(df: pd.DataFrame):
    id_col              = detect_id_col(df)
    target_col, pos_lbl = detect_target_col(df, id_col)
    cat_cols, num_cols, drop_cols = detect_col_types(df, id_col, target_col)

    SCHEMA["id_col"]           = id_col
    SCHEMA["target_col"]       = target_col
    SCHEMA["target_positive"]  = pos_lbl
    SCHEMA["categorical_cols"] = cat_cols
    SCHEMA["numerical_cols"]   = num_cols
    SCHEMA["drop_cols"]        = drop_cols

    log.info(f"  ID col      : {id_col}")
    log.info(f"  Target col  : {target_col}  (positive='{pos_lbl}')")
    log.info(f"  Categorical : {cat_cols}")
    log.info(f"  Numerical   : {num_cols}")
    log.info(f"  Drop        : {drop_cols}")


# ── MySQL helpers ──────────────────────────────────────────────

def _pandas_dtype_to_sql(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):   return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):     return "DOUBLE"
    if pd.api.types.is_bool_dtype(dtype):      return "TINYINT"
    return "VARCHAR(255)"


def create_raw_table(df: pd.DataFrame, engine):
    """Drop & recreate raw_data with primary key (required by Aiven)."""

    # Truncate column names > 64 chars
    rename_map = {}
    for col in df.columns:
        if len(col) > 64:
            short = col[:64]
            rename_map[col] = short
            log.warning(f"  Column truncated: '{col}' → '{short}'")
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    id_col = SCHEMA["id_col"]

    # Add synthetic PK if no ID col
    if not id_col or id_col not in df.columns:
        df.insert(0, "_row_id", range(1, len(df) + 1))
        SCHEMA["id_col"] = "_row_id"
        id_col = "_row_id"

    col_defs = []
    for col in df.columns:
        safe     = f"`{col}`"
        sql_type = _pandas_dtype_to_sql(df[col].dtype)
        if col == id_col:
            sql_type = "VARCHAR(100)" if sql_type == "VARCHAR(255)" else sql_type
            col_defs.append(f"  {safe} {sql_type} PRIMARY KEY")
        else:
            col_defs.append(f"  {safe} {sql_type}")

    ddl = ("DROP TABLE IF EXISTS raw_data;\n"
           "CREATE TABLE raw_data (\n" +
           ",\n".join(col_defs) + "\n);")

    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
    log.info("  raw_data table created ✓")


# ── Main ───────────────────────────────────────────────────────

def load_csv(csv_path) -> pd.DataFrame:
    log.info(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"  Shape: {df.shape}")
    return df


def validate(df: pd.DataFrame):
    nulls = df.isnull().sum()
    if nulls.any():
        log.warning(f"Nulls: {nulls[nulls > 0].to_dict()}")
    id_col = SCHEMA["id_col"]
    if id_col and id_col in df.columns:
        df[id_col] = df[id_col].astype(str)
        dupes = df.duplicated(subset=[id_col]).sum()
        if dupes:
            df.drop_duplicates(subset=[id_col], inplace=True)
            log.warning(f"  Dropped {dupes} duplicate rows")


def to_mysql(df: pd.DataFrame, engine):
    cols_to_drop = [c for c in SCHEMA["drop_cols"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw_data"))

    df.to_sql("raw_data", con=engine, if_exists="append",
              index=False, chunksize=500, method="multi")
    log.info(f"  Inserted {len(df):,} rows into raw_data ✓")


def run(csv_path=None):
    from src.config import DATA_DIR
    if csv_path is None:
        csvs = sorted(DATA_DIR.glob("*.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {DATA_DIR}")
        csv_path = csvs[0]

    df = load_csv(csv_path)

    # Drop columns with names > 64 chars (MySQL limit)
    long_cols = [c for c in df.columns if len(c) > 64]
    if long_cols:
        log.info(f"  Dropping {len(long_cols)} cols with names > 64 chars")
        df.drop(columns=long_cols, inplace=True)

    detect_schema(df)
    validate(df)
    engine = create_engine(DB_URL, echo=False)
    create_raw_table(df, engine)
    to_mysql(df, engine)
    log.info("Ingestion complete.")
    return df