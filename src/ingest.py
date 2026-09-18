"""
ingest.py — Load any CSV → MySQL raw_data table.
Auto-detects:
  • ID column (high-cardinality unique integer col)
  • Target column (binary string col with 2 unique values)
  • Categorical vs numerical columns
  • Columns to drop (near-zero variance, all-unique text)
"""
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from src.config import DB_URL, SCHEMA
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Schema detection ───────────────────────────────────────────

def detect_id_col(df: pd.DataFrame) -> str | None:
    """Return the column most likely to be a row identifier."""
    # Priority 1: column name contains 'id', 'num', 'clientnum', 'customerid'
    id_keywords = re.compile(r"(^id$|clientnum|customerid|cust.*id|account.*id|member.*id)", re.I)
    for col in df.columns:
        if id_keywords.search(col):
            if df[col].nunique() / len(df) > 0.95:
                return col
    # Priority 2: integer column where every value is unique
    for col in df.select_dtypes(include="number").columns:
        if df[col].nunique() == len(df):
            return col
    return None


def detect_target_col(df: pd.DataFrame, id_col: str | None) -> tuple[str | None, str | None]:
    """
    Return (target_col, positive_label).
    Looks for a binary string/object column — the minority class is the positive label.
    """
    skip = {id_col} if id_col else set()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col in skip:
            continue
        vals = df[col].dropna().unique()
        if len(vals) == 2:
            # Minority class = positive (churn)
            counts = df[col].value_counts()
            positive = counts.index[-1]   # least frequent
            log.info(f"  Detected target: '{col}' | positive class: '{positive}'")
            return col, str(positive)
    # Fallback: binary integer column (0/1)
    for col in df.select_dtypes(include="number").columns:
        if col in skip:
            continue
        vals = df[col].dropna().unique()
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            log.info(f"  Detected binary target: '{col}'")
            return col, "1"
    return None, None


def detect_col_types(df: pd.DataFrame, id_col: str | None, target_col: str | None):
    """Split columns into categorical and numerical, skipping id and target."""
    skip = {c for c in [id_col, target_col] if c}
    cat_cols, num_cols, drop_cols = [], [], []

    for col in df.columns:
        if col in skip:
            continue
        n_unique = df[col].nunique()
        n_rows   = len(df)

        # Drop: all unique (likely free-text), zero-variance
        if n_unique == n_rows or n_unique <= 1:
            drop_cols.append(col)
            continue

        dtype = df[col].dtype
        if dtype == "object" or str(dtype) == "category":
            cat_cols.append(col)
        else:
            num_cols.append(col)

    return cat_cols, num_cols, drop_cols


def detect_schema(df: pd.DataFrame):
    """Run full auto-detection and populate SCHEMA dict."""
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
    """Drop & recreate raw_data table to match the CSV schema exactly."""
    
    # Truncate column names longer than 64 chars (MySQL limit)
    rename_map = {}
    for col in df.columns:
        if len(col) > 64:
            short = col[:64]
            rename_map[col] = short
            log.warning(f"  Column name truncated: '{col}' → '{short}'")
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    
    col_defs = []
    id_col   = SCHEMA["id_col"]
    for col in df.columns:
        safe     = f"`{col}`"
        sql_type = _pandas_dtype_to_sql(df[col].dtype)
        pk       = " PRIMARY KEY" if col == id_col else ""
        col_defs.append(f"  {safe} {sql_type}{pk}")

    ddl = "DROP TABLE IF EXISTS raw_data;\nCREATE TABLE raw_data (\n" + ",\n".join(col_defs) + "\n);"
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
        log.warning(f"Nulls found:\n{nulls[nulls > 0]}")
    id_col = SCHEMA["id_col"]
    if id_col and id_col in df.columns:
        df[id_col] = df[id_col].astype(str)
        dupes = df.duplicated(subset=[id_col]).sum()
        if dupes:
            df.drop_duplicates(subset=[id_col], inplace=True)
            log.warning(f"  Dropped {dupes} duplicate rows")


def to_mysql(df: pd.DataFrame, engine):
    # Drop schema-detected junk cols before inserting
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
        csvs = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {DATA_DIR}")
        csv_path = csvs[0]

    df = load_csv(csv_path)

    # Drop columns with names too long for MySQL (> 64 chars)
    long_cols = [c for c in df.columns if len(c) > 64]
    if long_cols:
        log.info(f"  Dropping {len(long_cols)} columns with names > 64 chars: {long_cols}")
        df.drop(columns=long_cols, inplace=True)

    detect_schema(df)
    validate(df)
    engine = create_engine(DB_URL, echo=False)
    create_raw_table(df, engine)
    to_mysql(df, engine)
    log.info("Ingestion complete.")
    return df