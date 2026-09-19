"""
preprocess.py — Dynamic feature engineering + SMOTE.
Works with any schema. Adds primary keys for Aiven MySQL compatibility.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from src.config import DB_URL, SCHEMA, TEST_SIZE, RANDOM_STATE
import logging

log = logging.getLogger(__name__)


def load_raw() -> pd.DataFrame:
    engine = create_engine(DB_URL, echo=False)
    df = pd.read_sql("SELECT * FROM raw_data", con=engine)
    log.info(f"  Loaded {len(df):,} rows from raw_data")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    id_col     = SCHEMA["id_col"]
    target_col = SCHEMA["target_col"]
    pos_label  = SCHEMA["target_positive"]
    cat_cols   = SCHEMA["categorical_cols"]
    num_cols   = SCHEMA["numerical_cols"]

    keep_cols = ([id_col] if id_col else []) + [target_col] + cat_cols + num_cols
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Target → 0/1
    df["churn"] = (df[target_col].astype(str) == str(pos_label)).astype(int)
    df.drop(columns=[target_col], inplace=True)
    log.info(f"  Churn rate: {df['churn'].mean():.1%}")

    # Encode categoricals
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Fill numeric nulls
    for col in num_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    # Final safety fill
    for col in df.columns:
        if col in ["churn"] + ([id_col] if id_col else []):
            continue
        if df[col].isnull().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(0)

    # Engineered features
    c = df.columns.tolist()
    if "Total_Trans_Amt" in c and "Total_Trans_Ct" in c:
        df["Trans_Amt_per_Ct"] = np.where(
            df["Total_Trans_Ct"] > 0,
            df["Total_Trans_Amt"] / df["Total_Trans_Ct"], 0)
    if "Total_Revolving_Bal" in c and "Credit_Limit" in c:
        df["Credit_Usage_Pct"] = np.where(
            df["Credit_Limit"] > 0,
            df["Total_Revolving_Bal"] / df["Credit_Limit"], 0)
    if "Months_Inactive_12_mon" in c and "Contacts_Count_12_mon" in c:
        df["Inactivity_Score"] = df["Months_Inactive_12_mon"] * df["Contacts_Count_12_mon"]
    if "Total_Relationship_Count" in c and "Total_Trans_Ct" in c:
        df["Engagement_Score"] = df["Total_Relationship_Count"] * df["Total_Trans_Ct"]

    return df


def save_features(df: pd.DataFrame):
    """Write features to MySQL with explicit primary key (required by Aiven)."""
    engine  = create_engine(DB_URL, echo=False)
    id_col  = SCHEMA.get("id_col")

    # Decide primary key column
    if id_col and id_col in df.columns:
        pk_col = id_col
    else:
        # Add a synthetic integer primary key
        df = df.copy()
        df.insert(0, "_row_id", range(1, len(df) + 1))
        pk_col = "_row_id"

    # Build column definitions
    def col_type(series, col_name):
        if col_name == pk_col:
            if pd.api.types.is_integer_dtype(series.dtype):
                return "BIGINT PRIMARY KEY"
            else:
                return "VARCHAR(100) PRIMARY KEY"
        if pd.api.types.is_integer_dtype(series.dtype):
            return "BIGINT"
        if pd.api.types.is_float_dtype(series.dtype):
            return "DOUBLE"
        return "VARCHAR(255)"

    col_defs = [f"`{col}` {col_type(df[col], col)}" for col in df.columns]
    ddl      = f"CREATE TABLE features ({', '.join(col_defs)})"

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features"))
        conn.execute(text(ddl))

    # Insert rows
    df.to_sql("features", con=engine, if_exists="append",
              index=False, chunksize=500, method="multi")
    log.info(f"  Written {len(df):,} rows to features ✓")


def split_and_balance(df: pd.DataFrame):
    id_col = SCHEMA["id_col"]
    drop   = ["churn"] + ([id_col] if id_col and id_col in df.columns else [])
    # Also drop synthetic key if added
    if "_row_id" in df.columns:
        drop.append("_row_id")
    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df["churn"]
    feature_names = X.columns.tolist()

    # Fill any leftover NaNs
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else "Unknown")

    # Drop rows with NaN target
    mask = y.notna()
    X    = X[mask]
    y    = y[mask]

    log.info(f"  Class distribution before SMOTE: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_b, y_train_b = smote.fit_resample(X_train, y_train)
    log.info(f"  After SMOTE: {pd.Series(y_train_b).value_counts().to_dict()}")
    log.info(f"  Train: {len(X_train_b):,} | Test: {len(X_test):,}")

    return X_train_b, X_test, y_train_b, y_test, feature_names


def run():
    log.info("Loading raw data and engineering features...")
    df = load_raw()
    df = engineer_features(df)
    save_features(df)
    return split_and_balance(df)