"""
preprocess.py — Train on sample, predict on ALL rows.
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
TRAIN_SAMPLE = 3000


def load_raw():
    engine = create_engine(DB_URL, echo=False)
    df     = pd.read_sql("SELECT * FROM raw_data", con=engine)
    log.info(f"  Loaded {len(df):,} rows")
    return df


def engineer_features(df):
    id_col     = SCHEMA["id_col"]
    target_col = SCHEMA["target_col"]
    pos_label  = SCHEMA["target_positive"]
    cat_cols   = SCHEMA["categorical_cols"]
    num_cols   = SCHEMA["numerical_cols"]

    keep = ([id_col] if id_col else []) + [target_col] + cat_cols + num_cols
    df   = df[[c for c in keep if c in df.columns]].copy()

    # Target → binary
    df["churn"] = (df[target_col].astype(str) == str(pos_label)).astype(int)
    df.drop(columns=[target_col], inplace=True)
    log.info(f"  Churn rate: {df['churn'].mean():.1%} ({df['churn'].sum():,} of {len(df):,})")

    for col in cat_cols:
        if col not in df.columns: continue
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    for col in num_cols:
        if col not in df.columns: continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        df[col] = df[col].fillna(med if not pd.isna(med) else 0)

    # Safety fill
    for col in df.columns:
        if col in (["churn"] + ([id_col] if id_col else [])): continue
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown" if df[col].dtype == "object" else 0)

    # Engineered features
    c = df.columns.tolist()
    if "Total_Trans_Amt" in c and "Total_Trans_Ct" in c:
        df["Trans_Amt_per_Ct"] = np.where(df["Total_Trans_Ct"]>0, df["Total_Trans_Amt"]/df["Total_Trans_Ct"], 0)
    if "Total_Revolving_Bal" in c and "Credit_Limit" in c:
        df["Credit_Usage_Pct"] = np.where(df["Credit_Limit"]>0, df["Total_Revolving_Bal"]/df["Credit_Limit"], 0)
    if "Months_Inactive_12_mon" in c and "Contacts_Count_12_mon" in c:
        df["Inactivity_Score"] = df["Months_Inactive_12_mon"] * df["Contacts_Count_12_mon"]
    if "Total_Relationship_Count" in c and "Total_Trans_Ct" in c:
        df["Engagement_Score"] = df["Total_Relationship_Count"] * df["Total_Trans_Ct"]

    return df


def save_features(df):
    engine = create_engine(DB_URL, echo=False)
    id_col = SCHEMA.get("id_col")
    if id_col and id_col in df.columns:
        pk_col = id_col
    else:
        df = df.copy()
        df.insert(0, "_row_id", range(1, len(df)+1))
        pk_col = "_row_id"

    def ctype(s, col):
        if col == pk_col:
            return "BIGINT PRIMARY KEY" if pd.api.types.is_integer_dtype(s.dtype) else "VARCHAR(100) PRIMARY KEY"
        if pd.api.types.is_integer_dtype(s.dtype): return "BIGINT"
        if pd.api.types.is_float_dtype(s.dtype):   return "DOUBLE"
        return "VARCHAR(255)"

    col_defs = [f"`{col}` {ctype(df[col], col)}" for col in df.columns]
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features"))
        conn.execute(text(f"CREATE TABLE features ({', '.join(col_defs)})"))
    df.to_sql("features", con=engine, if_exists="append",
              index=False, chunksize=500, method="multi")
    log.info(f"  Saved {len(df):,} rows to features ✓")


def run():
    df = load_raw()
    df = engineer_features(df)
    save_features(df)

    id_col = SCHEMA["id_col"]
    drop   = ["churn"] + ([id_col] if id_col and id_col in df.columns else [])
    if "_row_id" in df.columns: drop.append("_row_id")

    X_full = df.drop(columns=[c for c in drop if c in df.columns])
    y_full = df["churn"]
    feature_names = X_full.columns.tolist()

    # NaN fill
    for col in X_full.columns:
        if X_full[col].isnull().any():
            X_full[col] = X_full[col].fillna(
                X_full[col].median() if pd.api.types.is_numeric_dtype(X_full[col]) else "Unknown")

    mask   = y_full.notna()
    X_full = X_full[mask].reset_index(drop=True)
    y_full = y_full[mask].reset_index(drop=True)

    log.info(f"  Total rows for prediction: {len(X_full):,}")

    # Sample for training only
    if len(X_full) > TRAIN_SAMPLE:
        # Stratified sample to maintain churn ratio
        pos_idx = y_full[y_full==1].index.tolist()
        neg_idx = y_full[y_full==0].index.tolist()
        n_pos   = min(len(pos_idx), TRAIN_SAMPLE // 2)
        n_neg   = min(len(neg_idx), TRAIN_SAMPLE - n_pos)
        rng     = np.random.RandomState(RANDOM_STATE)
        sample_idx = (
            rng.choice(pos_idx, n_pos, replace=False).tolist() +
            rng.choice(neg_idx, n_neg, replace=False).tolist()
        )
        X_train = X_full.iloc[sample_idx].reset_index(drop=True)
        y_train = y_full.iloc[sample_idx].reset_index(drop=True)
    else:
        X_train = X_full.copy()
        y_train = y_full.copy()

    log.info(f"  Train sample: {len(X_train):,} | Churn dist: {y_train.value_counts().to_dict()}")

    # Test split from the FULL data (20% of all rows)
    _, X_test, _, y_test = train_test_split(
        X_full, y_full, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_full
    )

    # SMOTE
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_b, y_train_b = smote.fit_resample(X_train, y_train)
    log.info(f"  After SMOTE: {pd.Series(y_train_b).value_counts().to_dict()}")
    log.info(f"  Train: {len(X_train_b):,} | Test (eval): {len(X_test):,} | Predict ALL: {len(X_full):,}")

    return X_train_b, X_test, y_train_b, y_test, feature_names, X_full, y_full