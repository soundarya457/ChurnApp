"""
preprocess.py — Dynamic feature engineering + SMOTE.
Works entirely from SCHEMA dict populated by ingest.py.
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
    """
    1. Encode target → binary 0/1
    2. Ordinal-encode categoricals
    3. Fill nulls
    4. Add generic engineered features when standard columns detected
    5. Write features table to MySQL
    """
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

    # Fill numeric nulls with median
            # Fill numeric nulls with median
    for col in num_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    # Final safety — fill any remaining NaNs in the whole dataframe
    for col in df.columns:
        if col in ["churn"] + ([id_col] if id_col else []):
            continue
        if df[col].isnull().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(0)

    # ── Opportunistic feature engineering ──────────────────────
    # These fire only when the expected column names are present.
    c = df.columns.tolist()

    if "Total_Trans_Amt" in c and "Total_Trans_Ct" in c:
        df["Trans_Amt_per_Ct"] = np.where(
            df["Total_Trans_Ct"] > 0,
            df["Total_Trans_Amt"] / df["Total_Trans_Ct"], 0
        )

    if "Total_Revolving_Bal" in c and "Credit_Limit" in c:
        df["Credit_Usage_Pct"] = np.where(
            df["Credit_Limit"] > 0,
            df["Total_Revolving_Bal"] / df["Credit_Limit"], 0
        )

    if "Months_Inactive_12_mon" in c and "Contacts_Count_12_mon" in c:
        df["Inactivity_Score"] = df["Months_Inactive_12_mon"] * df["Contacts_Count_12_mon"]

    if "Total_Relationship_Count" in c and "Total_Trans_Ct" in c:
        df["Engagement_Score"] = df["Total_Relationship_Count"] * df["Total_Trans_Ct"]

    log.info(f"  Feature columns: {[c for c in df.columns if c not in [id_col, 'churn']]}")
    return df


def save_features(df: pd.DataFrame):
    """Write engineered features back to MySQL."""
    engine = create_engine(DB_URL, echo=False)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features"))
    df.to_sql("features", con=engine, if_exists="replace",
              index=False, chunksize=500, method="multi")
    log.info(f"  Written {len(df):,} rows to features table ✓")


def split_and_balance(df: pd.DataFrame):
    id_col = SCHEMA["id_col"]
    drop   = ["churn"] + ([id_col] if id_col and id_col in df.columns else [])
    X      = df.drop(columns=drop)
    y      = df["churn"]
    feature_names = X.columns.tolist()

    # Fill any remaining NaN values before SMOTE
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == "object":
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
            else:
                X[col] = X[col].fillna(X[col].median())

    # Drop any rows where y is NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]

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
