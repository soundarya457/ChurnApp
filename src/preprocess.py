"""
preprocess.py — Leak-free preprocessing.

Order of operations (this order is what prevents leakage):
  1. Load raw rows in upload order and keep `_source_row` + customer ID.
  2. Stratified train / test split on source rows  ← BEFORE anything is learned
  3. Optional stratified down-sample of the TRAIN part only
  4. Median imputation fitted on TRAIN only
  5. SMOTE fitted on TRAIN only (test rows are never resampled)
  6. Verify train and test share zero source rows (raises if they do)
Every row is later scored with a data_split tag: train / test / unused.
"""
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.config import DB_URL, SCHEMA, TEST_SIZE, RANDOM_STATE, SOURCE_COL, OUTPUT_DIR
from src.ingest import to_binary
from sqlalchemy import text
import logging

log = logging.getLogger(__name__)
TRAIN_SAMPLE = 15000


def load_raw():
    engine = create_engine(DB_URL, echo=False)
    cols = pd.read_sql("SELECT * FROM raw_data LIMIT 0", con=engine).columns
    if SOURCE_COL not in cols:
        raise RuntimeError(f"raw_data has no '{SOURCE_COL}' column — re-run ingestion.")
    df = pd.read_sql(f"SELECT * FROM raw_data ORDER BY `{SOURCE_COL}`", con=engine)
    log.info(f"  Loaded {len(df):,} rows")
    return df


def engineer_features(df):
    """Row-wise transforms only (no statistics learned from other rows)."""
    id_col     = SCHEMA["id_col"]
    target_col = SCHEMA["target_col"]
    pos_label  = SCHEMA["target_positive"]
    cat_cols   = SCHEMA["categorical_cols"]
    num_cols   = SCHEMA["numerical_cols"]

    keep = ([id_col] if id_col else []) + [SOURCE_COL, target_col] + cat_cols + num_cols
    df   = df[[c for c in dict.fromkeys(keep) if c in df.columns]].copy()

    # Target → binary (exact label match; blanks were rejected at confirmation)
    df["churn"] = to_binary(df[target_col], pos_label)
    df.drop(columns=[target_col], inplace=True)
    log.info(f"  Churn rate: {df['churn'].mean():.1%} ({df['churn'].sum():,} of {len(df):,})")

    # Category → one-hot columns (no arbitrary ordering). Only the *set of category
    # names* is read from all rows; no target information is used.
    cat_present = [c for c in cat_cols if c in df.columns]
    for col in cat_present:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    if cat_present:
        df = pd.get_dummies(df, columns=cat_present, prefix_sep="=", dtype=float)

    # Numeric coercion only — imputation happens after the split (train medians)
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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


# ── Leakage verification ───────────────────────────────────────

def verify_split(train_src, test_src, X_train, X_test, n_total):
    """Hard checks. Raises RuntimeError on any overlap. Returns a report dict."""
    train_set, test_set = set(map(int, train_src)), set(map(int, test_src))
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(overlap)} source rows appear in both train and test "
            f"(e.g. {sorted(overlap)[:5]})")
    if len(train_set) != len(train_src) or len(test_set) != len(test_src):
        raise RuntimeError("DATA LEAKAGE: duplicate source rows inside a split")

    # Content-level check: identical feature vectors on both sides
    h_tr = set(pd.util.hash_pandas_object(X_train, index=False).tolist())
    h_te = pd.util.hash_pandas_object(X_test, index=False)
    identical = int(h_te.isin(h_tr).sum())
    if identical:
        log.warning(f"  {identical} test rows have a feature vector identical to a train row "
                    "(different customers, same values) — usually harmless for continuous data")

    return {
        "total_rows": int(n_total),
        "train_rows_used": len(train_set),
        "test_rows": len(test_set),
        "source_row_overlap": 0,
        "identical_feature_rows_across_splits": identical,
        "passed": True,
    }


def smote_train_only(X_train, y_train):
    """SMOTE on the training split only; confirms the original rows are untouched."""
    n_min = int(y_train.value_counts().min())
    if n_min < 2:
        raise ValueError("Too few minority rows in the training split for SMOTE.")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, n_min - 1))
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res = np.asarray(X_res, dtype=float).copy()
    n = len(X_train)
    # imblearn keeps the originals first and appends synthetic rows
    if not np.array_equal(X_res[:n], X_train.to_numpy()):
        raise RuntimeError("SMOTE altered original training rows — unexpected ordering")
    # One-hot / yes-no columns must stay 0 or 1 in the synthetic rows
    is_flag = np.array([set(np.unique(X_train[c])) <= {0.0, 1.0} for c in X_train.columns])
    X_res[n:, is_flag] = np.round(X_res[n:, is_flag])
    return X_res, y_res, len(X_res) - n


def run():
    df = engineer_features(load_raw())
    id_col = SCHEMA["id_col"]

    if df[SOURCE_COL].duplicated().any():
        raise RuntimeError("Duplicate _source_row values in raw_data.")
    ids_all = df[id_col].astype(str).reset_index(drop=True) if id_col else df[SOURCE_COL].astype(str)
    if ids_all.duplicated().any():
        raise RuntimeError(f"Duplicate customer IDs in '{id_col}' — cannot match predictions by ID.")

    save_features(df)   # features table keeps churn + _source_row for the dashboard

    drop = {"churn", SOURCE_COL, "_row_id"} | ({id_col} if id_col else set())
    feature_cols = [c for c in df.columns if c not in drop]
    X_all = df[feature_cols].astype(float)
    X_all.index = df[SOURCE_COL].to_numpy()          # index == source row id, always
    y_all = pd.Series(df["churn"].to_numpy(dtype=int), index=X_all.index, name="churn")
    ids_all.index = X_all.index
    feature_names = X_all.columns.tolist()

    # ── 1. split FIRST ──
    train_src, test_src = train_test_split(
        X_all.index.to_numpy(), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_all)

    # ── 2. cap the TRAIN side only ──
    if len(train_src) > TRAIN_SAMPLE:
        fit_src, _ = train_test_split(
            train_src, train_size=TRAIN_SAMPLE, random_state=RANDOM_STATE,
            stratify=y_all.loc[train_src])
    else:
        fit_src = train_src
    fit_src = np.sort(fit_src); test_src = np.sort(test_src)

    # ── 3. impute with TRAIN medians, applied to every row ──
    medians = X_all.loc[fit_src].median().fillna(0)
    X_all = X_all.fillna(medians)

    X_train_raw, y_train_raw = X_all.loc[fit_src], y_all.loc[fit_src]
    X_test,      y_test      = X_all.loc[test_src], y_all.loc[test_src]

    # ── 4. verify BEFORE resampling ──
    report = verify_split(fit_src, test_src, X_train_raw, X_test, len(X_all))

    # ── 5. SMOTE on train only ──
    X_train, y_train, n_syn = smote_train_only(X_train_raw, y_train_raw)
    X_train = pd.DataFrame(X_train, columns=feature_names)
    y_train = pd.Series(y_train, name="churn")
    report.update(
        train_class_counts_before_smote={str(k): int(v) for k, v in y_train_raw.value_counts().items()},
        train_class_counts_after_smote={str(k): int(v) for k, v in y_train.value_counts().items()},
        synthetic_rows_added=int(n_syn),
        test_class_counts={str(k): int(v) for k, v in y_test.value_counts().items()},
        smote_applied_to="training split only",
    )
    (OUTPUT_DIR / "split_report.json").write_text(json.dumps(report, indent=2))

    split_tag = pd.Series("unused", index=X_all.index)
    split_tag.loc[fit_src] = "train"
    split_tag.loc[test_src] = "test"

    log.info(f"  Split verified: train={len(fit_src):,} test={len(test_src):,} overlap=0 "
             f"| SMOTE added {n_syn:,} synthetic train rows | test untouched")
    log.info(f"  Predict ALL: {len(X_all):,}")

    return {
        "X_train": X_train, "y_train": y_train,                    # after SMOTE (fit only)
        "X_train_raw": X_train_raw, "y_train_raw": y_train_raw,    # before SMOTE (for CV)
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
        "X_full": X_all, "y_full": y_all,
        "ids_full": ids_all, "ids_test": ids_all.loc[test_src].to_numpy(),
        "split_full": split_tag,
        "split_report": report,
    }
