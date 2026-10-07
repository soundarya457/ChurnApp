"""
predict.py — Score ALL rows and store them by customer ID.

Customer IDs travel with the data from the upload (they are the index/columns
of the objects passed in), so nothing is re-read from the database by position
and nothing is truncated: any length or ID mismatch raises instead of silently
trimming.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DB_URL, SCHEMA
import logging

log = logging.getLogger(__name__)
ID_COL = "CLIENTNUM"   # DB column name for the customer ID (value comes from SCHEMA['id_col'])


def assign_risk(proba, actual):
    """Already churned → 'Churned'. Others → risk by probability."""
    if int(actual) == 1:
        return "Churned"
    if proba >= 0.70: return "High"
    if proba >= 0.40: return "Medium"
    return "Low"


def batch_predict_all(best_name, best_model, data):
    X_full, y_full = data["X_full"], data["y_full"]
    ids, split = data["ids_full"], data["split_full"]
    n = len(X_full)
    log.info(f"  Predicting all {n:,} records...")

    if not (len(y_full) == len(ids) == len(split) == n):
        raise ValueError(f"Length mismatch: X={n} y={len(y_full)} ids={len(ids)} split={len(split)}")
    if not (X_full.index.equals(y_full.index) and X_full.index.equals(ids.index)
            and X_full.index.equals(split.index)):
        raise ValueError("Row identifiers of X / y / IDs / split are not aligned.")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Customer IDs must be present and unique before storing predictions.")

    probas, preds = [], []
    for start in range(0, n, 500):
        chunk = X_full.iloc[start:start+500]
        probas.append(best_model.predict_proba(chunk)[:, 1])
        preds.append(best_model.predict(chunk))
    probas, preds = np.concatenate(probas), np.concatenate(preds)
    if len(probas) != n:
        raise ValueError("Prediction count differs from input row count.")

    y_vals = y_full.to_numpy()
    df = pd.DataFrame({
        ID_COL:            ids.to_numpy(),
        "source_row":      X_full.index.to_numpy().astype(np.int64),
        "data_split":      split.to_numpy(),
        "churn_actual":    y_vals.astype(int),
        "churn_proba":     np.round(probas.astype(float), 4),
        "churn_predicted": preds.astype(int),
        "risk_segment":    [assign_risk(p, v) for p, v in zip(probas, y_vals)],
        "model_name":      best_name,
    })
    log.info(f"  Total predicted: {len(df):,} | Segments: {df['risk_segment'].value_counts().to_dict()}")
    return df


def write_predictions(df):
    engine = create_engine(DB_URL, echo=False)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS churn_predictions"))
        conn.execute(text("""
            CREATE TABLE churn_predictions (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                CLIENTNUM       VARCHAR(100) NOT NULL,
                source_row      BIGINT,
                data_split      VARCHAR(10),
                churn_actual    TINYINT,
                churn_proba     DECIMAL(6,4),
                churn_predicted TINYINT,
                risk_segment    VARCHAR(10),
                model_name      VARCHAR(40),
                scored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_cid (CLIENTNUM),
                INDEX idx_seg   (risk_segment),
                INDEX idx_proba (churn_proba),
                INDEX idx_split (data_split)
            )
        """))
    rows = df.to_dict(orient="records")
    with engine.begin() as conn:
        for start in range(0, len(rows), 500):
            conn.execute(text("""
                INSERT INTO churn_predictions
                (CLIENTNUM,source_row,data_split,churn_actual,churn_proba,churn_predicted,risk_segment,model_name)
                VALUES
                (:CLIENTNUM,:source_row,:data_split,:churn_actual,:churn_proba,:churn_predicted,:risk_segment,:model_name)
            """), rows[start:start+500])
    log.info(f"  Written {len(df):,} predictions ✓")


def write_shap(shap_df):
    engine = create_engine(DB_URL, echo=False)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS shap_explanations"))
        conn.execute(text("""
            CREATE TABLE shap_explanations (
                id            INT AUTO_INCREMENT PRIMARY KEY,
                CLIENTNUM     VARCHAR(100),
                feature_name  VARCHAR(100),
                shap_value    DECIMAL(12,6),
                feature_value DECIMAL(15,4),
                rank_order    TINYINT,
                INDEX idx_client (CLIENTNUM)
            )
        """))
    rows = shap_df.to_dict(orient="records")
    with engine.begin() as conn:
        for start in range(0, len(rows), 1000):
            conn.execute(text("""
                INSERT INTO shap_explanations
                (CLIENTNUM,feature_name,shap_value,feature_value,rank_order)
                VALUES
                (:CLIENTNUM,:feature_name,:shap_value,:feature_value,:rank_order)
            """), rows[start:start+1000])
    log.info(f"  Written {len(shap_df):,} SHAP rows ✓")


def verify_stored(pred_df):
    """Re-read the DB and confirm every stored prediction matches a raw_data customer by ID."""
    engine = create_engine(DB_URL, echo=False)
    id_col = SCHEMA["id_col"]
    with engine.connect() as conn:
        n_pred = conn.execute(text("SELECT COUNT(*) FROM churn_predictions")).scalar()
        n_join = conn.execute(text(
            f"SELECT COUNT(*) FROM churn_predictions p JOIN raw_data r "
            f"ON CAST(r.`{id_col}` AS CHAR) = p.CLIENTNUM")).scalar()
        n_raw = conn.execute(text("SELECT COUNT(*) FROM raw_data")).scalar()
    if not (n_pred == n_join == n_raw == len(pred_df)):
        raise RuntimeError(
            f"ID alignment check FAILED: predictions={n_pred}, matched to raw_data by ID={n_join}, "
            f"raw rows={n_raw}, scored={len(pred_df)}")
    log.info(f"  ID alignment verified: {n_pred:,} predictions ↔ {n_raw:,} customers ✓")


def run(best_name, best_model, data, shap_df):
    log.info("Writing predictions...")
    pred_df = batch_predict_all(best_name, best_model, data)
    write_predictions(pred_df)
    write_shap(shap_df)
    verify_stored(pred_df)
    return pred_df
