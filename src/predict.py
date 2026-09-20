"""
predict.py — Batch score, segment by risk, write results to MySQL.
Uses dynamic CLIENTNUM / id_col from SCHEMA.
"""
"""
predict.py — Batch score, segment by risk, write results to MySQL.
Uses explicit CREATE TABLE with primary key for Aiven compatibility.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DB_URL, SCHEMA
import logging

log = logging.getLogger(__name__)
ID_COL = "CLIENTNUM"


def assign_risk(p: float) -> str:
    if p >= 0.70: return "High"
    if p >= 0.40: return "Medium"
    return "Low"


def batch_predict(best_name, best_model, X_test, y_test, client_ids) -> pd.DataFrame:
    yp = best_model.predict_proba(X_test)[:, 1]
    yh = best_model.predict(X_test)
    df = pd.DataFrame({
        ID_COL:            [str(c) for c in client_ids],
        "churn_actual":    y_test.values.tolist(),
        "churn_proba":     [round(float(p), 4) for p in yp],
        "churn_predicted": yh.tolist(),
        "risk_segment":    [assign_risk(p) for p in yp],
        "model_name":      best_name,
    })
    log.info(f"  Risk segments: {df['risk_segment'].value_counts().to_dict()}")
    return df


def write_predictions(df: pd.DataFrame):
    """Write predictions with explicit primary key for Aiven."""
    engine = create_engine(DB_URL, echo=False)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS churn_predictions"))
        conn.execute(text("""
            CREATE TABLE churn_predictions (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                CLIENTNUM       VARCHAR(100),
                churn_actual    TINYINT,
                churn_proba     DECIMAL(6,4),
                churn_predicted TINYINT,
                risk_segment    VARCHAR(10),
                model_name      VARCHAR(40),
                scored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_seg   (risk_segment),
                INDEX idx_proba (churn_proba)
            )
        """))

    # Insert rows
    rows = df.to_dict(orient="records")
    with engine.begin() as conn:
        for chunk_start in range(0, len(rows), 500):
            chunk = rows[chunk_start:chunk_start + 500]
            conn.execute(
                text("""
                    INSERT INTO churn_predictions
                    (CLIENTNUM, churn_actual, churn_proba, churn_predicted, risk_segment, model_name)
                    VALUES (:CLIENTNUM, :churn_actual, :churn_proba, :churn_predicted, :risk_segment, :model_name)
                """),
                chunk
            )
    log.info(f"  Written {len(df):,} predictions ✓")


def write_shap(shap_df: pd.DataFrame):
    """Write SHAP explanations with primary key for Aiven."""
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
        for chunk_start in range(0, len(rows), 1000):
            chunk = rows[chunk_start:chunk_start + 1000]
            conn.execute(
                text("""
                    INSERT INTO shap_explanations
                    (CLIENTNUM, feature_name, shap_value, feature_value, rank_order)
                    VALUES (:CLIENTNUM, :feature_name, :shap_value, :feature_value, :rank_order)
                """),
                chunk
            )
    log.info(f"  Written {len(shap_df):,} SHAP rows ✓")


def run(best_name, best_model, X_test, y_test, client_ids, shap_df):
    log.info("Writing predictions to MySQL...")
    pred_df = batch_predict(best_name, best_model, X_test, y_test, client_ids)
    write_predictions(pred_df)
    write_shap(shap_df)
    return pred_df
