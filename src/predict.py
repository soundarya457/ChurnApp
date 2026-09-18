"""
predict.py — Batch score, segment by risk, write results to MySQL.
Uses dynamic CLIENTNUM / id_col from SCHEMA.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DB_URL, SCHEMA
import logging

log = logging.getLogger(__name__)
ID_COL = "CLIENTNUM"   # output column name — always use this for consistency


def assign_risk(p: float) -> str:
    if p >= 0.70: return "High"
    if p >= 0.40: return "Medium"
    return "Low"


def batch_predict(best_name, best_model, X_test, y_test, client_ids) -> pd.DataFrame:
    yp = best_model.predict_proba(X_test)[:, 1]
    yh = best_model.predict(X_test)
    df = pd.DataFrame({
        ID_COL:           client_ids,
        "churn_actual":   y_test.values,
        "churn_proba":    np.round(yp, 4),
        "churn_predicted":yh,
        "risk_segment":   [assign_risk(p) for p in yp],
        "model_name":     best_name,
    })
    log.info(f"  Risk segments: {df['risk_segment'].value_counts().to_dict()}")
    return df


def write_predictions(df: pd.DataFrame):
    engine = create_engine(DB_URL, echo=False)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS churn_predictions"))
    df.to_sql("churn_predictions", con=engine, if_exists="replace",
              index=False, chunksize=500, method="multi")
    log.info(f"  Written {len(df):,} predictions ✓")


def write_shap(shap_df: pd.DataFrame):
    engine = create_engine(DB_URL, echo=False)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS shap_explanations"))
    shap_df.to_sql("shap_explanations", con=engine, if_exists="replace",
                   index=False, chunksize=1000, method="multi")
    log.info(f"  Written {len(shap_df):,} SHAP rows ✓")


def run(best_name, best_model, X_test, y_test, client_ids, shap_df):
    log.info("Writing predictions to MySQL...")
    pred_df = batch_predict(best_name, best_model, X_test, y_test, client_ids)
    write_predictions(pred_df)
    write_shap(shap_df)
    return pred_df
