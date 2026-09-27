"""
predict.py — Predict for ALL rows. Label confirmed churners separately.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DB_URL, SCHEMA
import logging

log = logging.getLogger(__name__)
ID_COL = "CLIENTNUM"


def assign_risk(proba, actual):
    """Already churned → 'Churned'. Others → risk by probability."""
    if int(actual) == 1:
        return "Churned"
    if proba >= 0.70: return "High"
    if proba >= 0.40: return "Medium"
    return "Low"


def batch_predict_all(best_name, best_model, X_full, y_full):
    log.info(f"  Predicting all {len(X_full):,} records...")

    # Predict in chunks
    X_arr = X_full.values if hasattr(X_full, "values") else np.array(X_full)
    all_probas, all_preds = [], []
    for start in range(0, len(X_arr), 500):
        chunk  = X_arr[start:start+500]
        probas = best_model.predict_proba(chunk)[:, 1]
        preds  = best_model.predict(chunk)
        all_probas.extend(probas.tolist())
        all_preds.extend(preds.tolist())

    # Get IDs in same order as X_full (use index to match)
    engine = create_engine(DB_URL, echo=False)
    id_col = SCHEMA.get("id_col")
    try:
        # Read ALL IDs ordered by table row — must match X_full order
        id_df      = pd.read_sql(f"SELECT `{id_col}` FROM features LIMIT {len(X_full)}", con=engine)
        client_ids = id_df[id_col].astype(str).tolist()
    except Exception:
        client_ids = [str(i) for i in range(1, len(X_full)+1)]

    y_vals = y_full.values.tolist() if hasattr(y_full, "values") else list(y_full)

    # Ensure all same length
    n = min(len(client_ids), len(all_probas), len(y_vals))
    client_ids = client_ids[:n]
    all_probas = all_probas[:n]
    all_preds  = all_preds[:n]
    y_vals     = y_vals[:n]

    df = pd.DataFrame({
        ID_COL:            client_ids,
        "churn_actual":    [int(v) for v in y_vals],
        "churn_proba":     [round(float(p), 4) for p in all_probas],
        "churn_predicted": [int(p) for p in all_preds],
        "risk_segment":    [assign_risk(p, v) for p, v in zip(all_probas, y_vals)],
        "model_name":      best_name,
    })

    log.info(f"  Total predicted: {len(df):,} | Segments: {df['risk_segment'].value_counts().to_dict()}")
    return df

    # Get IDs
    engine = create_engine(DB_URL, echo=False)
    id_col = SCHEMA.get("id_col")
    try:
        id_df      = pd.read_sql(f"SELECT `{id_col}` FROM features", con=engine)
        client_ids = id_df[id_col].astype(str).tolist()
    except Exception:
        client_ids = [str(i) for i in range(1, len(X_full)+1)]

    y_vals    = y_full.values.tolist() if hasattr(y_full, "values") else list(y_full)
    min_len   = min(len(client_ids), len(all_probas), len(y_vals))
    client_ids = client_ids[:min_len]
    all_probas = all_probas[:min_len]
    all_preds  = all_preds[:min_len]
    y_vals     = y_vals[:min_len]

    df = pd.DataFrame({
        ID_COL:            client_ids,
        "churn_actual":    [int(v) for v in y_vals],
        "churn_proba":     [round(float(p), 4) for p in all_probas],
        "churn_predicted": [int(p) for p in all_preds],
        "risk_segment":    [assign_risk(p, v) for p, v in zip(all_probas, y_vals)],
        "model_name":      best_name,
    })

    log.info(f"  Segments: {df['risk_segment'].value_counts().to_dict()}")
    return df


def write_predictions(df):
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
                INDEX idx_proba (churn_proba),
                INDEX idx_cid   (CLIENTNUM)
            )
        """))
    rows = df.to_dict(orient="records")
    with engine.begin() as conn:
        for start in range(0, len(rows), 500):
            conn.execute(text("""
                INSERT INTO churn_predictions
                (CLIENTNUM,churn_actual,churn_proba,churn_predicted,risk_segment,model_name)
                VALUES
                (:CLIENTNUM,:churn_actual,:churn_proba,:churn_predicted,:risk_segment,:model_name)
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


def run(best_name, best_model, X_test, y_test, test_ids, shap_df,
        X_full=None, y_full=None):
    log.info("Writing predictions...")
    if X_full is not None and y_full is not None:
        pred_df = batch_predict_all(best_name, best_model, X_full, y_full)
    else:
        yp  = best_model.predict_proba(X_test)[:, 1]
        yh  = best_model.predict(X_test)
        y_v = y_test.values.tolist()
        pred_df = pd.DataFrame({
            ID_COL:            [str(c) for c in test_ids],
            "churn_actual":    [int(v) for v in y_v],
            "churn_proba":     [round(float(p),4) for p in yp],
            "churn_predicted": yh.tolist(),
            "risk_segment":    [assign_risk(p,v) for p,v in zip(yp, y_v)],
            "model_name":      best_name,
        })
    write_predictions(pred_df)
    write_shap(shap_df)
    return pred_df