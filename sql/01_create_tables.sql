-- ============================================================
-- 01_create_tables.sql
-- Creates only the fixed OUTPUT tables.
-- raw_data and features are created dynamically by Python
-- to match whatever CSV the user uploads.
-- ============================================================

CREATE DATABASE IF NOT EXISTS churn_db;
USE churn_db;

-- Predictions output
DROP TABLE IF EXISTS churn_predictions;
CREATE TABLE churn_predictions (
    CLIENTNUM           VARCHAR(100),
    churn_actual        TINYINT,
    churn_proba         DECIMAL(6,4),
    churn_predicted     TINYINT,
    risk_segment        VARCHAR(10),
    model_name          VARCHAR(40),
    scored_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_seg (risk_segment),
    INDEX idx_proba (churn_proba)
);

-- SHAP explanations
DROP TABLE IF EXISTS shap_explanations;
CREATE TABLE shap_explanations (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    CLIENTNUM       VARCHAR(100),
    feature_name    VARCHAR(100),
    shap_value      DECIMAL(12,6),
    feature_value   DECIMAL(15,4),
    rank_order      TINYINT,
    INDEX idx_client (CLIENTNUM)
);
