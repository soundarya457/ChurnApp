-- ============================================================
-- 03_useful_queries.sql — Query results after pipeline run
-- ============================================================
USE churn_db;

-- 1. Risk summary
SELECT risk_segment, COUNT(*) AS total,
       SUM(churn_predicted) AS predicted_churners,
       ROUND(AVG(churn_proba)*100,1) AS avg_churn_prob_pct
FROM churn_predictions GROUP BY risk_segment
ORDER BY FIELD(risk_segment,'High','Medium','Low');

-- 2. Top 10 highest-risk customers
SELECT CLIENTNUM, ROUND(churn_proba*100,1) AS prob_pct, risk_segment, churn_actual
FROM churn_predictions ORDER BY churn_proba DESC LIMIT 10;

-- 3. Top global SHAP features
SELECT feature_name, ROUND(AVG(ABS(shap_value)),5) AS mean_abs_shap
FROM shap_explanations GROUP BY feature_name ORDER BY mean_abs_shap DESC;

-- 4. False negatives (missed churners)
SELECT CLIENTNUM, ROUND(churn_proba*100,1) AS prob_pct
FROM churn_predictions WHERE churn_actual=1 AND churn_predicted=0
ORDER BY churn_proba DESC;

-- 5. Model accuracy
SELECT model_name,
       ROUND(AVG(CASE WHEN churn_actual=churn_predicted THEN 1.0 ELSE 0 END)*100,2) AS accuracy_pct
FROM churn_predictions GROUP BY model_name;
