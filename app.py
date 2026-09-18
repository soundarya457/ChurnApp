
import os, base64, threading, traceback, json
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from src.config import DB_URL, OUTPUT_DIR, DATA_DIR, SCHEMA

app    = Flask(__name__)
engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)

pipeline_status = {
    "running": False, "stage": "", "progress": 0,
    "log": [], "error": "", "done": False,
}


def img64(f):
    p = OUTPUT_DIR / f
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

def qry(sql):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

def log_s(msg, stage="", progress=None):
    pipeline_status["log"].append(msg)
    if stage:    pipeline_status["stage"]    = stage
    if progress is not None: pipeline_status["progress"] = progress
    print(msg)

# ── Pages ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

# ── Dashboard API ─────────────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    try:
        pred = qry("""
            SELECT COUNT(*) AS total_scored,
                   SUM(churn_predicted) AS predicted_churners,
                   SUM(CASE WHEN risk_segment='High'   THEN 1 ELSE 0 END) AS high_risk,
                   SUM(CASE WHEN risk_segment='Medium' THEN 1 ELSE 0 END) AS medium_risk,
                   SUM(CASE WHEN risk_segment='Low'    THEN 1 ELSE 0 END) AS low_risk,
                   MAX(model_name) AS model_name,
                   ROUND(SUM(CASE WHEN churn_actual=churn_predicted THEN 1.0 ELSE 0 END)/COUNT(*),4) AS accuracy,
                   ROUND(SUM(CASE WHEN churn_actual=1 AND churn_predicted=1 THEN 1.0 ELSE 0 END)
                         /NULLIF(SUM(churn_predicted),0),4) AS precision_val,
                   ROUND(SUM(CASE WHEN churn_actual=1 AND churn_predicted=1 THEN 1.0 ELSE 0 END)
                         /NULLIF(SUM(churn_actual),0),4) AS recall_val,
                   SUM(CASE WHEN churn_actual=1 AND churn_predicted=0 THEN 1 ELSE 0 END) AS false_negatives,
                   ROUND(AVG(churn_proba)*100,1) AS avg_churn_proba_pct
            FROM churn_predictions
        """).iloc[0]

        # Generic raw_data stats — works for any dataset
        raw_total = qry("SELECT COUNT(*) AS n FROM raw_data").iloc[0]["n"]

        # Try to compute actual churn rate from raw_data if target col known
        actual_churn_rate = None
        target_col = SCHEMA.get("target_col")
        target_pos = SCHEMA.get("target_positive")
        if target_col and target_pos:
            try:
                r = qry(f"""
                    SELECT ROUND(SUM(CASE WHEN `{target_col}`='{target_pos}' THEN 1 ELSE 0 END)/COUNT(*)*100,1) AS cr
                    FROM raw_data
                """).iloc[0]
                actual_churn_rate = float(r["cr"] or 0)
            except Exception:
                pass

        # Try avg numeric columns generically
        num_stats = {}
        try:
            cols = qry("SELECT * FROM raw_data LIMIT 1").columns.tolist()
            num_cols = [c for c in cols if c not in [SCHEMA.get("id_col"), SCHEMA.get("target_col")]]
            for col in ["Customer_Age","Credit_Limit","Total_Trans_Ct","Avg_Utilization_Ratio"]:
                if col in num_cols:
                    val = qry(f"SELECT ROUND(AVG(`{col}`),1) AS v FROM raw_data").iloc[0]["v"]
                    num_stats[col] = float(val or 0)
        except Exception:
            pass

        # Avg age by churn class if available
        avg_age_churned = avg_age_existing = None
        if "Customer_Age" in num_stats and target_col and target_pos:
            try:
                r = qry(f"""
                    SELECT
                        ROUND(AVG(CASE WHEN `{target_col}`='{target_pos}' THEN Customer_Age END),1) AS ac,
                        ROUND(AVG(CASE WHEN `{target_col}`!='{target_pos}' THEN Customer_Age END),1) AS ae
                    FROM raw_data
                """).iloc[0]
                avg_age_churned  = float(r["ac"] or 0)
                avg_age_existing = float(r["ae"] or 0)
            except Exception:
                pass

        return jsonify({
            "total_raw":          int(raw_total or 0),
            "total_scored":       int(pred["total_scored"]       or 0),
            "churn_rate":         round(float(pred["predicted_churners"] or 0)/float(pred["total_scored"] or 1)*100, 1),
            "actual_churn_rate":  actual_churn_rate,
            "high_risk":          int(pred["high_risk"]          or 0),
            "medium_risk":        int(pred["medium_risk"]        or 0),
            "low_risk":           int(pred["low_risk"]           or 0),
            "best_model":         str(pred["model_name"]         or ""),
            "accuracy":           float(pred["accuracy"]         or 0),
            "precision":          float(pred["precision_val"]    or 0),
            "recall":             float(pred["recall_val"]       or 0),
            "false_negatives":    int(pred["false_negatives"]    or 0),
            "avg_churn_proba":    float(pred["avg_churn_proba_pct"] or 0),
            "avg_age_churned":    avg_age_churned,
            "avg_age_existing":   avg_age_existing,
            "avg_credit_limit":   num_stats.get("Credit_Limit"),
            "avg_trans_ct":       num_stats.get("Total_Trans_Ct"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/risk_distribution")
def api_risk_distribution():
    try:
        df = qry("SELECT risk_segment, COUNT(*) AS count FROM churn_predictions GROUP BY risk_segment")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/churn_by_proba")
def api_churn_by_proba():
    try:
        df = qry("SELECT churn_proba, churn_actual FROM churn_predictions")
        h1, bins = np.histogram(df[df["churn_actual"]==1]["churn_proba"], bins=20, range=(0,1))
        h0, _    = np.histogram(df[df["churn_actual"]==0]["churn_proba"], bins=20, range=(0,1))
        return jsonify({"labels":[f"{bins[i]:.2f}–{bins[i+1]:.2f}" for i in range(len(bins)-1)],
                        "churn":h1.tolist(),"no_churn":h0.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/top_features")
def api_top_features():
    try:
        df = qry("""SELECT feature_name, ROUND(AVG(ABS(shap_value)),5) AS mean_abs_shap
                    FROM shap_explanations GROUP BY feature_name
                    ORDER BY mean_abs_shap DESC LIMIT 10""")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/customers")
def api_customers():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    segment  = request.args.get("segment", "")
    offset   = (page-1)*per_page
    where    = f"WHERE risk_segment='{segment}'" if segment else ""
    try:
        total = qry(f"SELECT COUNT(*) AS n FROM churn_predictions {where}").iloc[0]["n"]
        df    = qry(f"""SELECT CLIENTNUM,churn_actual,churn_proba,churn_predicted,risk_segment
                        FROM churn_predictions {where}
                        ORDER BY churn_proba DESC LIMIT {per_page} OFFSET {offset}""")
        return jsonify({"total":int(total),"page":page,"per_page":per_page,
                        "customers":df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/customer/<client_id>")
def api_customer_detail(client_id):
    try:
        pred = qry(f"SELECT * FROM churn_predictions WHERE CLIENTNUM='{client_id}'").to_dict(orient="records")
        shap = qry(f"""SELECT feature_name,shap_value,feature_value,rank_order
                       FROM shap_explanations WHERE CLIENTNUM='{client_id}'
                       ORDER BY rank_order""").to_dict(orient="records")
        return jsonify({"prediction":pred[0] if pred else {},"shap":shap})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/plots/<name>")
def api_plot(name):
    allowed = ["roc_curves","confusion_matrices","shap_summary","shap_bar","metrics_comparison"]
    if name not in allowed: return jsonify({"error":"not found"}), 404
    return jsonify({"img": img64(f"{name}.png")})

# ── Upload + Pipeline ──────────────────────────────────────────

@app.route("/api/upload_dataset", methods=["POST"])
def api_upload_dataset():
    if "file" not in request.files:
        return jsonify({"error":"No file provided"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error":"Only CSV files are supported"}), 400
    try:
        df = pd.read_csv(f)
        if len(df) < 50:
            return jsonify({"error":"Dataset too small — need at least 50 rows"}), 400
        if len(df.columns) < 3:
            return jsonify({"error":"Dataset needs at least 3 columns"}), 400

        # Auto-detect schema on upload for preview
        from src import ingest as _ing
        _ing.detect_schema(df)

        target_col = SCHEMA.get("target_col")
        target_pos = SCHEMA.get("target_positive")
        id_col     = SCHEMA.get("id_col")

        if not target_col:
            return jsonify({"error":"Could not detect a binary target column. Ensure your dataset has a column with exactly 2 unique values indicating churn vs non-churn."}), 400

        churn_count = int((df[target_col].astype(str) == str(target_pos)).sum())

        # Save CSV
        save_path = DATA_DIR / "uploaded_dataset.csv"
        df.to_csv(save_path, index=False)

        return jsonify({
            "success":      True,
            "rows":         len(df),
            "columns":      len(df.columns),
            "filename":     f.filename,
            "target_col":   target_col,
            "target_pos":   str(target_pos),
            "id_col":       str(id_col) if id_col else "auto-generated",
            "churn_count":  churn_count,
            "churn_rate":   round(churn_count/len(df)*100, 1),
            "cat_cols":     SCHEMA["categorical_cols"],
            "num_cols":     SCHEMA["numerical_cols"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_pipeline", methods=["POST"])
def api_run_pipeline():
    global pipeline_status
    if pipeline_status["running"]:
        return jsonify({"error":"Pipeline already running"}), 400

    pipeline_status = {"running":True,"stage":"Starting","progress":0,"log":[],"error":"","done":False}

    def run():
        global pipeline_status
        try:
            import importlib, sys

            def fresh(mod):
                if mod in sys.modules: del sys.modules[mod]
                return importlib.import_module(mod)

            # Stage 1 — Ingest
            log_s("Stage 1/6 — Loading dataset into MySQL...", "Ingesting", 8)
            ingest = fresh("src.ingest")
            csv_path = DATA_DIR / "uploaded_dataset.csv"
            if not csv_path.exists():
                # Fallback to any CSV in data/
                csvs = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not csvs: raise FileNotFoundError("No dataset found. Please upload a CSV first.")
                csv_path = csvs[0]
            ingest.run(csv_path)
            log_s(f"  Target: '{SCHEMA['target_col']}' | ID: '{SCHEMA['id_col']}'", progress=16)

            # Stage 2 — Preprocess
            log_s("Stage 2/6 — Feature engineering + SMOTE balancing...", "Preprocessing", 20)
            preprocess = fresh("src.preprocess")
            X_train, X_test, y_train, y_test, feature_names = preprocess.run()
            log_s(f"  {len(feature_names)} features | Train: {len(X_train):,} | Test: {len(X_test):,}", progress=35)

            # Fetch IDs
            from sqlalchemy import create_engine as CE
            from src.config import TEST_SIZE, RANDOM_STATE, SCHEMA as SC
            from sklearn.model_selection import train_test_split
            eng2 = CE(DB_URL)
            id_col = SC.get("id_col")
            if id_col:
                all_ids = pd.read_sql(f"SELECT `{id_col}` FROM features ORDER BY `{id_col}`", eng2)
                _, test_ids = train_test_split(all_ids[id_col].values, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            else:
                n = len(X_train) + len(X_test)
                _, test_ids = train_test_split(list(range(n)), test_size=TEST_SIZE, random_state=RANDOM_STATE)

            # Stage 3 — Train
            log_s("Stage 3/6 — Training Logistic Regression, Random Forest, XGBoost, MLP...", "Training", 38)
            train_mod = fresh("src.train")
            fitted = train_mod.train_all(X_train, y_train, tune_xgb=False)
            log_s("  All 4 models trained.", progress=58)

            # Stage 4 — Evaluate
            log_s("Stage 4/6 — Evaluating and selecting best model...", "Evaluating", 62)
            evaluate = fresh("src.evaluate")
            all_metrics, cv, best_name, best_model = evaluate.run(fitted, X_train, y_train, X_test, y_test)
            bm = next(m for m in all_metrics if m["model"]==best_name)
            log_s(f"  Best: {best_name} | AUC={bm['roc_auc']} | F1={bm['f1']}", progress=72)

            # Stage 5 — Explain
            log_s("Stage 5/6 — Generating SHAP explanations...", "Explaining", 76)
            explain = fresh("src.explain")
            shap_df, _ = explain.run(best_name, best_model, X_train, X_test, feature_names, test_ids)
            log_s("  SHAP complete.", progress=88)

            # Stage 6 — Predict
            log_s("Stage 6/6 — Writing predictions to MySQL...", "Saving", 91)
            predict = fresh("src.predict")
            pred_df = predict.run(best_name, best_model, X_test, y_test, test_ids, shap_df)
            log_s(f"  {len(pred_df):,} customers scored and saved.", progress=100)

            pipeline_status.update({"stage":"Complete","done":True,"running":False,
                                    "best_model":best_name,"roc_auc":bm["roc_auc"],
                                    "accuracy":bm["accuracy"],"scored":len(pred_df)})
        except Exception as e:
            pipeline_status.update({"error":str(e),"stage":"Error","running":False})
            pipeline_status["log"].append(f"ERROR: {traceback.format_exc()}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"started":True})


@app.route("/api/pipeline_status")
def api_pipeline_status():
    return jsonify(pipeline_status)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)