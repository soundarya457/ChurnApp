import os, base64, threading, traceback, json, io, hashlib
from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from src.config import DB_URL, OUTPUT_DIR, DATA_DIR, SCHEMA, SOURCE_COL

app    = Flask(__name__)
engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)

pipeline_status = {
    "running": False, "stage": "", "progress": 0,
    "log": [], "error": "", "done": False,
}


def img64(f):
    p = OUTPUT_DIR / f
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

def qry(sql, params=None):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

UPLOAD_CSV    = DATA_DIR / "uploaded_dataset.csv"
PENDING_FILE  = DATA_DIR / "pending_schema.json"
CONFIRMED_FILE = DATA_DIR / "confirmed_schema.json"   # files (not memory) so it survives multiple workers

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def confirmed_for_current_csv():
    """Return the user's confirmed target selection only if it matches the CSV on disk."""
    conf = read_json(CONFIRMED_FILE)
    if conf and UPLOAD_CSV.exists() and conf.get("sha") == file_sha(UPLOAD_CSV):
        return conf
    return None

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
                   SUM(CASE WHEN risk_segment='Churned' THEN 1 ELSE 0 END) AS already_churned,
                   SUM(CASE WHEN risk_segment='High'    THEN 1 ELSE 0 END) AS high_risk,
                   SUM(CASE WHEN risk_segment='Medium'  THEN 1 ELSE 0 END) AS medium_risk,
                   SUM(CASE WHEN risk_segment='Low'     THEN 1 ELSE 0 END) AS low_risk,
                   MAX(model_name) AS model_name,
                   ROUND(SUM(CASE WHEN data_split='test' AND churn_actual=churn_predicted THEN 1.0 ELSE 0 END)
                         /NULLIF(SUM(data_split='test'),0),4) AS accuracy,
                   ROUND(SUM(CASE WHEN data_split='test' AND churn_actual=1 AND churn_predicted=1 THEN 1.0 ELSE 0 END)
                         /NULLIF(SUM(CASE WHEN data_split='test' THEN churn_predicted ELSE 0 END),0),4) AS precision_val,
                   ROUND(SUM(CASE WHEN data_split='test' AND churn_actual=1 AND churn_predicted=1 THEN 1.0 ELSE 0 END)
                         /NULLIF(SUM(CASE WHEN data_split='test' THEN churn_actual ELSE 0 END),0),4) AS recall_val,
                   SUM(CASE WHEN data_split='test' AND churn_actual=1 AND churn_predicted=0 THEN 1 ELSE 0 END) AS false_negatives,
                   ROUND(AVG(churn_proba)*100,1) AS avg_churn_proba_pct
            FROM churn_predictions
        """).iloc[0]

        # Generic raw_data stats — works for any dataset
        raw_total = qry("SELECT COUNT(*) AS n FROM raw_data").iloc[0]["n"]

        # Try to compute actual churn rate from raw_data if target col known
               # REPLACE the target_col block with features table query (always works)
        try:
            r = qry("SELECT ROUND(SUM(churn)/COUNT(*)*100,1) AS cr FROM features").iloc[0]
            actual_churn_rate = float(r["cr"] or 0)
        except Exception:
            actual_churn_rate = None
        # Try avg numeric columns generically
                # Try avg numeric columns generically
        num_stats = {}
        try:
            cols = qry("SELECT * FROM raw_data LIMIT 1").columns.tolist()
            for col in ["Customer_Age","Credit_Limit","Total_Trans_Ct","Avg_Utilization_Ratio"]:
                if col in cols:
                    val = qry(f"SELECT ROUND(AVG(`{col}`),1) AS v FROM raw_data").iloc[0]["v"]
                    num_stats[col] = float(val or 0)
        except Exception:
            pass

        # Avg age by churn class — read from features table (always has churn col)
        avg_age_churned = avg_age_existing = None
        try:
            r = qry("""
                SELECT
                    ROUND(AVG(CASE WHEN churn=1 THEN Customer_Age END),1) AS ac,
                    ROUND(AVG(CASE WHEN churn=0 THEN Customer_Age END),1) AS ae
                FROM features
            """).iloc[0]
            avg_age_churned  = float(r["ac"]) if r["ac"] is not None else None
            avg_age_existing = float(r["ae"]) if r["ae"] is not None else None
        except Exception:
            pass

        # Use held-out evaluation metrics for model quality.
        test_metrics = {}
        try:
            metrics_path = OUTPUT_DIR / "metrics.json"
            if metrics_path.exists():
                test_metrics = json.loads(metrics_path.read_text()).get("test_metrics", {})
        except Exception:
            pass

        return jsonify({
            "total_raw":          int(raw_total or 0),
            "total_scored":       int(pred["total_scored"]       or 0),
            "churn_rate":         round(float(pred["predicted_churners"] or 0)/float(pred["total_scored"] or 1)*100, 1),
            "actual_churn_rate":  actual_churn_rate,
            "already_churned":    int(pred["already_churned"]    or 0),
            "high_risk":          int(pred["high_risk"]          or 0),
            "medium_risk":        int(pred["medium_risk"]        or 0),
            "low_risk":           int(pred["low_risk"]           or 0),
            "best_model":         str(pred["model_name"]         or ""),
            "accuracy":           float(test_metrics.get("accuracy", pred["accuracy"] or 0)),
            "precision":          float(test_metrics.get("precision", pred["precision_val"] or 0)),
            "recall":             float(test_metrics.get("recall", pred["recall_val"] or 0)),
            "f1":                 float(test_metrics.get("f1", 0)),
            "roc_auc":            float(test_metrics.get("roc_auc", 0)),
            "confusion_matrix":   test_metrics.get("confusion_matrix"),
            "false_negatives":    int((test_metrics.get("confusion_matrix") or {}).get("fn", pred["false_negatives"] or 0)),
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
    per_page = int(request.args.get("per_page", 50))
    segment  = request.args.get("segment", "")
    offset   = (page - 1) * per_page
    where    = "WHERE risk_segment = :segment" if segment else ""
    params   = {"segment": segment} if segment else {}
    try:
        total = qry(f"SELECT COUNT(*) AS n FROM churn_predictions {where}", params).iloc[0]["n"]
        df    = qry(f"""
            SELECT CLIENTNUM, churn_actual, churn_proba, churn_predicted, risk_segment
            FROM churn_predictions {where}
            ORDER BY churn_proba DESC
            LIMIT {int(per_page)} OFFSET {int(offset)}
        """, params)
        return jsonify({
            "total":     int(total),
            "page":      page,
            "per_page":  per_page,
            "customers": df.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/customers/download")
def api_customers_download():
    """Download the complete risk table: every raw customer field + prediction fields."""
    segment = request.args.get("segment", "").strip()
    try:
        id_col = SCHEMA.get("id_col") or "_row_id"
        # Use a parameter for the filter; quote the dynamic column identifier.
        seg_clause = "WHERE p.risk_segment = :segment" if segment else ""
        sql = text(f"""
            SELECT r.*,
                   p.churn_actual,
                   p.churn_proba,
                   p.churn_predicted,
                   p.risk_segment,
                   p.model_name,
                   p.scored_at
            FROM raw_data r
            INNER JOIN churn_predictions p
              ON CAST(r.`{id_col}` AS CHAR) = p.CLIENTNUM
            {seg_clause}
            ORDER BY p.churn_proba DESC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"segment": segment} if segment else {})
        df = df.drop(columns=[SOURCE_COL], errors="ignore")

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        fname = f"churn_risk_table{'_'+segment if segment else ''}.csv"
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={fname}"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/customer/<path:client_id>")
def api_customer_detail(client_id):
    """Return prediction + original uploaded-record fields + SHAP details."""
    try:
        id_col = SCHEMA.get("id_col") or "_row_id"
        pred_sql = text("""
            SELECT *
            FROM churn_predictions
            WHERE CLIENTNUM = :client_id
            LIMIT 1
        """)
        raw_sql = text(f"""
            SELECT *
            FROM raw_data
            WHERE CAST(`{id_col}` AS CHAR) = :client_id
            LIMIT 1
        """)
        shap_sql = text("""
            SELECT feature_name, shap_value, feature_value, rank_order
            FROM shap_explanations
            WHERE CLIENTNUM = :client_id
            ORDER BY rank_order
        """)
        with engine.connect() as conn:
            pred = pd.read_sql(pred_sql, conn, params={"client_id": str(client_id)})
            raw = pd.read_sql(raw_sql, conn, params={"client_id": str(client_id)})
            shap = pd.read_sql(shap_sql, conn, params={"client_id": str(client_id)})

        def clean(df):
            # NaN/NaT are not valid JSON; convert to None so the browser can parse it
            if df.empty:
                return {}
            # Series.to_dict() returns native Python types (numpy ints aren't JSON-serializable)
            row = df.iloc[0].to_dict()
            return {k: (None if pd.isna(v) else v) for k, v in row.items()}

        prediction = clean(pred)
        record = clean(raw.drop(columns=[SOURCE_COL], errors="ignore"))
        return jsonify({
            "prediction": prediction,
            "record": record,
            "shap": shap.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/plots/<name>")
def api_plot(name):
    allowed = ["roc_curves","confusion_matrices","shap_summary","shap_bar","metrics_comparison"]
    if name not in allowed: return jsonify({"error":"not found"}), 404
    return jsonify({"img": img64(f"{name}.png")})

# ── Upload + Pipeline ──────────────────────────────────────────

@app.route("/api/model_metrics")
def api_model_metrics():
    """Held-out test metrics (accuracy/precision/recall/F1/confusion matrix) + leakage report."""
    return jsonify({
        "metrics": read_json(OUTPUT_DIR / "metrics.json"),
        "split":   read_json(OUTPUT_DIR / "split_report.json"),
    })


@app.route("/api/upload_dataset", methods=["POST"])
def api_upload_dataset():
    """Step 1: store the CSV and RECOMMEND target columns. Nothing is trained yet."""
    if "file" not in request.files:
        return jsonify({"error":"No file provided"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error":"Only CSV files are supported"}), 400
    try:
        from src import ingest as _ing
        df = pd.read_csv(f)
        if len(df) < 50:
            return jsonify({"error":"Dataset too small — need at least 50 rows"}), 400
        if len(df.columns) < 3:
            return jsonify({"error":"Dataset needs at least 3 columns"}), 400

        id_col = _ing.detect_id_col(df)
        candidates, rec = _ing.find_target_candidates(df, id_col)
        if not candidates:
            return jsonify({"error":"No possible target column found. The target needs a column with exactly 2 distinct values (churn / not churn)."}), 400

        df.to_csv(UPLOAD_CSV, index=False)
        if CONFIRMED_FILE.exists():
            CONFIRMED_FILE.unlink()          # a new file always needs a new confirmation
        PENDING_FILE.write_text(json.dumps({"sha": file_sha(UPLOAD_CSV), "filename": f.filename}))

        return jsonify({
            "success": True,
            "requires_confirmation": True,
            "rows": len(df), "columns": len(df.columns), "filename": f.filename,
            "id_col": str(id_col) if id_col else "auto-generated",
            "duplicate_ids": _ing.count_duplicate_ids(df, id_col),
            "candidates": candidates,
            "recommendation": rec,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/confirm_target", methods=["POST"])
def api_confirm_target():
    """Step 2: user picks the target column + which value means churn. Re-validated server-side."""
    body = request.get_json(silent=True) or {}
    col, pos = body.get("target_col"), body.get("positive_label")
    if not col or pos is None:
        return jsonify({"error":"Choose a target column and the value that means churn."}), 400
    if not UPLOAD_CSV.exists() or not PENDING_FILE.exists():
        return jsonify({"error":"Upload a dataset first."}), 400
    try:
        from src import ingest as _ing
        df = pd.read_csv(UPLOAD_CSV)
        id_col = _ing.detect_id_col(df)
        ok, errors, warnings, info = _ing.validate_target(df, col, pos, id_col)
        if not ok:
            return jsonify({"error":" ".join(errors), "errors":errors}), 400
        cat_cols, num_cols, drop_cols = _ing.detect_col_types(df, id_col, col)
        conf = {
            "sha": file_sha(UPLOAD_CSV), "target_col": col, "positive_label": info["positive"],
            "n_positive": info["n_positive"], "n_negative": info["n_negative"],
            "churn_rate": info["churn_rate"], "warnings": warnings,
        }
        CONFIRMED_FILE.write_text(json.dumps(conf))
        return jsonify({
            "success": True, **conf,
            "class_distribution": info["values"],
            "id_col": str(id_col) if id_col else "auto-generated",
            "cat_cols": cat_cols, "num_cols": num_cols, "drop_cols": drop_cols,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/schema_state")
def api_schema_state():
    conf = confirmed_for_current_csv()
    return jsonify({"confirmed": bool(conf), "selection": conf})


@app.route("/api/run_pipeline", methods=["POST"])
def api_run_pipeline():
    global pipeline_status
    if pipeline_status["running"]:
        return jsonify({"error":"Pipeline already running"}), 400
    conf = confirmed_for_current_csv()
    if not conf:
        return jsonify({"error":"Confirm the target column (and which value means churn) before training."}), 400

    pipeline_status = {"running":True,"stage":"Starting","progress":0,"log":[],"error":"","done":False}

    def run():
        global pipeline_status
        try:
            from src import pipeline
            res = pipeline.run_pipeline(UPLOAD_CSV, conf["target_col"], conf["positive_label"], step=log_s)
            m = res["metrics"]
            pipeline_status.update({"stage":"Complete","done":True,"running":False,
                                    "best_model":res["best_model"],"roc_auc":m["roc_auc"],
                                    "accuracy":m["accuracy"],"precision":m["precision"],
                                    "recall":m["recall"],"f1":m["f1"],
                                    "confusion_matrix":m["confusion_matrix"],
                                    "leaderboard":res["leaderboard"],
                                    "selection_note":res["selection_note"],
                                    "scored":res["scored"]})
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
