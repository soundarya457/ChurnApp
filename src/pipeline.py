"""
pipeline.py — Single orchestrator used by both app.py (dashboard) and main.py (CLI).
"""
import logging
from src import ingest, preprocess, train, evaluate, explain, predict
from src.config import SCHEMA

log = logging.getLogger(__name__)


def run_pipeline(csv_path, target_col, positive_label, step=None, skip_ingest=False):
    """
    step(msg, stage="", progress=None) is an optional progress callback.
    target_col / positive_label must already be user-confirmed.
    """
    def say(msg, stage="", progress=None):
        log.info(msg)
        if step: step(msg, stage, progress)

    say("Stage 1/6 — Loading dataset into MySQL...", "Ingesting", 8)
    if skip_ingest:
        ingest.load_schema_from_db(target_col, positive_label)
    else:
        ingest.run(csv_path, target_col, positive_label)
    say(f"  Target: '{SCHEMA['target_col']}' (churn = '{SCHEMA['target_positive']}') | ID: '{SCHEMA['id_col']}'", progress=16)

    say("Stage 2/6 — Split → SMOTE (train only) → leakage check...", "Preprocessing", 20)
    data = preprocess.run()
    rep = data["split_report"]
    say(f"  {len(data['feature_names'])} features | Train: {len(data['X_train']):,} "
        f"(incl. {rep['synthetic_rows_added']:,} synthetic) | Test: {len(data['X_test']):,} "
        f"| train/test overlap: {rep['source_row_overlap']} ✓", progress=35)

    say("Stage 3/6 — Training Logistic Regression, Random Forest and XGBoost...", "Training", 38)
    fitted = train.train_all(data["X_train"], data["y_train"])
    say("  Models trained.", progress=58)

    say("Stage 4/6 — Evaluating on held-out test set...", "Evaluating", 62)
    all_metrics, cv, best_name, best_model, board, sel_note = evaluate.run(
        fitted, data["X_train_raw"], data["y_train_raw"], data["X_test"], data["y_test"])
    bm = next(m for m in all_metrics if m["model"] == best_name)
    cm = bm["confusion_matrix"]
    say("  Model leaderboard (winner chosen by cross-validated accuracy on training data):", progress=70)
    for r in board:
        say(f"   {'★' if r['selected'] else '·'} {r['model']:<19} CV acc {r['cv_accuracy']:.1%} | "
            f"test acc {r['test_accuracy']:.1%} · recall {r['test_recall']:.1%} · "
            f"F1 {r['test_f1']:.1%} · AUC {r['test_roc_auc']:.3f}")
    if sel_note:
        say("  Note: " + sel_note)
    say(f"  ★ Selected model: {best_name} | acc={bm['accuracy']} prec={bm['precision']} "
        f"rec={bm['recall']} F1={bm['f1']} AUC={bm['roc_auc']}", progress=72)
    say(f"  Confusion matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}", progress=74)

    say("Stage 5/6 — Generating SHAP explanations...", "Explaining", 76)
    shap_df, _ = explain.run(best_name, best_model, data["X_train"], data["X_test"],
                             data["feature_names"], data["ids_test"])
    say("  SHAP complete.", progress=88)

    say("Stage 6/6 — Writing predictions to MySQL...", "Saving", 91)
    pred_df = predict.run(best_name, best_model, data, shap_df)
    say(f"  {len(pred_df):,} records scored and matched to customers by ID.", progress=100)
    return {"best_model": best_name, "metrics": bm, "scored": len(pred_df),
            "leaderboard": board, "selection_note": sel_note}
