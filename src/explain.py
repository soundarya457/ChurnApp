"""
explain.py — SHAP global + local explanations. Fully dynamic feature names.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from src.config import OUTPUT_DIR
import logging

log = logging.getLogger(__name__)


def get_explainer(best_name, best_model, X_train_arr):
    clf = best_model.named_steps["clf"]
    if best_name in ("XGBoost", "RandomForest"):
        return shap.TreeExplainer(clf)
    bg = shap.sample(X_train_arr, min(100, len(X_train_arr)))
    predict_fn = (lambda x: best_model.named_steps["clf"].predict_proba(
        best_model.named_steps["scaler"].transform(x)
        if "scaler" in best_model.named_steps else x)[:, 1])
    return shap.KernelExplainer(predict_fn, bg)


def transform(best_model, X):
    arr = X.values if hasattr(X, "values") else X
    if "scaler" in best_model.named_steps:
        arr = best_model.named_steps["scaler"].transform(arr)
    return arr


def compute_shap(explainer, X_arr):
    sv = explainer.shap_values(X_arr)
    if isinstance(sv, list): sv = sv[1]
    return sv


def plot_summary(sv, X_arr, feature_names):
    fig, _ = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_arr, feature_names=feature_names, show=False, plot_type="dot")
    plt.title("SHAP Summary — Global Feature Impact")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_bar(sv, feature_names):
    mean_sv = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": mean_sv}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df["feature"], df["importance"], color="#2563eb")
    ax.set_title("Mean |SHAP| — Top Churn Drivers"); ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"shap_bar.png", dpi=150)
    plt.close()
    return df.sort_values("importance", ascending=False)


def build_shap_table(sv, X_arr, feature_names, client_ids):
    rows = []
    for i, cid in enumerate(client_ids):
        ranked = np.argsort(np.abs(sv[i]))[::-1][:5]
        for rank, fi in enumerate(ranked, 1):
            rows.append(dict(CLIENTNUM=cid, feature_name=feature_names[fi],
                             shap_value=round(float(sv[i][fi]), 6),
                             feature_value=round(float(X_arr[i, fi]), 4),
                             rank_order=rank))
    return pd.DataFrame(rows)


def run(best_name, best_model, X_train, X_test, feature_names, client_ids):
    X_train_arr = transform(best_model, X_train)
    X_test_arr  = transform(best_model, X_test)
    sample_n    = min(100, len(X_train_arr))
    X_sample    = X_train_arr[:sample_n]
    ids_sample  = client_ids[:sample_n]

    log.info(f"  SHAP with {best_name} on {sample_n} samples...")
    explainer = get_explainer(best_name, best_model, X_train_arr)
    sv        = compute_shap(explainer, X_sample)

    plot_summary(sv, X_sample, feature_names)
    imp_df = plot_bar(sv, feature_names)
    shap_df = build_shap_table(sv, X_sample, feature_names, ids_sample)

    # LIME on first test customer
    try:
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            X_train_arr, feature_names=feature_names,
            class_names=["No Churn","Churn"], mode="classification", random_state=42)
        predict_fn = (lambda x: best_model.named_steps["clf"].predict_proba(
            best_model.named_steps["scaler"].transform(x)
            if "scaler" in best_model.named_steps else x))
        exp = lime_exp.explain_instance(X_test_arr[0], predict_fn, num_features=10)
        exp.save_to_file(str(OUTPUT_DIR/"lime_customer_0.html"))
        log.info("  LIME saved ✓")
    except Exception as e:
        log.warning(f"  LIME skipped: {e}")

    return shap_df, imp_df
