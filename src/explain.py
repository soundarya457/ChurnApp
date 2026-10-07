"""
explain.py — SHAP + LIME. Explanations are computed on HELD-OUT TEST rows and
stored against each row's own customer ID (previously the first training rows
were labelled with test IDs).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from src.config import OUTPUT_DIR, RANDOM_STATE
import logging

log = logging.getLogger(__name__)
SHAP_MAX_ROWS = 2000


def _parts(model):
    return model.named_steps.get("sc"), model.named_steps["clf"]


def _scaled(model, X):
    scaler, _ = _parts(model)
    arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    return scaler.transform(arr) if scaler is not None else arr


def get_explainer(best_name, best_model, X_train):
    _, clf = _parts(best_model)
    bg = shap.sample(_scaled(best_model, X_train), min(100, len(X_train)), random_state=RANDOM_STATE)
    if best_name in ("XGBoost", "RandomForest"):
        return shap.TreeExplainer(clf), True
    if best_name == "LogisticRegression":
        return shap.LinearExplainer(clf, bg), True
    return shap.KernelExplainer(lambda x: clf.predict_proba(x)[:, 1], bg), False


def compute_shap(explainer, X_arr):
    sv = explainer.shap_values(X_arr)
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    return sv


def plot_summary(sv, raw_arr, feature_names):
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, raw_arr, feature_names=feature_names, show=False, plot_type="dot")
    plt.title("SHAP Summary — Global Feature Impact (test rows)")
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


def build_shap_table(sv, raw_arr, feature_names, client_ids):
    rows = []
    for i, cid in enumerate(client_ids):
        ranked = np.argsort(np.abs(sv[i]))[::-1][:5]
        for rank, fi in enumerate(ranked, 1):
            rows.append(dict(CLIENTNUM=str(cid), feature_name=feature_names[fi],
                             shap_value=round(float(sv[i][fi]), 6),
                             feature_value=round(float(raw_arr[i, fi]), 4),
                             rank_order=rank))
    return pd.DataFrame(rows)


def run(best_name, best_model, X_train, X_test, feature_names, test_ids):
    test_ids = np.asarray(test_ids)
    if len(test_ids) != len(X_test):
        raise ValueError("test_ids and X_test differ in length — cannot align SHAP rows to customers.")

    explainer, fast = get_explainer(best_name, best_model, X_train)
    n = min(SHAP_MAX_ROWS if fast else 100, len(X_test))
    idx = np.sort(np.random.RandomState(RANDOM_STATE).choice(len(X_test), n, replace=False))
    X_s = X_test.iloc[idx]                 # rows and IDs are selected with the SAME index
    ids_s = test_ids[idx]
    raw_arr = X_s.to_numpy()

    log.info(f"  SHAP with {best_name} on {n} held-out test rows...")
    sv = compute_shap(explainer, _scaled(best_model, X_s))

    plot_summary(sv, raw_arr, feature_names)
    imp_df = plot_bar(sv, feature_names)
    shap_df = build_shap_table(sv, raw_arr, feature_names, ids_s)

    try:   # LIME on the first test customer, through the full pipeline
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            X_train.to_numpy(), feature_names=feature_names,
            class_names=["No Churn", "Churn"], mode="classification", random_state=RANDOM_STATE)
        fn = lambda x: best_model.predict_proba(pd.DataFrame(x, columns=feature_names))
        exp = lime_exp.explain_instance(X_test.iloc[0].to_numpy(), fn, num_features=10)
        exp.save_to_file(str(OUTPUT_DIR/"lime_customer_0.html"))
        log.info("  LIME saved ✓")
    except Exception as e:
        log.warning(f"  LIME skipped: {e}")

    return shap_df, imp_df
