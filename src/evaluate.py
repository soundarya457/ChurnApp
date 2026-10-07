"""
evaluate.py — Held-out metrics, confusion matrices, leak-free cross-validation.

Reported per model on the untouched test split:
accuracy, precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix
(TN / FP / FN / TP).  Accuracy alone is misleading on churn data, so recall
(how many real churners are caught) and precision are shown beside it.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, average_precision_score,
    precision_score, recall_score, confusion_matrix, RocCurveDisplay,
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from src.config import OUTPUT_DIR, CV_FOLDS, RANDOM_STATE, SELECTION_METRIC, MIN_RECALL
import logging

log = logging.getLogger(__name__)


def eval_model(name, pipe, X_test, y_test):
    yp = pipe.predict_proba(X_test)[:, 1]
    yh = pipe.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, yh, labels=[0, 1]).ravel()
    m = dict(
        model=name,
        accuracy=round(accuracy_score(y_test, yh), 4),
        precision=round(precision_score(y_test, yh, zero_division=0), 4),
        recall=round(recall_score(y_test, yh, zero_division=0), 4),
        f1=round(f1_score(y_test, yh, zero_division=0), 4),
        roc_auc=round(roc_auc_score(y_test, yp), 4),
        pr_auc=round(average_precision_score(y_test, yp), 4),
        confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        test_rows=int(len(y_test)),
        actual_churners=int(tp + fn),
        churners_caught=int(tp),
        churners_missed=int(fn),
        baseline_accuracy=round(float(max(np.mean(y_test), 1 - np.mean(y_test))), 4),
    )
    log.info(f"  {name}: acc={m['accuracy']} prec={m['precision']} rec={m['recall']} "
             f"f1={m['f1']} auc={m['roc_auc']} | TN={tn} FP={fp} FN={fn} TP={tp}")
    return m, yp, yh


def cross_validate_all(fitted, X_train_raw, y_train_raw):
    """
    CV on the ORIGINAL (pre-SMOTE) training rows with SMOTE inside each fold,
    so synthetic copies of a validation row never sit in that fold's training set.
    """
    res = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_min = int(y_train_raw.value_counts().min())
    k = max(1, min(3, (n_min * (CV_FOLDS - 1) // CV_FOLDS) - 1))
    for name, pipe in fitted.items():
        leak_free = ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k)),
                                 *pipe.steps])
        s = cross_validate(leak_free, X_train_raw, y_train_raw, cv=cv,
                           scoring={"roc_auc": "roc_auc", "recall": "recall", "accuracy": "accuracy",
                                    "precision": "precision", "f1": "f1"}, n_jobs=1)
        res[name] = {
            "mean": round(float(s["test_roc_auc"].mean()), 4),      # CV ROC-AUC
            "std":  round(float(s["test_roc_auc"].std()), 4),
            "roc_auc_mean":   round(float(s["test_roc_auc"].mean()), 4),
            "accuracy_mean":  round(float(s["test_accuracy"].mean()), 4),
            "precision_mean": round(float(s["test_precision"].mean()), 4),
            "recall_mean":    round(float(s["test_recall"].mean()), 4),
            "f1_mean":        round(float(s["test_f1"].mean()), 4),
        }
        log.info(f"  CV {name}: accuracy {res[name]['accuracy_mean']:.4f} | recall {res[name]['recall_mean']:.4f} | "
                 f"F1 {res[name]['f1_mean']:.4f} | AUC {res[name]['mean']:.4f} ± {res[name]['std']:.4f}")
    return res


def plot_roc(fitted, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0,1],[0,1],"k--")
    ax.set_title("ROC Curves (held-out test set)"); ax.legend(loc="lower right")
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/"roc_curves.png", dpi=150); plt.close(fig)


def plot_cm(all_metrics):
    n = len(all_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.2*n, 4.6))
    if n == 1: axes = [axes]
    for ax, m in zip(axes, all_metrics):
        c = m["confusion_matrix"]
        cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        row = cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm / row, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                tag = [["TN", "FP"], ["FN", "TP"]][i][j]
                ax.text(j, i, f"{tag}\n{cm[i, j]:,}\n({cm[i, j]/row[i, 0]:.1%} of row)",
                        ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j]/row[i, 0] > 0.55 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        ax.set_title(f"{m['model']}\nacc {m['accuracy']:.1%} · prec {m['precision']:.1%} · "
                     f"rec {m['recall']:.1%} · F1 {m['f1']:.1%}", fontsize=10)
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/"confusion_matrices.png", dpi=150); plt.close(fig)


def plot_metrics(all_metrics):
    df = pd.DataFrame(all_metrics).set_index("model")
    fig, ax = plt.subplots(figsize=(10, 5))
    df[["accuracy","precision","recall","f1","roc_auc"]].plot(
        kind="bar", ax=ax, colormap="tab10", edgecolor="white")
    ax.set_title("Model Comparison (held-out test set)"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right"); ax.set_xticklabels(df.index, rotation=30, ha="right")
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/"metrics_comparison.png", dpi=150); plt.close(fig)


def pick_best(all_metrics, cv, fitted):
    """
    Select the winner from CROSS-VALIDATION on training data (never from the test set).
    Highest CV `SELECTION_METRIC` wins among models whose CV recall >= MIN_RECALL.
    If no model reaches the recall floor, the one with the best CV recall is used instead.
    Returns (best_name, best_model, leaderboard, note).
    """
    key = {"accuracy": "accuracy_mean", "f1": "f1_mean", "roc_auc": "roc_auc_mean"}[SELECTION_METRIC]
    names = [m["model"] for m in all_metrics]
    eligible = [n for n in names if cv[n]["recall_mean"] >= MIN_RECALL]
    note = ""
    if eligible:
        best = max(eligible, key=lambda n: (cv[n][key], cv[n]["recall_mean"]))
        skipped = [n for n in names if n not in eligible]
        if skipped:
            note = (f"{', '.join(skipped)} skipped: caught under {MIN_RECALL:.0%} of churners in "
                    f"cross-validation (high accuracy there only means 'predict no churn').")
    else:
        best = max(names, key=lambda n: cv[n]["recall_mean"])
        note = (f"No model reached {MIN_RECALL:.0%} cross-validated recall, so the one catching the most "
                f"churners was chosen. This data may carry little signal about churn.")
    board = []
    for m in all_metrics:
        n = m["model"]
        board.append(dict(
            model=n, selected=(n == best), eligible=(n in eligible),
            cv_accuracy=cv[n]["accuracy_mean"], cv_recall=cv[n]["recall_mean"],
            cv_f1=cv[n]["f1_mean"], cv_roc_auc=cv[n]["roc_auc_mean"],
            test_accuracy=m["accuracy"], test_precision=m["precision"],
            test_recall=m["recall"], test_f1=m["f1"], test_roc_auc=m["roc_auc"]))
    board.sort(key=lambda r: (not r["selected"], -r["cv_accuracy"]))
    log.info(f"  ── Model leaderboard (selected by cross-validated {SELECTION_METRIC}) ──")
    for r in board:
        log.info(f"  {'★' if r['selected'] else ' '} {r['model']:<19} CV acc {r['cv_accuracy']:.3f} · CV recall {r['cv_recall']:.3f}"
                 f"  |  test acc {r['test_accuracy']:.3f} · prec {r['test_precision']:.3f} · rec {r['test_recall']:.3f} · "
                 f"F1 {r['test_f1']:.3f} · AUC {r['test_roc_auc']:.3f}")
    if note:
        log.info("  " + note)
    log.info(f"  Selected: {best}")
    return best, fitted[best], board, note


def run(fitted, X_train_raw, y_train_raw, X_test, y_test):
    all_metrics = []
    for name, pipe in fitted.items():
        m, _, _ = eval_model(name, pipe, X_test, y_test)
        all_metrics.append(m)
    cv = cross_validate_all(fitted, X_train_raw, y_train_raw)
    plot_roc(fitted, X_test, y_test)
    plot_cm(all_metrics)
    plot_metrics(all_metrics)
    best_name, best_model, board, sel_note = pick_best(all_metrics, cv, fitted)
    best_m = next(m for m in all_metrics if m["model"] == best_name)
    warning = ""
    if best_m["roc_auc"] < 0.55:
        warning = (f"Best ROC-AUC is {best_m['roc_auc']} — essentially random guessing (0.50). "
                   "These columns carry no usable signal about the target (e.g. synthetic or randomly "
                   "generated data), so no model can do better. Check the target column and the data source.")
    elif best_m["accuracy"] <= best_m["baseline_accuracy"] + 0.01:
        warning = (f"Accuracy {best_m['accuracy']:.1%} is no better than always predicting the majority "
                   f"class ({best_m['baseline_accuracy']:.1%}). Judge the model by recall, precision and AUC.")
    if warning:
        log.warning("  " + warning)
    try:
        (OUTPUT_DIR / "metrics.json").write_text(json.dumps({
            "all_metrics": all_metrics,
            "cv": cv,
            "best_model": best_name,
            "selection_criterion": (f"highest cross-validated {SELECTION_METRIC} on training data, "
                                    f"among models with CV recall >= {MIN_RECALL:.0%}"),
            "selection_note": sel_note,
            "leaderboard": board,
            "test_metrics": best_m,
            "baseline_accuracy": best_m["baseline_accuracy"],
            "signal_warning": warning,
        }, indent=2))
    except Exception as e:
        log.warning(f"Could not save metrics.json: {e}")
    return all_metrics, cv, best_name, best_model, board, sel_note
