"""
evaluate.py — Metrics, plots, cross-validation for all models.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, RocCurveDisplay,
)
from sklearn.model_selection import cross_val_score
from src.config import OUTPUT_DIR, CV_FOLDS
import logging

log = logging.getLogger(__name__)


def eval_model(name, pipe, X_test, y_test):
    yp  = pipe.predict_proba(X_test)[:, 1]
    yh  = pipe.predict(X_test)
    m   = dict(model=name,
               roc_auc=round(roc_auc_score(y_test, yp), 4),
               f1=round(f1_score(y_test, yh), 4),
               precision=round(precision_score(y_test, yh, zero_division=0), 4),
               recall=round(recall_score(y_test, yh, zero_division=0), 4),
               accuracy=round(accuracy_score(y_test, yh), 4))
    log.info(f"  {name}: AUC={m['roc_auc']} F1={m['f1']} Rec={m['recall']}")
    return m, yp, yh


def cross_validate_all(fitted, X_train, y_train):
    res = {}
    for name, pipe in fitted.items():
        s = cross_val_score(pipe, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1)
        res[name] = {"mean": round(s.mean(), 4), "std": round(s.std(), 4)}
        log.info(f"  CV {name}: {s.mean():.4f} ± {s.std():.4f}")
    return res


def plot_roc(fitted, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0,1],[0,1],"k--")
    ax.set_title("ROC Curves"); ax.legend(loc="lower right")
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/"roc_curves.png", dpi=150); plt.close(fig)


def plot_cm(fitted, X_test, y_test):
    n = len(fitted)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    for ax, (name, pipe) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn","Churn"],
                    yticklabels=["No Churn","Churn"], ax=ax)
        ax.set_title(name); ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/"confusion_matrices.png", dpi=150); plt.close(fig)


def plot_metrics(all_metrics):
    df = pd.DataFrame(all_metrics).set_index("model")
    fig, ax = plt.subplots(figsize=(10, 5))
    df[["roc_auc","f1","precision","recall","accuracy"]].plot(kind="bar", ax=ax,
        colormap="tab10", edgecolor="white")
    ax.set_title("Model Comparison"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right"); ax.set_xticklabels(df.index, rotation=30, ha="right")
    plt.tight_layout(); fig.savefig(OUTPUT_DIR/"metrics_comparison.png", dpi=150); plt.close(fig)


def pick_best(all_metrics, fitted):
    best = max(all_metrics, key=lambda m: m["roc_auc"])
    log.info(f"  Best: {best['model']} AUC={best['roc_auc']}")
    return best["model"], fitted[best["model"]]


def run(fitted, X_train, y_train, X_test, y_test):
    all_metrics, all_probas = [], {}
    for name, pipe in fitted.items():
        m, yp, _ = eval_model(name, pipe, X_test, y_test)
        all_metrics.append(m); all_probas[name] = yp
    cv  = cross_validate_all(fitted, X_train, y_train)
    plot_roc(fitted, X_test, y_test)
    plot_cm(fitted, X_test, y_test)
    plot_metrics(all_metrics)
    best_name, best_model = pick_best(all_metrics, fitted)
    return all_metrics, cv, best_name, best_model
