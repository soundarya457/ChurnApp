"""
train.py — Train 4 models. Feature names are dynamic (from preprocess).
"""
"""
train.py — Lightweight training for deployment.
Uses only 2 fast models to fit within free tier RAM/CPU limits.
"""
import joblib
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.config import RANDOM_STATE, OUTPUT_DIR
import logging

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def build_logistic():
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=500,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="saga",
            n_jobs=1,
        ))
    ])


def build_xgb():
    return Pipeline([
        ("clf", XGBClassifier(
            n_estimators=100,       # reduced from 300
            max_depth=4,            # reduced from 6
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=1,
            tree_method="hist",     # fastest method
        ))
    ])


MODELS = {
    "LogisticRegression": build_logistic,
    "XGBoost":            build_xgb,
}


def train_all(X_train, y_train, tune_xgb=False):
    # Use at most 3000 rows for training on free tier
    if len(X_train) > 3000:
        log.info(f"  Sampling 3000 rows from {len(X_train):,} for speed")
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train), 3000, replace=False
        )
        X_tr = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
        y_tr = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train

    fitted = {}
    for name, builder in MODELS.items():
        log.info(f"  Training {name}...")
        pipe = builder()
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        log.info(f"  {name} done ✓")

    joblib.dump(fitted, OUTPUT_DIR / "models.pkl")
    log.info("  Models saved ✓")
    return fitted


def load_models():
    return joblib.load(OUTPUT_DIR / "models.pkl")