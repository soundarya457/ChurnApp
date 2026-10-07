
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
            n_estimators=250,       # enough capacity for larger datasets
            max_depth=4,            # reduced from 6
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,     # SMOTE already balances the training set
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=1,
            tree_method="hist",     # fastest method
        ))
    ])


def build_rf():
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=3,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1,
        ))
    ])


MODELS = {
    "LogisticRegression": build_logistic,
    "RandomForest":       build_rf,
    "XGBoost":            build_xgb,
}


def train_all(X_train, y_train, tune_xgb=False):
    """Fit every candidate on the (already SMOTE-balanced) training split.
    No further sampling here: the 15,000-row cap is applied once, before SMOTE."""
    fitted = {}
    for name, builder in MODELS.items():
        log.info(f"  Training {name}...")
        pipe = builder()
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        log.info(f"  {name} done ✓")

    joblib.dump(fitted, OUTPUT_DIR / "models.pkl")
    log.info("  Models saved ✓")
    return fitted


def load_models():
    return joblib.load(OUTPUT_DIR / "models.pkl")

