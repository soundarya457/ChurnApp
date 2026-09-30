
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


MODELS = {
    "LogisticRegression": build_logistic,
    "XGBoost":            build_xgb,
}


def train_all(X_train, y_train, tune_xgb=False):
    # Cap training for Render/free-tier runtime, but use substantially more
    # information than the old 3,000-row cap. The full dataset is still
    # scored below; this cap affects training only.
    TRAIN_CAP = 15000
    if len(X_train) > TRAIN_CAP:
        log.info(f"  Sampling {TRAIN_CAP:,} rows from {len(X_train):,} for training")
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train), TRAIN_CAP, replace=False
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
