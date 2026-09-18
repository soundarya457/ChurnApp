"""
train.py — Train 4 models. Feature names are dynamic (from preprocess).
"""
import joblib
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import optuna
from src.config import RANDOM_STATE, OUTPUT_DIR
import logging

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_logistic():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"))])

def build_rf():
    return Pipeline([("clf", RandomForestClassifier(n_estimators=200, max_depth=10,
                     min_samples_leaf=5, random_state=RANDOM_STATE,
                     class_weight="balanced", n_jobs=-1))])

def build_xgb():
    return Pipeline([("clf", XGBClassifier(n_estimators=300, max_depth=6,
                     learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                     scale_pos_weight=5, eval_metric="logloss",
                     random_state=RANDOM_STATE, verbosity=0))])

def build_mlp():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                     activation="relu", max_iter=500, random_state=RANDOM_STATE,
                     early_stopping=True, n_iter_no_change=20))])


MODELS = {
    "LogisticRegression": build_logistic,
    "RandomForest":       build_rf,
    "XGBoost":            build_xgb,
    "MLP":                build_mlp,
}


def tune_xgboost(X_train, y_train, n_trials=40):
    log.info("  Running Optuna for XGBoost...")
    def objective(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators",100,500),
                 max_depth=trial.suggest_int("max_depth",3,10),
                 learning_rate=trial.suggest_float("learning_rate",.01,.3,log=True),
                 subsample=trial.suggest_float("subsample",.6,1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree",.6,1.0),
                 min_child_weight=trial.suggest_int("min_child_weight",1,10))
        clf = XGBClassifier(**p, scale_pos_weight=5, eval_metric="logloss",
                            random_state=RANDOM_STATE, verbosity=0)
        return cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1).mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"  Best ROC-AUC: {study.best_value:.4f}")
    return Pipeline([("clf", XGBClassifier(**study.best_params, scale_pos_weight=5,
                                           eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0))])


def train_all(X_train, y_train, tune_xgb=False):
    fitted = {}
    for name, builder in MODELS.items():
        log.info(f"  Training {name}...")
        pipe = tune_xgboost(X_train, y_train) if name == "XGBoost" and tune_xgb else builder()
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        log.info(f"  {name} done ✓")
    joblib.dump(fitted, OUTPUT_DIR / "models.pkl")
    log.info(f"  Models saved ✓")
    return fitted


def load_models():
    return joblib.load(OUTPUT_DIR / "models.pkl")
