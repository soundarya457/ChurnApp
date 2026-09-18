"""
main.py — Full pipeline orchestrator.
Usage:
  python main.py                  # uses latest CSV in data/
  python main.py --csv path.csv   # explicit file
  python main.py --tune           # Optuna XGBoost tuning
  python main.py --skip-ingest    # skip CSV load (reuse existing raw_data table)
"""
import argparse
import logging
from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd

from src.config import DB_URL, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(OUTPUT_DIR / "pipeline.log")],
)
log = logging.getLogger(__name__)


def main(csv_path=None, skip_ingest=False, tune=False):
    log.info("=" * 60)
    log.info("  CHURN ML PIPELINE — START")
    log.info("=" * 60)

    # 1 — Ingest
    if not skip_ingest:
        log.info("[1/6] Ingesting CSV → MySQL")
        from src import ingest
        ingest.run(csv_path)
    else:
        log.info("[1/6] Skipping ingest")
        # Still need to populate SCHEMA from raw_data
        from src import ingest as _ing
        from src.config import DB_URL, SCHEMA
        engine = create_engine(DB_URL)
        df_raw = pd.read_sql("SELECT * FROM raw_data LIMIT 500", engine)
        _ing.detect_schema(df_raw)

    # 2 — Preprocess
    log.info("[2/6] Feature engineering + SMOTE")
    from src import preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess.run()

    # Fetch client IDs for the test split
    from src.config import SCHEMA, TEST_SIZE, RANDOM_STATE
    from sklearn.model_selection import train_test_split
    engine  = create_engine(DB_URL)
    id_col  = SCHEMA["id_col"] or "rowid"
    if SCHEMA["id_col"]:
        all_ids = pd.read_sql(f"SELECT `{id_col}` FROM features ORDER BY `{id_col}`", engine)
        _, test_ids = train_test_split(all_ids[id_col].values, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    else:
        n = len(X_train) + len(X_test)
        _, test_ids = train_test_split(list(range(n)), test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 3 — Train
    log.info("[3/6] Training models")
    from src import train
    fitted = train.train_all(X_train, y_train, tune_xgb=tune)

    # 4 — Evaluate
    log.info("[4/6] Evaluating models")
    from src import evaluate
    all_metrics, cv, best_name, best_model = evaluate.run(fitted, X_train, y_train, X_test, y_test)

    # 5 — Explain
    log.info("[5/6] SHAP + LIME")
    from src import explain
    shap_df, imp_df = explain.run(best_name, best_model, X_train, X_test, feature_names, test_ids)

    # 6 — Predict & store
    log.info("[6/6] Writing predictions")
    from src import predict
    pred_df = predict.run(best_name, best_model, X_test, y_test, test_ids, shap_df)

    best_m = next(m for m in all_metrics if m["model"] == best_name)
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Best model  : {best_name}")
    log.info(f"  ROC-AUC     : {best_m['roc_auc']}")
    log.info(f"  F1          : {best_m['f1']}")
    log.info(f"  Precision   : {best_m['precision']}")
    log.info(f"  Recall      : {best_m['recall']}")
    log.info(f"  Scored      : {len(pred_df):,} customers")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         type=str,  default=None)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--tune",        action="store_true")
    args = parser.parse_args()
    main(csv_path=Path(args.csv) if args.csv else None,
         skip_ingest=args.skip_ingest, tune=args.tune)
