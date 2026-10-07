"""
ingest.py — Load any CSV → MySQL raw_data table.

Target detection is a *recommendation only*: the user must confirm the target
column and positive (churn) label before anything is trained.  Every row keeps
a stable `_source_row` identifier (its position in the uploaded file) that is
carried through preprocessing, prediction and database storage.
"""
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from src.config import DB_URL, SCHEMA, SOURCE_COL, MIN_CLASS_ROWS
import logging

log = logging.getLogger(__name__)


# ── ID detection ───────────────────────────────────────────────

ID_RE = re.compile(
    r"(^id$|clientnum|customerid|cust.*id|account.*id|member.*id|rowid|row_id)", re.I)
MAX_CATEGORIES = 30   # text columns with more distinct values (names, free text) are dropped


def detect_id_col(df: pd.DataFrame) -> str | None:
    """Return the column most likely to be a row identifier."""
    id_keywords = ID_RE
    for col in df.columns:
        if col == SOURCE_COL:
            continue
        if id_keywords.search(col):
            if df[col].nunique() / max(len(df), 1) > 0.90:
                return col
    # Integer column where every value is unique
    for col in df.select_dtypes(include="number").columns:
        if col != SOURCE_COL and df[col].nunique() == len(df):
            return col
    return None


# ── Target detection / validation ──────────────────────────────

def _norm(v) -> str:
    return re.sub(r"[\s\-]+", "_", str(v).strip().lower())


# Normalised label vocabularies (churn = positive class)
POSITIVE_LABELS = {
    "yes", "true", "1", "1.0", "y", "t", "churn", "churned", "attrited",
    "attrited_customer", "attrition", "exited", "left", "cancelled",
    "canceled", "closed", "lost", "inactive", "default", "defaulted",
    "fraud", "converted", "positive",
}
NEGATIVE_LABELS = {
    "no", "false", "0", "0.0", "n", "f", "existing", "existing_customer",
    "stay", "stayed", "active", "retained", "current", "loyal", "negative",
    "not_churned", "no_churn", "non_churn", "nonchurn", "not_attrited",
}

NAME_PATTERNS = [
    (100, re.compile(r"(churn|churned|attrition|attrited|exited|cancelled|canceled|"
                     r"customer.?status|subscription.?status)", re.I)),
    (90,  re.compile(r"(^|_)(left|default|fraud|converted)(_|$)", re.I)),
    (70,  re.compile(r"(target|outcome|response|label)", re.I)),
    (40,  re.compile(r"(^|_)(class)(_|$)", re.I)),
]
DEMOGRAPHIC = re.compile(
    r"^(gender|sex|geography|country|region|state|city|name|surname|"
    r"firstname|lastname|email|phone|address|zip|postal|senior.?citizen|"
    r"dependents?|partner|married|tenure|age)$", re.I
)


def _name_score(col) -> int:
    best = 0
    for score, pat in NAME_PATTERNS:
        if pat.search(str(col)):
            best = max(best, score)
    return best


def to_binary(series: pd.Series, positive) -> pd.Series:
    """1 where the (stripped, string-form) value equals the positive label."""
    return (series.astype(str).str.strip() == str(positive).strip()).astype(int)


def infer_positive_label(values, counts: pd.Series):
    """
    Return (label, recognized, reason) for a 2-valued column.
    `recognized` is True only when the labels match known churn vocabulary.
    """
    vals = [str(v).strip() for v in values]
    pos = [v for v in vals if _norm(v) in POSITIVE_LABELS]
    neg = [v for v in vals if _norm(v) in NEGATIVE_LABELS]

    if len(vals) == 2:
        if len(pos) == 1 and len(neg) <= 1 and (not neg or neg[0] != pos[0]):
            return pos[0], True, f"'{pos[0]}' matches a known churn label"
        if len(neg) == 1 and not pos:
            other = next(v for v in vals if v != neg[0])
            return other, True, f"'{neg[0]}' is a known non-churn label, so '{other}' is churn"
    minority = str(counts.sort_values().index[0]).strip()
    return minority, False, "labels not recognised — defaulting to the minority class"


def describe_target(df: pd.DataFrame, col: str) -> dict:
    """Summary of one column as a potential target."""
    s = df[col]
    n_null = int(s.isna().sum())
    stripped = s.dropna().astype(str).str.strip()
    counts = stripped.value_counts()
    n = len(df)
    info = {
        "column": col,
        "n_unique": int(len(counts)),
        "n_null": n_null,
        "values": [{"value": str(v), "count": int(c), "pct": round(c / max(n - n_null, 1) * 100, 1)}
                   for v, c in counts.items()][:10],
        "name_score": _name_score(col),
        "is_binary": len(counts) == 2,
        "suggested_positive": None,
        "positive_recognized": False,
        "positive_reason": "",
        "value_score": 0,
    }
    if len(counts) == 2:
        pos, recognized, reason = infer_positive_label(list(counts.index), counts)
        info.update(suggested_positive=pos, positive_recognized=recognized,
                    positive_reason=reason)
        vals_norm = {_norm(v) for v in counts.index}
        has_pos = bool(vals_norm & POSITIVE_LABELS)
        has_neg = bool(vals_norm & NEGATIVE_LABELS)
        info["value_score"] = 80 if (has_pos and has_neg) else (60 if recognized else 0)
    return info


def validate_target(df: pd.DataFrame, col: str, positive=None, id_col=None):
    """
    Validate a chosen target column (+ positive label).
    Returns (ok, errors, warnings, info).
    """
    errors, warnings = [], []
    if col not in df.columns:
        return False, [f"Column '{col}' does not exist in the file."], [], {}
    if col == SOURCE_COL or (id_col and col == id_col):
        errors.append("The ID column cannot be used as the target.")
    info = describe_target(df, col)

    if info["n_null"] > 0:
        errors.append(f"{info['n_null']:,} rows have a blank value in '{col}'. "
                      "Fill or remove them before training.")
    if info["n_unique"] != 2:
        errors.append(f"'{col}' has {info['n_unique']} distinct values; "
                      "the target must have exactly 2 (churn / not churn).")

    if not errors:
        labels = [v["value"] for v in info["values"]]
        if positive is not None and str(positive).strip() not in labels:
            errors.append(f"'{positive}' is not a value of '{col}' (found: {labels}).")
        else:
            pos = str(positive).strip() if positive is not None else info["suggested_positive"]
            counts = {v["value"]: v["count"] for v in info["values"]}
            n_pos, n_neg = counts[pos], sum(counts.values()) - counts[pos]
            info.update(positive=pos, n_positive=n_pos, n_negative=n_neg,
                        churn_rate=round(n_pos / (n_pos + n_neg) * 100, 2))
            if min(n_pos, n_neg) < MIN_CLASS_ROWS:
                errors.append(f"Smaller class has only {min(n_pos, n_neg)} rows; "
                              f"at least {MIN_CLASS_ROWS} are needed to split, balance and test.")
            elif min(n_pos, n_neg) / (n_pos + n_neg) < 0.05:
                warnings.append("Very imbalanced target (<5% minority). "
                                "Watch recall and precision, not accuracy.")
            if n_pos > n_neg:
                warnings.append(f"'{pos}' is the MAJORITY class ({info['churn_rate']}%). "
                                "Churn is usually the minority — double-check the positive label.")
            if not info["positive_recognized"] and positive is None:
                warnings.append("Label wording not recognised; positive label is a guess.")
    return (not errors), errors, warnings, info


def find_target_candidates(df: pd.DataFrame, id_col: str | None, max_n: int = 12):
    """
    Rank possible target columns.  Returns (candidates, recommendation).
    The recommendation is advisory: confidence is 'high' only when exactly one
    strongly-named column with recognised labels exists; otherwise 'ambiguous'
    or 'none'.  Either way the UI requires an explicit user confirmation.
    """
    cands = []
    for col in df.columns:
        if col == SOURCE_COL or (id_col and col == id_col):
            continue
        info = describe_target(df, col)
        if not info["is_binary"]:      # a target must have exactly 2 values — hide everything else
            continue
        # binary yes/no/0/1 flag with no name hint = weak fallback
        weak_flag = info["is_binary"] and info["value_score"] > 0
        score = max(info["name_score"], info["value_score"])
        if info["is_binary"] and DEMOGRAPHIC.match(str(col)) and info["name_score"] == 0:
            score = 0
        elif score == 0 and weak_flag:
            score = 1
        info["score"] = score + (5 if info["name_score"] and info["value_score"] else 0)
        ok, errs, warns, _ = validate_target(df, col, None, id_col)
        info.update(valid=ok, issues=errs, warnings=warns)
        cands.append(info)

    cands.sort(key=lambda c: (-int(c["valid"]), -c["score"], str(c["column"])))
    cands = cands[:max_n]

    rec = {"column": None, "positive": None, "confidence": "none", "reason": ""}
    valid = [c for c in cands if c["valid"]]
    if valid:
        top = valid[0]
        strong_others = [c for c in valid[1:] if c["score"] >= 90]
        confident = top["score"] >= 90 and top["positive_recognized"] and not strong_others
        no_name_match = top["name_score"] < 70
        rec.update(
            column=top["column"], positive=top["suggested_positive"],
            confidence="high" if confident else "ambiguous",
            reason=("Strong churn-style name and recognised labels."
                    if confident else
                    ("None of these columns is named like a churn label — this file may not contain "
                     "the target at all (e.g. a Kaggle test file). Only continue if one of them really is churn."
                     if no_name_match else
                     "Several plausible columns or unrecognised labels — please pick the correct one.")),
        )
    return cands, rec


# ── Feature type detection ─────────────────────────────────────

def _is_text(series: pd.Series) -> bool:
    """True for text/categorical columns (works for object, pandas 'string' and category dtypes)."""
    return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series))


def detect_col_types(df: pd.DataFrame, id_col: str | None, target_col: str | None):
    skip = {c for c in [id_col, target_col, SOURCE_COL] if c}
    cat_cols, num_cols, drop_cols = [], [], []

    for col in df.columns:
        if col in skip:
            continue
        n_unique = df[col].nunique()
        n_rows = len(df)

        if n_unique == n_rows and _is_text(df[col]):
            drop_cols.append(col)
            continue
        if n_unique <= 1:
            drop_cols.append(col)
            continue
        # Secondary identifier columns (e.g. CustomerId next to id) carry no signal
        if ID_RE.search(str(col)):
            log.info(f"  Excluding '{col}' from features: looks like an identifier")
            drop_cols.append(col)
            continue
        # High-cardinality text (surnames, free text) only adds noise
        if _is_text(df[col]) and n_unique > MAX_CATEGORIES:
            log.info(f"  Excluding '{col}' from features: {n_unique} distinct text values")
            drop_cols.append(col)
            continue
        # A second churn-style column is almost certainly a duplicate of the target
        if target_col and col != target_col and _name_score(col) >= 90:
            log.warning(f"  Excluding '{col}' from features: looks like another churn label (leakage risk)")
            drop_cols.append(col)
            continue

        if _is_text(df[col]):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    return cat_cols, num_cols, drop_cols


def detect_schema(df: pd.DataFrame, target_col: str, positive_label: str):
    """Populate SCHEMA for a *user-confirmed* target column + positive label."""
    id_col = detect_id_col(df)
    ok, errors, warnings, _ = validate_target(df, target_col, positive_label, id_col)
    if not ok:
        raise ValueError("Invalid target selection: " + " ".join(errors))
    for w in warnings:
        log.warning(f"  {w}")
    cat_cols, num_cols, drop_cols = detect_col_types(df, id_col, target_col)

    SCHEMA["id_col"]           = id_col
    SCHEMA["target_col"]       = target_col
    SCHEMA["target_positive"]  = str(positive_label).strip()
    SCHEMA["categorical_cols"] = cat_cols
    SCHEMA["numerical_cols"]   = num_cols
    SCHEMA["drop_cols"]        = drop_cols

    log.info(f"  ID col      : {id_col}")
    log.info(f"  Target col  : {target_col}  (positive='{positive_label}')")
    log.info(f"  Categorical : {cat_cols}")
    log.info(f"  Numerical   : {num_cols}")
    log.info(f"  Drop        : {drop_cols}")


# ── MySQL helpers ──────────────────────────────────────────────

def _pandas_dtype_to_sql(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):   return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):     return "DOUBLE"
    if pd.api.types.is_bool_dtype(dtype):      return "TINYINT"
    return "VARCHAR(255)"


def create_raw_table(df: pd.DataFrame, engine):
    """Drop & recreate raw_data with primary key (required by Aiven)."""
    id_col = SCHEMA["id_col"]

    # Add synthetic PK if no ID col
    if not id_col or id_col not in df.columns:
        df.insert(0, "_row_id", range(1, len(df) + 1))
        SCHEMA["id_col"] = "_row_id"
        id_col = "_row_id"

    col_defs = []
    for col in df.columns:
        safe     = f"`{col}`"
        sql_type = _pandas_dtype_to_sql(df[col].dtype)
        if col == id_col:
            sql_type = "VARCHAR(100)" if sql_type == "VARCHAR(255)" else sql_type
            col_defs.append(f"  {safe} {sql_type} PRIMARY KEY")
        else:
            col_defs.append(f"  {safe} {sql_type}")

    ddl = ("DROP TABLE IF EXISTS raw_data;\n"
           "CREATE TABLE raw_data (\n" +
           ",\n".join(col_defs) + "\n);")

    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
    log.info("  raw_data table created ✓")


# ── Main ───────────────────────────────────────────────────────

def load_csv(csv_path) -> pd.DataFrame:
    log.info(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"  Shape: {df.shape}")
    return df


def count_duplicate_ids(df: pd.DataFrame, id_col: str | None) -> int:
    if not id_col or id_col not in df.columns:
        return 0
    return int(df[id_col].astype(str).str.strip().duplicated().sum())


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Clean IDs.  Duplicate IDs are removed (first kept) and *reported*."""
    nulls = df.isnull().sum()
    if nulls.any():
        log.warning(f"Nulls: {nulls[nulls > 0].to_dict()}")
    id_col = SCHEMA["id_col"]
    if id_col and id_col in df.columns:
        if df[id_col].isna().any():
            raise ValueError(f"ID column '{id_col}' has {int(df[id_col].isna().sum())} blank values.")
        df[id_col] = df[id_col].astype(str).str.strip()
        dupes = int(df.duplicated(subset=[id_col]).sum())
        if dupes:
            df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
            log.warning(f"  Removed {dupes} rows with a duplicate {id_col} (first occurrence kept)")
    return df


def to_mysql(df: pd.DataFrame, engine):
    # Keep ALL original columns in raw_data. Columns unsuitable for ML are
    # excluded later by preprocess.py, but must remain available for the
    # customer-detail modal and complete CSV export.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw_data"))

    df.to_sql("raw_data", con=engine, if_exists="append",
              index=False, chunksize=500, method="multi")
    log.info(f"  Inserted {len(df):,} rows into raw_data ✓")


def run(csv_path, target_col: str, positive_label: str):
    """Ingest a CSV using a target column the user has already confirmed."""
    df = load_csv(csv_path)

    # MySQL column-name limit
    long_cols = [c for c in df.columns if len(c) > 64]
    if long_cols:
        log.info(f"  Dropping {len(long_cols)} cols with names > 64 chars")
        df = df.drop(columns=long_cols)
    if SOURCE_COL in df.columns:
        df = df.drop(columns=[SOURCE_COL])

    detect_schema(df, target_col, positive_label)
    df = validate(df)
    # Stable row identifier = position in the (de-duplicated) upload
    df[SOURCE_COL] = np.arange(len(df), dtype=np.int64)

    engine = create_engine(DB_URL, echo=False)
    create_raw_table(df, engine)
    to_mysql(df, engine)
    log.info("Ingestion complete.")
    return df


def load_schema_from_db(target_col: str, positive_label: str):
    """For --skip-ingest: rebuild SCHEMA from the existing raw_data table."""
    engine = create_engine(DB_URL, echo=False)
    df = pd.read_sql("SELECT * FROM raw_data", engine)
    detect_schema(df, target_col, positive_label)
    if SCHEMA["id_col"] is None:
        SCHEMA["id_col"] = "_row_id" if "_row_id" in df.columns else None
