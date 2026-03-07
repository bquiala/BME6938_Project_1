"""
preprocess.py
=============
Full data-preprocessing pipeline for the CKD ML project.

Pipeline stages
---------------
1. Rename raw ARFF attribute names to human-readable column names.
2. Replace placeholder missing-value tokens (``?``, ``\\t?``, ``''``).
3. Standardise and binarise the target column (1 = CKD, 0 = not CKD).
4. Drop features whose missing-value rate exceeds ``MISSING_THRESHOLD``.
5. Label-encode categorical (non-numeric) columns.
6. Convert all columns to numeric dtype.
7. KNN-impute remaining NaN values.
8. Stratified 70 / 15 / 15 train / validation / test split.
9. Standardise features with ``StandardScaler`` (fit on train only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    KNN_NEIGHBORS,
    MISSING_THRESHOLD,
    RANDOM_STATE,
    TARGET_COLUMN,
    TRAIN_SIZE,
    TEST_SIZE,
    get_logger,
)

logger = get_logger(__name__)

# ─── Column-name mapping ─────────────────────────────────────────────────────
# Maps the terse UCI/ARFF attribute names to more readable identifiers.
COLUMN_RENAME_MAP: dict[str, str] = {
    "age":   "age",
    "bp":    "blood_pressure",
    "sg":    "specific_gravity",
    "al":    "albumin",
    "su":    "sugar",
    "rbc":   "red_blood_cells",
    "pc":    "pus_cell",
    "pcc":   "pus_cell_clumps",
    "ba":    "bacteria",
    "bgr":   "blood_glucose_random",
    "bu":    "blood_urea",
    "sc":    "serum_creatinine",
    "sod":   "sodium",
    "pot":   "potassium",
    "hemo":  "hemoglobin",
    "pcv":   "packed_cell_volume",
    "wc":    "white_blood_cell_count",
    "rc":    "red_blood_cell_count",
    "htn":   "hypertension",
    "dm":    "diabetes_mellitus",
    "cad":   "coronary_artery_disease",
    "appet": "appetite",
    "pe":    "pedal_edema",
    "ane":   "anemia",
    "class": "class",
}

# Known positive / negative CKD label variants (after lowercasing & stripping)
_CKD_POSITIVE = {"ckd", "ckd\t", "1", "yes", "1.0"}
_CKD_NEGATIVE  = {"notckd", "notckd\t", "0", "no", "0.0"}


# ─── Private helpers ─────────────────────────────────────────────────────────

def _standardise_target(series: pd.Series) -> pd.Series:
    """
    Map raw target strings to integer labels.

    Returns
    -------
    pd.Series
        1 for CKD-positive, 0 for CKD-negative, pd.NA for unrecognised values.
    """
    mapping: dict = {}
    for val in series.dropna().unique():
        clean = str(val).strip().lower()
        if clean in _CKD_POSITIVE:
            mapping[val] = 1
        elif clean in _CKD_NEGATIVE:
            mapping[val] = 0
        else:
            logger.warning("Unrecognised target value '%s' – will be dropped.", val)
            mapping[val] = pd.NA
    return series.map(mapping)


def _drop_high_missing(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove columns whose missing-value fraction exceeds *threshold*.

    Returns
    -------
    (filtered DataFrame, list of dropped column names)
    """
    missing_rate = df.isnull().mean()
    drop_cols = missing_rate[missing_rate > threshold].index.tolist()
    if drop_cols:
        logger.info(
            "Dropping %d feature(s) with >%.0f%% missing: %s",
            len(drop_cols), threshold * 100, drop_cols,
        )
        df = df.drop(columns=drop_cols)
    else:
        logger.info("No features exceeded the %.0f%% missing threshold.", threshold * 100)
    return df, drop_cols


def _encode_categoricals(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode every non-numeric column (the target column is excluded).

    Whitespace-only strings are treated as missing values (``np.nan``) before
    fitting the encoder so that they do not become a spurious category.

    Returns
    -------
    (transformed DataFrame, dict mapping column name → fitted LabelEncoder)
    """
    encoders: dict[str, LabelEncoder] = {}

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == TARGET_COLUMN:
            continue

        # Treat whitespace-only strings as NaN
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

        non_null_mask = df[col].notna()
        if not non_null_mask.any():
            logger.warning("Column '%s' is entirely NaN after cleaning – leaving as NaN.", col)
            df[col] = np.nan
            continue

        le = LabelEncoder()
        le.fit(df.loc[non_null_mask, col].unique())

        # Apply transform only to non-null values; leave NaN intact
        df[col] = df[col].apply(
            lambda x, _le=le: float(_le.transform([x])[0]) if pd.notna(x) else np.nan
        )
        encoders[col] = le

    return df, encoders


# ─── Public API ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> dict:
    """
    Execute the complete preprocessing pipeline on a raw CKD DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset loaded from an ARFF file (output of ``data_loader.load_arff``).

    Returns
    -------
    dict
        Keys
        ----
        ``X_train``, ``X_val``, ``X_test``   – scaled numpy feature matrices
        ``y_train``, ``y_val``, ``y_test``   – integer label arrays
        ``feature_names``                    – list of feature column names
        ``scaler``                           – fitted ``StandardScaler``
        ``encoders``                         – dict of fitted ``LabelEncoder`` objects
        ``dropped_features``                 – columns removed for high missingness
        ``df_clean``                         – cleaned & imputed DataFrame (pre-scaling)
        ``imputer``                          – fitted ``KNNImputer``
    """
    logger.info("─── Preprocessing pipeline start ───")

    # ── Step 1: Rename columns ───────────────────────────────────────────────
    rename_active = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_active)
    logger.info("Columns after rename: %s", df.columns.tolist())

    # ── Step 2: Replace placeholder missing-value tokens ────────────────────
    df.replace({"?": np.nan, "\t?": np.nan, "": np.nan}, inplace=True)

    # ── Step 3: Standardise target ───────────────────────────────────────────
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
    df[TARGET_COLUMN] = _standardise_target(df[TARGET_COLUMN])
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d rows with unrecognised target values.", n_dropped)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Separate target from features
    target = df.pop(TARGET_COLUMN).reset_index(drop=True)
    df = df.reset_index(drop=True)

    # ── Step 4: Drop high-missing features ──────────────────────────────────
    df, dropped_features = _drop_high_missing(df, MISSING_THRESHOLD)

    # ── Step 5: Encode categoricals ─────────────────────────────────────────
    df, encoders = _encode_categoricals(df)

    # ── Step 6: Coerce everything to numeric ─────────────────────────────────
    df = df.apply(pd.to_numeric, errors="coerce")

    # ── Step 7: KNN imputation ───────────────────────────────────────────────
    feature_names = df.columns.tolist()
    logger.info("Running KNN imputation (k=%d) on %d features …", KNN_NEIGHBORS, len(feature_names))
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    X_imputed = imputer.fit_transform(df)
    df_clean = pd.DataFrame(X_imputed, columns=feature_names)
    df_clean[TARGET_COLUMN] = target.values

    # ── Step 8: Stratified train / (val + test) split ────────────────────────
    # 70 % train, 15 % val, 15 % test
    X = df_clean[feature_names].values
    y = df_clean[TARGET_COLUMN].values

    logger.info(
        "Splitting dataset — train: 70%%, val: 15%%, test: 15%% (n=%d)", len(y)
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # Split the remaining 30 % evenly (50/50 → 15/15 of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=TEST_SIZE,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", len(y_train), len(y_val), len(y_test)
    )

    # ── Step 9: Feature scaling ───────────────────────────────────────────────
    logger.info("Applying StandardScaler (fit on train only) …")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    logger.info("─── Preprocessing pipeline complete ───")

    return {
        "X_train":          X_train,
        "X_val":            X_val,
        "X_test":           X_test,
        "y_train":          y_train,
        "y_val":            y_val,
        "y_test":           y_test,
        "feature_names":    feature_names,
        "scaler":           scaler,
        "encoders":         encoders,
        "dropped_features": dropped_features,
        "df_clean":         df_clean,
        "imputer":          imputer,
    }
