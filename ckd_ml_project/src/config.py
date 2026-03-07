"""
config.py
=========
Central configuration for the CKD Machine Learning pipeline.

All tunable constants – paths, random seeds, dataset parameters, model
hyper-parameter search spaces, and logging setup – live here so that the
rest of the codebase stays free of hard-coded values.

Clinical specification
----------------------
- KNN imputation (k=5) for missing values; drop features >20% missing.
- Z-score (StandardScaler) normalisation on numeric features.
- SMOTE oversampling applied to the training set after splitting.
- PCA for dimensionality-reduction visualisation.
- RFECV / LASSO-based feature selection.
- 10-fold stratified cross-validation scored on Recall (sensitivity).
- Models: XGBoost, Extra Trees, LightGBM, Linear SVM (L2).
"""

from __future__ import annotations

import logging
from pathlib import Path

# ─── Directory layout ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # ckd_ml_project/
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure directories exist at import time
for _directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    _directory.mkdir(parents=True, exist_ok=True)

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ─── Dataset ─────────────────────────────────────────────────────────────────
TARGET_COLUMN: str = "class"
MISSING_THRESHOLD: float = 0.20   # drop features with > 20 % missing values
KNN_NEIGHBORS: int = 5            # neighbours used by KNNImputer

# ─── High-priority clinical features (used for weighting in feature analysis) ─
HIGH_PRIORITY_FEATURES: list[str] = [
    "hemoglobin",
    "specific_gravity",
    "albumin",
    "hypertension",
    "diabetes_mellitus",
]

# ─── Data split ratios ───────────────────────────────────────────────────────
TRAIN_SIZE: float = 0.70          # 70 % for training
VALIDATION_SIZE: float = 0.50     # 50 % of the remaining 30 % → 15 % overall
TEST_SIZE: float = 0.50           # 50 % of the remaining 30 % → 15 % overall

# ─── Cross-validation ────────────────────────────────────────────────────────
CV_FOLDS: int = 10                # 10-fold CV as required by specification
CV_SCORING: str = "recall"        # Prioritise sensitivity to minimise false negatives

# ─── Feature selection ───────────────────────────────────────────────────────
# LASSO (L1) regularisation strength for feature selection step
LASSO_C: float = 0.1
# Number of PCA components to retain for the 2-D visualisation scatter plot
PCA_COMPONENTS_VIZ: int = 2

# ─── Hyper-parameter search grids ────────────────────────────────────────────
PARAM_GRIDS: dict = {
    # Linear SVM with L2 penalty (robust baseline for structured clinical data)
    "LinearSVM": {
        "C": [0.01, 0.1, 1, 10, 100],
    },
    # Extra Trees – fast, low-variance ensemble
    "ExtraTrees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5],
    },
    # XGBoost – state-of-the-art gradient boosting
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    # LightGBM – fast gradient boosting, strong on tabular clinical data
    "LightGBM": {
        "n_estimators": [100, 200],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    },
}

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "pipeline.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with both console and rotating file handlers.

    Each logger is configured once; subsequent calls return the same instance
    without adding duplicate handlers.

    Parameters
    ----------
    name : str
        Dotted module name, e.g. ``src.preprocess``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
